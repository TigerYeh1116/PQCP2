"""Single-row norm Hensel obstruction and complete mod-8 zero-field lifts.

Self-reciprocal F2 components are fields with star=Frobenius. Norm zero
forces the row component zero; consequently its integer norm vanishes mod4.
At mod8 the remaining row/2 component has a prescribed field norm. Enumerate
ALL its roots and solve their linear projections of the audited mod4 affine
sign space, including every dependent-column nullspace vector.
Never use this SINGLE-norm rule on a sum of two unknown rows.
"""
import ctypes
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from time import perf_counter

from .cyclic_support import support_ring,poly_divmod,_norm_options,field_pow
from .structured_energy import integer_periodic_autocorrelation_profile as pacf


def cyclic_product(a,b,modulus):
    d=len(a);out=[0]*d
    for i,x in enumerate(a):
        for j,y in enumerate(b):out[(i+j)%d]=(out[(i+j)%d]+x*y)%modulus
    return tuple(out)


def _planes(values):return tuple(sum(((v%8)>>j&1)<<i for i,v in enumerate(values)) for j in range(3))


@lru_cache(maxsize=4)
def lifted_idempotents(d):
    ring=support_ring(d);lifts=[]
    for mask in ring.idempotents:
        e=tuple((mask>>i)&1 for i in range(d));e4=cyclic_product(e,e,4);e8=cyclic_product(e4,e4,8)
        if (cyclic_product(e8,e8,8)!=e8 or any(e8[i]!=e8[-i] for i in range(1,d))
                or sum((v%2)<<i for i,v in enumerate(e8))!=mask):raise RuntimeError('idempotent mod8/star lift audit failed')
        lifts.append(e8)
    for i,e in enumerate(lifts):
        for j,f in enumerate(lifts):
            expected=e if i==j else (0,)*d
            if cyclic_product(e,f,8)!=expected:raise RuntimeError('lifted idempotent CRT orthogonality failed')
    if tuple(sum(e[i] for e in lifts)%8 for i in range(d))!=(1,)+(0,)*(d-1):
        raise RuntimeError('lifted idempotents do not sum to identity')
    return tuple(lifts)


@lru_cache(maxsize=1)
def _library():
    root=Path(__file__).resolve().parents[1];source=root/'csrc/two_adic_norm.c';build=root/'build';build.mkdir(exist_ok=True)
    suffix='.dylib' if sys.platform=='darwin' else '.so';path=build/('two_adic_norm_'+hashlib.sha256(source.read_bytes()).hexdigest()[:16]+suffix)
    if not path.exists():
        fd,tmp=tempfile.mkstemp(dir=build,suffix=suffix);os.close(fd)
        try:
            subprocess.run(['cc','-O3','-std=c11','-Wall','-Wextra','-Werror','-fPIC',
                '-dynamiclib' if sys.platform=='darwin' else '-shared',str(source),'-o',tmp],check=True,capture_output=True,timeout=30)
            os.replace(tmp,path)
        finally:
            if os.path.exists(tmp):os.unlink(tmp)
    lib=ctypes.CDLL(str(path));u=ctypes.c_uint64;p=ctypes.POINTER(u);i=ctypes.c_int
    lib.pqcp_project_mod8.argtypes=[p,p,i,p];lib.pqcp_project_mod8.restype=i
    lib.pqcp_lifted_unary_signs.argtypes=[u,u,p,p,i,i,u,p,i,p,i,p,i,i,i,ctypes.POINTER(ctypes.c_int32),u,p,p]
    lib.pqcp_lifted_unary_signs.restype=i
    return lib


def native_project(values,e,d):
    u=ctypes.c_uint64;out=(u*3)()
    if _library().pqcp_project_mod8((u*3)(*values),(u*3)(*_planes(e)),d,out):raise RuntimeError('native cyclic mod8 input audit failed')
    return tuple(map(int,out))


def verify_two_adic_certificate(c):
    profile=tuple(c['profile']);d=len(profile);i=c['component'];ring=support_ring(d)
    if not 1<=i<len(ring.factors) or c['factor']!=ring.factors[i]:return False
    e=lifted_idempotents(d)[i];actual=cyclic_product(profile,e,4)
    return (poly_divmod(sum((v%2)<<j for j,v in enumerate(profile)),ring.factors[i])[1]==0
            and any(actual) and all(v%2==0 for v in actual) and actual==tuple(c['projected_mod4']))


@lru_cache(maxsize=64)
def prepare_two_adic_profile(profile):
    profile=tuple(profile);d=len(profile);ring=support_ring(d)
    if (any(type(v) is not int or not -d<=v<=d for v in profile)
            or any(profile[i]!=profile[-i] for i in range(1,d))):raise ValueError('supported single signed integer norm profile required')
    planes=_planes(profile);components=[]
    for i,f in enumerate(ring.factors[1:],1):
        if poly_divmod(planes[0],f)[1]:continue
        e=lifted_idempotents(d)[i];projected=native_project(planes,e,d)
        if projected[0]:raise RuntimeError('native projected low plane differs from exact F2 factor residue')
        if projected[1]:
            values=tuple(2*((projected[1]>>j)&1) for j in range(d))
            proof=dict(profile=profile,component=i,factor=f,projected_mod4=values,
                       proof='Single norm zero in a self-reciprocal residue field must vanish in its lifted component modulo four.')
            if not verify_two_adic_certificate(proof):raise RuntimeError('mod4 norm obstruction failed independent integer convolution audit')
            return dict(obstruction=proof,components=())
        desired=poly_divmod(projected[2],f)[1]
        if field_pow(desired,1<<(ring.degree//2),f)!=desired:raise RuntimeError('lifted norm quotient outside fixed field')
        components.append((i,desired))
    return dict(obstruction=None,components=tuple(components))


def two_adic_norm_obstruction(profile):return prepare_two_adic_profile(tuple(profile))['obstruction']


def prefer_lifted_unary(plan,profile):
    data=prepare_two_adic_profile(tuple(profile))
    if not data['components']:return False
    q=plan['meta'][2] if plan['selected']==-1 else plan['meta'][4]
    if q<10:return False
    ring=support_ring(plan['d']);roots=1
    for index,desired in data['components']:roots*=((1<<(ring.degree//2))+1) if desired else 1
    # Scheduling uses explicit finite work counts, never a mathematical
    # exclusion. The original complete kernel remains the fallback.
    old_steps=1<<max(0,q-6) if 12<=q<=24 else 1<<q
    return roots*(1+min(q,len(data['components'])*ring.degree))<100*old_steps


def lifted_unary_signs(plan,profile,negative,budget,*,minimum_free=10):
    """Complete zero-component norm-root projection, or None for fallback.

    Requires an unconditioned ONE-row plan. Budget limits integer candidate
    checks, not the many affine assignments excluded by proved norm roots.
    """
    empty_conditioned=(plan['selected']==1 and plan['meta'][5]==0 and not plan['masks'][1])
    if plan['masks'][1] or (plan['selected']!=-1 and not empty_conditioned):
        raise ValueError('SINGLE unknown row with raw or empty-side conditioning required, never two unknown norms')
    if tuple(plan['wanted'])!=tuple(profile) or tuple(plan['negative'])!=(negative,0):raise ValueError('plan/profile/negative counts mismatch')
    q=plan['meta'][4] if empty_conditioned else plan['meta'][2];d=plan['d'];data=prepare_two_adic_profile(tuple(profile))
    if not plan['consistent'] or data['obstruction'] is not None:return None
    if not data['components'] or q<minimum_free:return None
    tick=perf_counter();ring=support_ring(d);base=int(plan['base'][0]);support=plan['masks'][0]
    toggles=tuple(int(plan['toggles'][2*j]) for j in range(q));columns=[0]*q;origin=0;groups=[]
    for at,(index,desired) in enumerate(data['components']):
        f=ring.factors[index];e=lifted_idempotents(d)[index];part=native_project((support,0,0),e,d)
        if part[0]:raise RuntimeError('norm-zero support has a nonzero residue component')
        offset=at*ring.degree
        origin^=(poly_divmod(base,f)[1]^poly_divmod(part[1],f)[1])<<offset
        for j,mask in enumerate(toggles):columns[j]^=poly_divmod(mask,f)[1]<<offset
        roots=_norm_options(d,ring.degree,f,ring.idempotents[index],ring.norm_generators[index-1],desired)
        groups.append(tuple(poly_divmod(v,f)[1]<<offset for v in roots))
    groups+= [(0,)]*(3-len(groups));u=ctypes.c_uint64;i=ctypes.c_int32
    arrays=[(u*len(g))(*g) for g in groups];solution=u();stats=(u*6)()
    status=_library().pqcp_lifted_unary_signs(support,base,(u*q)(*toggles),(u*q)(*columns),q,
        len(data['components'])*ring.degree,origin,arrays[0],len(groups[0]),arrays[1],len(groups[1]),arrays[2],len(groups[2]),
        d,negative,(i*d)(*profile),budget,ctypes.byref(solution),stats)
    if status not in (0,1,2):raise RuntimeError('lifted norm root/linear/nullspace enumeration failed')
    words=None
    if status==1:
        row=tuple(0 if not support>>j&1 else -1 if solution.value>>j&1 else 1 for j in range(d))
        if tuple(pacf(row))!=tuple(profile) or row.count(-1)!=negative:raise RuntimeError('lifted signed witness failed independent ALL-lag integer audit')
        words=row,(0,)*d
    return dict(status=('UNSAT_FIXED_SUPPORT','SAT_COMPRESSED','UNKNOWN')[status],words=words,
        rank=plan['meta'][1],free_variables=q,assignments=int(stats[2]),count_matches=int(stats[3]),
        correlation_checks=int(stats[4]),covered_affine_assignments=int(stats[5]),
        lifted_root_combinations=int(stats[0]),compatible_linear_cosets=int(stats[1]),
        zero_norm_components=len(data['components']),two_adic_backend=True,
        construction_seconds=plan['construction_seconds'],solve_seconds=perf_counter()-tick,
        reason='candidate assignment cap' if status==2 else None)
