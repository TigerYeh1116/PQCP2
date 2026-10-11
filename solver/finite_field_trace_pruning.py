"""Odd-characteristic norm/trace obstruction with complete zero-partner fallback.

Let z=U W*, so trace(z)=h and norm(z)=n Norm(W). For p==-1 mod d,
star=Frobenius in F_(p^2). A nonzero-square discriminant cannot produce
two conjugate roots: its roots lie in F_p and would each have trace 2z.
Zero discriminant and nonsquares pass. This is a necessary condition only.
"""
import ctypes
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from .prime_norm_obstruction import real_roots_mod_p,_traces,_psi_and_derivative,is_prime
from .sds_norm_completion import intersection_profile

DEFAULT_PRIMES={3:5,5:19,11:43,43:257}


@lru_cache(maxsize=8)
def _tables(d,p):
    roots=real_roots_mod_p(d,p);rows=tuple((1,)+_traces(t,d//2,p) for t in roots)
    u=ctypes.c_uint32;b=ctypes.c_uint8;squares=(b*p)()
    for value in range(1,p):squares[value*value%p]=1
    return roots,rows,(u*(len(rows)*(d//2+1)))(*(v for row in rows for v in row)),squares


@lru_cache(maxsize=1)
def _library():
    root=Path(__file__).resolve().parents[1];source=root/'csrc/finite_field_trace.c';build=root/'build';build.mkdir(exist_ok=True)
    suffix='.dylib' if sys.platform=='darwin' else '.so';path=build/('finite_trace_'+hashlib.sha256(source.read_bytes()).hexdigest()[:16]+suffix)
    if not path.exists():
        fd,temporary=tempfile.mkstemp(dir=build,suffix=suffix);os.close(fd)
        try:
            subprocess.run(['cc','-O3','-std=c11','-Wall','-Wextra','-Werror','-fPIC',
                            '-dynamiclib' if sys.platform=='darwin' else '-shared',str(source),'-o',temporary],
                           check=True,capture_output=True,timeout=30);os.replace(temporary,path)
        finally:
            if os.path.exists(temporary):os.unlink(temporary)
    lib=ctypes.CDLL(str(path));i=ctypes.POINTER(ctypes.c_int32)
    lib.pqcp_finite_trace_targets.argtypes=[i,i,i,ctypes.POINTER(ctypes.c_uint32),ctypes.POINTER(ctypes.c_uint8),i,i,
        ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.POINTER(ctypes.c_uint64),i]
    lib.pqcp_finite_trace_targets.restype=ctypes.c_int
    lib.pqcp_finite_affine_targets.argtypes=lib.pqcp_finite_trace_targets.argtypes[:7]+[i]+lib.pqcp_finite_trace_targets.argtypes[7:]
    lib.pqcp_finite_affine_targets.restype=ctypes.c_int
    return lib


@lru_cache(maxsize=4096)
def _norm_profile(partner,d):return intersection_profile(partner,d)


def _tau(k,d):return min(((k-d)//2)%d,(-((k-d)//2))%d) if k%2 else -1


def verify_finite_trace_certificate(c):
    wanted=tuple(c['wanted']);base=tuple(c['base_cross']);d=len(wanted);p=c['prime'];root=c['real_root_mod_p']
    if d not in DEFAULT_PRIMES or not is_prime(p) or p<3 or (p+1)%d:return False
    if _psi_and_derivative(root,d//2,p)[0] or pow((root*root-4)%p,(p-1)//2,p)!=p-1:return False
    coefficients=(1,)+_traces(root,d//2,p)
    n=sum(v*a for v,a in zip(wanted[:d//2+1],coefficients))%p
    w=sum(v*a for v,a in zip(_norm_profile(c['partner'],d)[:d//2+1],coefficients))%p
    h=sum(v*a for v,a in zip(base[:d//2+1],coefficients))%p;k,eta=c['target'];tau=_tau(k,d)
    if tau>=0:h=(h+eta*coefficients[tau])%p
    delta=(h*h-4*n*w)%p
    if c['reason']=='ZERO_PARTNER_NONZERO_TRACE':valid=w==0 and h!=0
    elif c['reason']=='NONZERO_SQUARE_DISCRIMINANT':valid=w!=0 and delta!=0 and pow(delta,(p-1)//2,p)==1
    else:return False
    return valid and delta==c['discriminant_mod_p']


def finite_trace_target_filter(wanted,partner,base_cross,targets,*,prime=None):
    return _target_filter(wanted,partner,base_cross,targets,prime,False)


def finite_trace_affine_filter(norm_base,partner,base_cross,targets,*,prime=None):
    """Same necessary test, sharing norm/cross dots across all k/eta.

    Even k adds eta at +/-k/2 to the norm. Odd k instead adds the
    centered delta to the cross trace. Each proof includes the actual norm.
    """
    return _target_filter(norm_base,partner,base_cross,targets,prime,True)


def _target_filter(wanted,partner,base_cross,targets,prime,affine):
    wanted=tuple(wanted);base_cross=tuple(base_cross);targets=tuple(targets);d=len(wanted)
    if prime is None:prime=DEFAULT_PRIMES.get(d)
    if (d not in DEFAULT_PRIMES or len(base_cross)!=d or type(prime) is not int or not 3<=prime<=65000
            or type(partner) is not int or not 0<=partner<1<<d
            or any(type(v) is not int or not -d<=v<=(2*d if affine else d) for v in wanted)
            or any(type(v) is not int or not -4*d<=v<=4*d for v in base_cross)
            or any(wanted[u]!=wanted[-u] or base_cross[u]!=base_cross[-u] for u in range(1,d))
            or any(type(k) is not int or not 1<=k<d or type(eta) is not int or eta not in (-1,1) for k,eta in targets)):
        raise ValueError('supported symmetric norm/cross profiles, literal partner and strict integer targets required')
    roots,rows,coefficients,squares=_tables(d,prime);i=ctypes.c_int32;u=ctypes.c_uint64
    norm=(i*d)(*wanted);known=(i*d)(*_norm_profile(partner,d));base=(i*d)(*base_cross);allowed=[];certificates=[]
    for start in range(0,len(targets),64):
        group=targets[start:start+64];count=len(group);tau=(i*count)(*(_tau(k,d) for k,eta in group));eta=(i*count)(*(v for k,v in group))
        retained=u();proofs=(i*(3*count))()
        if affine:
            norm_tau=(i*count)(*(k//2 if not k%2 else -1 for k,e in group))
            status=_library().pqcp_finite_affine_targets(norm,known,base,coefficients,squares,tau,eta,norm_tau,count,d,prime,ctypes.byref(retained),proofs)
        else:
            status=_library().pqcp_finite_trace_targets(norm,known,base,coefficients,squares,tau,eta,count,d,prime,ctypes.byref(retained),proofs)
        if status:raise RuntimeError('finite-field trace input/arithmetic audit failed')
        for at,target in enumerate(group):
            if retained.value>>at&1:allowed.append(target);continue
            reason,row,delta=map(int,proofs[3*at:3*at+3])
            if reason not in (1,2) or not 0<=row<len(roots):raise RuntimeError('field target removed without a valid root certificate')
            actual=list(wanted);k,e=target
            if affine and not k%2:actual[k//2]+=e;actual[-(k//2)]+=e
            certificates.append(dict(reason='ZERO_PARTNER_NONZERO_TRACE' if reason==1 else 'NONZERO_SQUARE_DISCRIMINANT',
                prime=prime,real_root_mod_p=roots[row],discriminant_mod_p=delta,target=target,
                wanted=tuple(actual),partner=partner,base_cross=base_cross))
    return dict(complete=True,allowed_targets=tuple(allowed),certificates=certificates,tested_targets=len(targets),
                excluded_targets=len(certificates),scope='Necessary single-unknown norm AND anchored cross-trace over F_(p^2); '
                'nonsquare/zero discriminants retained, zero partner with zero trace not restricted.')
