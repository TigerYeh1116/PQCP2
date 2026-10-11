"""Complete native same-O weight-four fibers with independently audited tables."""
import ctypes
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from .certified_profile_psd import cosine_upper_grid
from .cyclic_support import support_ring,poly_divmod,field_pow,field_mul
from .norm_trace_completion import _frobenius_columns,_frobenius,_lift_columns,_lift
from .norm_trace_native import field_norm_trace_roots


@lru_cache(maxsize=1)
def _library():
    root=Path(__file__).resolve().parents[1];source=root/'csrc/same_o_weight4.c';build=root/'build';build.mkdir(exist_ok=True)
    suffix='.dylib' if sys.platform=='darwin' else '.so'
    path=build/('same_o_weight4_'+hashlib.sha256(source.read_bytes()).hexdigest()[:16]+suffix)
    if not path.exists():
        fd,temporary=tempfile.mkstemp(dir=build,suffix=suffix);os.close(fd)
        try:
            subprocess.run(['cc','-O3','-std=c11','-Wall','-Wextra','-Werror','-fPIC',
                            '-dynamiclib' if sys.platform=='darwin' else '-shared',str(source),'-o',temporary],
                           check=True,capture_output=True,timeout=30)
            os.replace(temporary,path)
        finally:
            if os.path.exists(temporary):os.unlink(temporary)
    lib=ctypes.CDLL(str(path));u=ctypes.c_uint64;p=ctypes.POINTER(u)
    lib.pqcp_complete_same_o_weight4.argtypes=[u,p,ctypes.c_int,u,u,ctypes.c_int,ctypes.POINTER(ctypes.c_uint32),
        p,u,ctypes.POINTER(ctypes.c_int64),p,u,p,p,u,p]
    lib.pqcp_complete_same_o_weight4.restype=ctypes.c_int
    return lib


@lru_cache(maxsize=1)
def _common_tables():
    ring=support_ring(43);values=cosine_upper_grid(43)
    psd=(ctypes.c_int64*(21*21))(*(values[min(j*u%43,-j*u%43)] for j in range(1,22) for u in range(1,22)))
    residues=(ctypes.c_uint32*(3*43))(*(poly_divmod(1<<i,factor)[1] for factor in ring.factors[1:] for i in range(43)))
    return residues,psd


@lru_cache(maxsize=3)
def _fixed_subfield(factor):
    columns=_frobenius_columns(factor);stars=[0]*16384
    for u in range(1,16384):
        bit=u&-u;stars[u]=stars[u^bit]^columns[bit.bit_length()-1]
    fixed=tuple(u for u in range(16384) if stars[u]==u)
    if len(fixed)!=128:raise RuntimeError('F128 fixed subfield enumeration failed')
    return fixed


def anchored_root_lifts(plus,minus):
    ring=support_ring(43);partner=plus|minus;trace=(1<<1)|(1<<42)
    lifts=(ctypes.c_uint64*(6*16384))()
    ctypes.memset(lifts,255,ctypes.sizeof(lifts))
    for component,(factor,idempotent) in enumerate(zip(ring.factors[1:],ring.idempotents[1:])):
        w=poly_divmod(partner,factor)[1];h=poly_divmod(trace,factor)[1]
        if not h:raise RuntimeError('nonzero centered trace vanished in primitive-43 component')
        if not w:return None
        columns=_lift_columns(43,factor,idempotent);ws=_frobenius(w,factor)
        for n in _fixed_subfield(factor):
            roots=field_norm_trace_roots(n,h,w,factor)
            if len(roots) not in (0,2):raise RuntimeError('nonzero trace fiber must have zero or two roots')
            for index,u in enumerate(roots):
                us=_frobenius(u,factor)
                if field_mul(u,us,factor)!=n or field_mul(w,us,factor)^field_mul(ws,u,factor)!=h:
                    raise RuntimeError('weight-four anchored table failed independent Python field audit')
                lifts[(component*2+index)*16384+n]=_lift(u,columns)
    return lifts


def complete_weight4_c_fiber(certificate,*,capacity=256,seed_capacity=64):
    if certificate.get('g_weight',4)!=4 or certificate['negative_weight']!=15:
        raise ValueError('weight-four g and weight-15 C required')
    return _complete_sparse_c_fiber(certificate,capacity=capacity,seed_capacity=seed_capacity)


def complete_weight2_c_fiber(certificate,*,capacity=256,seed_capacity=64):
    if certificate.get('g_weight')!=2 or certificate['negative_weight']!=16:
        raise ValueError('weight-two g and weight-16 C required')
    return _complete_sparse_c_fiber(certificate,capacity=capacity,seed_capacity=seed_capacity)


def _complete_sparse_c_fiber(certificate,*,capacity,seed_capacity):
    """Every affine C, exact weight 15, every norm/trace O and integer target.

    May also operate on an explicitly selected affine subspace certificate.
    Its declared raw space always equals 2^len(basis), never the parent space.
    """
    if (certificate['d']!=43 or certificate['negative_weight'] not in (15,16) or type(capacity) is not int or capacity<1
            or type(seed_capacity) is not int or seed_capacity<1):
        raise ValueError('L43 complementary weight-15 C certificate and positive capacity required')
    if certificate['inconsistent']:return dict(complete=True,hits=(),odd_integer_models=(),stats={},raw_affine_assignments=0)
    basis=certificate['basis'];base=certificate['base'];plus=certificate['plus'];minus=certificate['minus']
    if (len(basis)>30 or any(type(v) is not int or not 0<=v<1<<43 or v&(plus|minus) for v in (base,)+tuple(basis))):
        raise ValueError('literal complementary affine base/kernel required')
    # Audit the base and every homogeneous column against the original
    # equations, including exact C-weight parity, before native enumeration.
    positions=certificate['positions'];m=len(positions)
    def packed(mask):return sum(((mask>>position)&1)<<i for i,position in enumerate(positions))
    packed_base=packed(base);packed_basis=tuple(map(packed,basis))
    pivots={}
    for value in packed_basis:
        while value:
            pivot=value.bit_length()-1
            if pivot not in pivots:pivots[pivot]=value;break
            value^=pivots[pivot]
        if not value:raise ValueError('affine C basis must be linearly independent')
    for row in certificate['equations']:
        coeff=row&((1<<m)-1)
        if (bin(coeff&packed_base).count('1')%2!=(row>>m)&1
                or any(bin(coeff&v).count('1')%2 for v in packed_basis)):
            raise ValueError('affine C table fails its original mod-four equations')
    lifts=anchored_root_lifts(plus,minus)
    if lifts is None:return dict(complete=True,hits=(),odd_integer_models=(),stats={'zero_partner_excluded':1},raw_affine_assignments=1<<len(basis))
    residues,psd=_common_tables();u=ctypes.c_uint64;columns=(u*len(basis))(*basis);stats=(u*11)();seed_stats=(u*2)()
    while True:
        output=(u*(2*capacity))()
        seed_output=(u*(3*seed_capacity))()
        status=_library().pqcp_complete_same_o_weight4(base,columns,len(basis),plus,minus,certificate['negative_weight'],residues,lifts,
                            support_ring(43).idempotents[0],psd,output,capacity,stats,seed_output,seed_capacity,seed_stats)
        if status or stats[0]!=1<<len(basis):raise RuntimeError(f'complete weight-four C enumeration/audit failed: {status}')
        if stats[9]==stats[10] and seed_stats[0]==seed_stats[1]:break
        capacity=max(2*capacity,int(stats[9]))
        seed_capacity=max(2*seed_capacity,int(seed_stats[0]))
    names=('affine_assignments','weight_matched_c','range_excluded','psd_excluded','norm_trace_empty',
           'literal_o_preimages','o_weight_excluded','odd_integer_excluded','even_integer_excluded','full_hits','stored_hits')
    hits=tuple((int(output[2*i]),int(output[2*i+1])) for i in range(stats[10]))
    if len(set(hits))!=len(hits):raise RuntimeError('duplicate literal C/O full models')
    odd_models=tuple(tuple(int(seed_output[3*i+j]) for j in range(3)) for i in range(seed_stats[1]))
    if len({(c,o) for c,o,energy in odd_models})!=len(odd_models):raise RuntimeError('duplicate odd-integer models')
    counters=dict(zip(names,map(int,stats)));counters['odd_integer_matched']=int(seed_stats[0])
    return dict(complete=True,hits=hits,odd_integer_models=odd_models,stats=counters,raw_affine_assignments=1<<len(basis))
