"""Certified joint norm/cross-spectrum obstruction for a literal unknown half.

For D=UW*+U*W in the integer cyclic ring, Fourier(D)=2Re(Uhat conj(What)).
Thus Fourier(D)^2 <= 4 Fourier(N_U) Fourier(N_W). Rational outward bounds
certify violations with integers only. Passing is not a feasibility proof.
"""
import ctypes
from fractions import Fraction
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from .certified_profile_psd import SCALE,rational_pi_bounds,cosine_upper_grid
from .sds_norm_completion import intersection_profile


@lru_cache(maxsize=4)
def cosine_interval_grid(d):
    upper=cosine_upper_grid(d);pi_lower,pi_upper=rational_pi_bounds();lower=[]
    for phase in range(d//2+1):
        angle=2*pi_upper*phase/d
        if phase and angle>=pi_lower:raise RuntimeError('cosine lower-bound angle outside proven decreasing interval')
        term=Fraction(1);partial=term
        for n in range(1,18):
            term=-term*angle*angle/((2*n-1)*(2*n));partial+=term
        value=partial*SCALE;lower.append(value.numerator//value.denominator)
    if any(lo>hi for lo,hi in zip(lower,upper)):raise RuntimeError('outward cosine interval inverted')
    return tuple(lower),upper


@lru_cache(maxsize=1)
def _library():
    root=Path(__file__).resolve().parents[1];source=root/'csrc/joint_gram.c';build=root/'build';build.mkdir(exist_ok=True)
    suffix='.dylib' if sys.platform=='darwin' else '.so'
    path=build/('joint_gram_'+hashlib.sha256(source.read_bytes()).hexdigest()[:16]+suffix)
    if not path.exists():
        fd,temporary=tempfile.mkstemp(dir=build,suffix=suffix);os.close(fd)
        try:
            subprocess.run(['cc','-O3','-std=c11','-Wall','-Wextra','-Werror','-fPIC',
                            '-dynamiclib' if sys.platform=='darwin' else '-shared',str(source),'-o',temporary],
                           check=True,capture_output=True,timeout=30)
            os.replace(temporary,path)
        finally:
            if os.path.exists(temporary):os.unlink(temporary)
    lib=ctypes.CDLL(str(path));p32=ctypes.POINTER(ctypes.c_int32);p64=ctypes.POINTER(ctypes.c_int64);pu=ctypes.POINTER(ctypes.c_uint64)
    lib.pqcp_joint_gram_targets.argtypes=[p32,p32,p32,p64,p64,p32,p32,ctypes.c_int,ctypes.c_int,pu,pu]
    lib.pqcp_joint_gram_targets.restype=ctypes.c_int
    lib.pqcp_profile_psd.argtypes=[p32,p64,ctypes.c_int,p32,p64]
    lib.pqcp_profile_psd.restype=ctypes.c_int
    lib.pqcp_sparse_profile_psd.argtypes=[p32,p32,p32,ctypes.c_int,p64,ctypes.c_int,p32,p64]
    lib.pqcp_sparse_profile_psd.restype=ctypes.c_int
    return lib


@lru_cache(maxsize=4)
def _cosine_arrays(d):
    lo,hi=cosine_interval_grid(d);a=ctypes.c_int64
    return (a*len(lo))(*lo),(a*len(hi))(*hi)


@lru_cache(maxsize=4)
def _upper_array(d):
    values=cosine_upper_grid(d)
    return (ctypes.c_int64*len(values))(*values)


def native_profile_psd(profile):
    """Internal bounded integral dot product; caller audits the numerator."""
    d=len(profile);array=(ctypes.c_int32*d)(*profile);frequency=ctypes.c_int32();numerator=ctypes.c_int64()
    status=_library().pqcp_profile_psd(array,_upper_array(d),d,ctypes.byref(frequency),ctypes.byref(numerator))
    if status:raise RuntimeError('native PSD input/bound audit failed')
    return (frequency.value,numerator.value) if frequency.value else None


def native_sparse_profile_psd(base,offsets):
    """Bounded internal affine batch; public caller checks every certificate."""
    d=len(base);i=ctypes.c_int32;n=len(offsets)
    frequencies=(i*n)();numerators=(ctypes.c_int64*n)()
    status=_library().pqcp_sparse_profile_psd((i*d)(*base),(i*n)(*(t for t,e in offsets)),
        (i*n)(*(e for t,e in offsets)),n,_upper_array(d),d,frequencies,numerators)
    if status:raise RuntimeError('native sparse PSD input/range audit failed')
    return tuple((int(f),int(v)) if f else None for f,v in zip(frequencies,numerators))


@lru_cache(maxsize=4096)
def _partner_profile(partner,d):return intersection_profile(partner,d)


def _tau(k,d):return min(((k-d)//2)%d,(-((k-d)//2))%d) if k%2 else -1


def _fourier_interval(profile,frequency):
    d=len(profile);lo,hi=cosine_interval_grid(d);lower=upper=profile[0]*SCALE
    for u in range(1,d//2+1):
        phase=min(frequency*u%d,-frequency*u%d);v=2*profile[u]
        lower+=v*(lo[phase] if v>=0 else hi[phase]);upper+=v*(hi[phase] if v>=0 else lo[phase])
    return lower,upper


def verify_joint_gram_certificate(certificate):
    """Independent Python integer reconstruction of every native certificate."""
    c=certificate;wanted=tuple(c['wanted']);d=len(wanted);frequency=c['frequency']
    upper_u=_fourier_interval(wanted,frequency)[1]
    if c['reason']=='NEGATIVE_UNKNOWN_PSD':
        value=upper_u;denominator=SCALE
    elif c['reason']=='NEGATIVE_JOINT_GRAM_UPPER':
        upper_w=_fourier_interval(_partner_profile(c['partner'],d),frequency)[1]
        lo,hi=_fourier_interval(tuple(c['base_cross']),frequency);k,eta=c['target'];tau=_tau(k,d)
        if tau>=0:
            lower_cos,upper_cos=cosine_interval_grid(d);phase=min(frequency*tau%d,-frequency*tau%d)
            lo+=2*eta*(lower_cos[phase] if eta>0 else upper_cos[phase])
            hi+=2*eta*(upper_cos[phase] if eta>0 else lower_cos[phase])
        minimum=lo if lo>0 else (-hi if hi<0 else 0)
        value=4*upper_u*upper_w-minimum*minimum;denominator=SCALE*SCALE
    else:return False
    return value<0 and value==c['upper_bound_numerator'] and denominator==c['denominator']


def joint_gram_target_filter(wanted,partner,base_cross,targets):
    """ALL supplied targets, with sparse odd delta added to signed base_cross.

    Odd k contributes eta at centered +/-((k-d)/2); even k contributes none.
    Only strict rational upper-bound violations may exclude a target.
    """
    wanted=tuple(wanted);base_cross=tuple(base_cross);targets=tuple(targets);d=len(wanted)
    if (d not in (3,5,11,43) or len(base_cross)!=d or type(partner) is not int or not 0<=partner<1<<d
            or any(type(v) is not int or not 0<=v<=d for v in wanted)
            or any(type(v) is not int or not -4*d<=v<=4*d for v in base_cross)
            or any(wanted[u]!=wanted[-u] or base_cross[u]!=base_cross[-u] for u in range(1,d))
            or any(type(k) is not int or not 1<=k<d or type(eta) is not int or eta not in (-1,1) for k,eta in targets)):
        raise ValueError('supported symmetric integral norm/cross profiles, literal partner and strict targets required')
    i=ctypes.c_int32;u=ctypes.c_uint64
    nu=(i*d)(*wanted);nw=(i*d)(*_partner_profile(partner,d));cross=(i*d)(*base_cross)
    lo,hi=_cosine_arrays(d);allowed=[];certificates=[]
    for start in range(0,len(targets),64):
        group=targets[start:start+64];n=len(group);taus=(i*n)(*(_tau(k,d) for k,eta in group));signs=(i*n)(*(eta for k,eta in group))
        retained=u();raw=(u*(4*n))()
        status=_library().pqcp_joint_gram_targets(nu,nw,cross,lo,hi,taus,signs,n,d,ctypes.byref(retained),raw)
        if status:raise RuntimeError(f'native joint Gram interval audit failed: {status}')
        for index,target in enumerate(group):
            if retained.value>>index&1:allowed.append(target);continue
            code,frequency,low,high=map(int,raw[4*index:4*index+4])
            if code not in (1,2):raise RuntimeError('target removed without a strict integer certificate')
            if high>=1<<63:high-=1<<64
            numerator=(high<<64)+low
            if numerator>=0:raise RuntimeError('nonnegative joint Gram bound cannot prune')
            certificates.append(dict(reason='NEGATIVE_UNKNOWN_PSD' if code==1 else 'NEGATIVE_JOINT_GRAM_UPPER',
                target=target,frequency=frequency,upper_bound_numerator=numerator,
                denominator=SCALE if code==1 else SCALE*SCALE,wanted=wanted,partner=partner,base_cross=base_cross))
    return dict(complete=True,allowed_targets=tuple(allowed),certificates=certificates,
                excluded_targets=len(certificates),tested_targets=len(targets),
                scope='Certified necessary joint Fourier Gram condition only; retained targets are not proved feasible.')
