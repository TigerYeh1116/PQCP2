"""p-adic obstruction for ONE integer cyclic norm, not a sum of two norms.

For an odd prime d and p == -1 mod d, primitive d roots lie in F_(p^2).
At a real-cyclotomic root lifted mod p^2, a norm that is zero mod p must
also be zero mod p^2: both conjugate factors are divisible by p. A valuation
exactly one certifies no integer row can have the specified autocorrelation.
"""
import ctypes
from functools import lru_cache
import hashlib
from math import isqrt
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def is_prime(p):
    if p<2:return False
    if not p%2:return p==2
    return all(p%q for q in range(3,isqrt(p)+1,2))


def _traces(t,half,modulus):
    a,b=2,t%modulus;out=[b]
    for u in range(2,half+1):a,b=b,(t*b-a)%modulus;out.append(b)
    return tuple(out)


def _psi_and_derivative(t,half,modulus):
    a,b=2,t%modulus;da,db=0,1;value=1+b;derivative=1
    for u in range(2,half+1):
        a,b,da,db=b,(t*b-a)%modulus,db,(b+t*db-da)%modulus
        value+=b;derivative+=db
    return value%modulus,derivative%modulus


def _field_multiply(x,y,p,nonsquare):
    a,b=x;c,d=y
    return ((a*c+nonsquare*b*d)%p,(a*d+b*c)%p)


def _field_power(x,exponent,p,nonsquare):
    out=(1,0)
    while exponent:
        if exponent&1:out=_field_multiply(out,x,p,nonsquare)
        x=_field_multiply(x,x,p,nonsquare);exponent>>=1
    return out


def real_roots_mod_p(d,p):
    if d not in (3,5,11,43) or not is_prime(p) or p<3 or (p+1)%d:
        raise ValueError('supported prime d and odd prime p == -1 mod d required')
    nonsquare=next(a for a in range(2,p) if pow(a,(p-1)//2,p)==p-1)
    alpha=None
    for a in range(p):
        candidate=_field_power((a,1),(p*p-1)//d,p,nonsquare)
        if candidate!=(1,0):alpha=candidate;break
    if alpha is None or _field_power(alpha,d,p,nonsquare)!=(1,0):raise RuntimeError('primitive d root construction failed')
    if (alpha[0]*alpha[0]-nonsquare*alpha[1]*alpha[1])%p!=1:raise RuntimeError('primitive root is not norm one')
    out=[];value=(1,0)
    for u in range(1,d//2+1):
        value=_field_multiply(value,alpha,p,nonsquare);out.append(2*value[0]%p)
    if len(set(out))!=d//2 or any(_psi_and_derivative(t,d//2,p)[0] for t in out):
        raise RuntimeError('real-cyclotomic root census incomplete')
    return tuple(sorted(out))


@lru_cache(maxsize=8)
def lifted_tables(d,limit):
    rows=[]
    for p in range(d-1,limit+1,d):
        if p<3 or not is_prime(p):continue
        modulus=p*p
        for root in real_roots_mod_p(d,p):
            value,derivative=_psi_and_derivative(root,d//2,modulus)
            if value%p or not derivative%p:raise RuntimeError('Hensel real root lacks a simple mod-p zero')
            correction=-(value//p)*pow(derivative,-1,p)%p;lift=root+p*correction
            if _psi_and_derivative(lift,d//2,modulus)[0]:raise RuntimeError('lifted real root failed exact mod-p-squared audit')
            if pow((root*root-4)%p,(p-1)//2,p)!=p-1:raise RuntimeError('primitive root factor is not irreducible quadratic')
            rows.append((p,lift,(1,)+_traces(lift,d//2,modulus)))
    return tuple(rows)


@lru_cache(maxsize=1)
def _library():
    root=Path(__file__).resolve().parents[1];source=root/'csrc/prime_norm.c';build=root/'build';build.mkdir(exist_ok=True)
    suffix='.dylib' if sys.platform=='darwin' else '.so';path=build/('prime_norm_'+hashlib.sha256(source.read_bytes()).hexdigest()[:16]+suffix)
    if not path.exists():
        fd,temporary=tempfile.mkstemp(dir=build,suffix=suffix);os.close(fd)
        try:
            subprocess.run(['cc','-O3','-std=c11','-Wall','-Wextra','-Werror','-fPIC',
                            '-dynamiclib' if sys.platform=='darwin' else '-shared',str(source),'-o',temporary],
                           check=True,capture_output=True,timeout=30);os.replace(temporary,path)
        finally:
            if os.path.exists(temporary):os.unlink(temporary)
    lib=ctypes.CDLL(str(path));i=ctypes.POINTER(ctypes.c_int32);u=ctypes.POINTER(ctypes.c_uint32)
    lib.pqcp_prime_norm_obstruction.argtypes=[i,ctypes.c_int,u,u,ctypes.c_int,i,ctypes.POINTER(ctypes.c_uint64)]
    lib.pqcp_prime_norm_obstruction.restype=ctypes.c_int
    return lib


@lru_cache(maxsize=8)
def _arrays(d,limit):
    rows=lifted_tables(d,limit);u=ctypes.c_uint32
    primes=(u*len(rows))(*(row[0] for row in rows));width=d//2+1
    coefficients=(u*(len(rows)*width))(*(v for row in rows for v in row[2]))
    return primes,coefficients


def verify_prime_norm_certificate(certificate):
    c=certificate;profile=tuple(c['profile']);d=len(profile);p=c['prime'];root=c['root_mod_p_squared']
    if d not in (3,5,11,43) or not is_prime(p) or p<3 or (p+1)%d or not 0<=root<p*p:return False
    if any(profile[u]!=profile[-u] for u in range(1,d)):return False
    if _psi_and_derivative(root,d//2,p*p)[0] or pow((root*root-4)%p,(p-1)//2,p)!=p-1:return False
    coefficients=(1,)+_traces(root,d//2,p*p)
    residue=sum(v*a for v,a in zip(profile[:d//2+1],coefficients))%(p*p)
    return residue==c['residue_mod_p_squared'] and residue>0 and residue%p==0


def prime_norm_obstruction(profile,*,limit=50000,native=True):
    profile=tuple(profile);d=len(profile)
    if (d not in (3,5,11,43) or any(type(v) is not int for v in profile)
            or any(profile[u]!=profile[-u] for u in range(1,d))
            or type(limit) is not int or not 3<=limit<=65000 or type(native) is not bool):
        raise ValueError('supported symmetric integral cyclic profile and finite prime limit 3..65000 required')
    rows=lifted_tables(d,limit);found=None
    if native and max(map(abs,profile))<=d:
        primes,coefficients=_arrays(d,limit);array=(ctypes.c_int32*d)(*profile);row=ctypes.c_int32();residue=ctypes.c_uint64()
        status=_library().pqcp_prime_norm_obstruction(array,d,primes,coefficients,len(rows),ctypes.byref(row),ctypes.byref(residue))
        if status:raise RuntimeError('native prime norm bound/input audit failed')
        if row.value>=0:found=(rows[row.value],residue.value)
    else:
        for data in rows:
            p,root,coefficients=data;residue=sum(v*a for v,a in zip(profile[:d//2+1],coefficients))%(p*p)
            if residue and not residue%p:found=(data,residue);break
    if found is None:return None
    (p,root,coefficients),residue=found
    certificate=dict(prime=p,root_mod_p_squared=root,residue_mod_p_squared=residue,profile=profile,
        proof='One integer cyclic norm zero in an irreducible quadratic residue field must vanish modulo p squared.',
        scope='Excludes this SINGLE integer-row autocorrelation. Does not exclude a sum of two unknown norms.')
    if not verify_prime_norm_certificate(certificate):raise RuntimeError('native prime norm certificate failed independent Python Hensel audit')
    return certificate
