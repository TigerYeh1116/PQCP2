"""Exact tail-bit projection and full verification for arbitrary prefixes.

The core has no Golay or symmetry premise. Only the experiment's source
family uses periodic Golay ingredients. All tail nullspace solutions remain.
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

from .structured_energy import target_profile
from .verifier import verify_pqcp


@lru_cache(maxsize=1)
def _library():
    root=Path(__file__).resolve().parents[1]
    source=root/'csrc/boundary_tail.c'
    digest=hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    build=root/'build';build.mkdir(exist_ok=True)
    path=build/('boundary_tail_'+digest+('.dylib' if sys.platform=='darwin' else '.so'))
    if not path.exists():
        fd,temp=tempfile.mkstemp(dir=build,suffix=path.suffix);os.close(fd)
        try:
            subprocess.run(['cc','-O3','-std=c11','-Wall','-Wextra','-Werror','-fPIC',
                '-dynamiclib' if sys.platform=='darwin' else '-shared',str(source),'-o',temp],
                check=True,capture_output=True,timeout=30)
            os.replace(temp,path)
        finally:
            if os.path.exists(temp):os.unlink(temp)
    library=ctypes.CDLL(str(path))
    library.pqcp_boundary_tails.argtypes=[ctypes.POINTER(ctypes.c_uint64),ctypes.c_int,ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint16),ctypes.POINTER(ctypes.c_uint8),ctypes.POINTER(ctypes.c_int8),
        ctypes.POINTER(ctypes.c_uint64)]
    library.pqcp_boundary_tails.restype=ctypes.c_int
    return library


def search_boundary_tails(prefixes,length):
    prefixes=tuple(map(tuple,prefixes))
    if (type(length) is not int or not 4<=length<=94 or length%2 or len(prefixes)!=2
            or len(prefixes[0])!=len(prefixes[1])
            or any(type(v) is not int or v not in (0,1) for row in prefixes for v in row)):
        raise ValueError('two literal binary prefixes and even target length in 4..94 required')
    tail=length-len(prefixes[0])
    if not 1<=tail<=4 or tail>=length//2:raise ValueError('one to four appended bits per row required')
    masks=tuple(sum(v << i for i,v in enumerate(row)) for row in prefixes)
    return search_boundary_tail_masks(masks,length,tail)


def search_boundary_tail_masks(masks,length=86,tail=4):
    """Packed variant for complete orbit traversals; every exported full pair is audited."""
    masks=tuple(masks)
    if (type(length) is not int or not 4<=length<=94 or length%2 or type(tail) is not int
            or not 1<=tail<=4 or tail>=length//2 or len(masks)!=2
            or any(type(m) is not int or not 0<=m<1<<(length-tail) for m in masks)):
        raise ValueError('two bounded packed prefixes, even length and short tail required')
    u=ctypes.c_uint64
    raw=(u*4)(*(v for m in masks for v in (m & ((1<<64)-1),m >> 64)))
    assignments,lags,signs=(ctypes.c_uint16*256)(),(ctypes.c_uint8*256)(),(ctypes.c_int8*256)()
    stats=(u*8)()
    tick=perf_counter()
    count=_library().pqcp_boundary_tails(raw,length,tail,assignments,lags,signs,stats)
    if count<0:raise RuntimeError('exact boundary projection failed: '+str(count))
    witnesses=[]
    for j in range(count):
        value=int(assignments[j]);low=(1<<tail)-1
        words=tuple(tuple((m >> i)&1 for i in range(length-tail))+
                    tuple((v >> i)&1 for i in range(tail))
                    for m,v in zip(masks,(value & low,value >> tail)))
        check=verify_pqcp(*words)
        k,eta=int(lags[j]),int(signs[j])
        if not check.is_valid or tuple(check.profile)!=target_profile(length,k,eta):
            raise RuntimeError('boundary witness failed independent full binary verifier')
        witnesses.append(dict(words=words,k=k,eta=eta,tail_bits=value))
    return dict(witnesses=witnesses,complete=True,rank=int(stats[0]),nullity=int(stats[1]),
                target_groups=int(stats[2]),excluded_groups=int(stats[3]),
                surviving_tail_assignments=int(stats[4]),correlation_checks=int(stats[5]),
                covered_tail_assignments=int(stats[7]),elapsed=perf_counter()-tick,
                scope='ALL short tails for these fixed prefixes and ALL strict Project target shifts/signs; no global exclusion.')
