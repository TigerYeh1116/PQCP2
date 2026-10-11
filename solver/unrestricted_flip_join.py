"""Exact content-changing Hamming repair with quadratic mod-8 MITM pruning."""
import ctypes
from functools import lru_cache
import hashlib
from math import comb
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from time import perf_counter

from .correlation import normalize_binary_sequence
from .structured_energy import target_profile
from .verifier import verify_pqcp


@lru_cache(maxsize=1)
def _library():
    root=Path(__file__).resolve().parents[1];source=root/'csrc/unrestricted_flip_join.c'
    build=root/'build';build.mkdir(exist_ok=True)
    suffix='.dylib' if sys.platform=='darwin' else '.so'
    path=build/('unrestricted_flip_join_'+hashlib.sha256(source.read_bytes()).hexdigest()[:16]+suffix)
    if not path.exists():
        fd,temp=tempfile.mkstemp(dir=build,suffix=suffix);os.close(fd)
        try:
            subprocess.run(['cc','-O3','-std=c11','-Wall','-Wextra','-Werror','-fPIC',
                            '-dynamiclib' if sys.platform=='darwin' else '-shared',str(source),'-o',temp],
                           check=True,capture_output=True,timeout=30)
            os.replace(temp,path)
        finally:
            if os.path.exists(temp):os.unlink(temp)
    lib=ctypes.CDLL(str(path));u=ctypes.c_uint64
    lib.pqcp_unrestricted_flip_join.argtypes=[u,u,u,u,ctypes.c_int,ctypes.c_int,ctypes.c_int,
                                             ctypes.POINTER(u),ctypes.POINTER(u)]
    lib.pqcp_unrestricted_flip_join.restype=ctypes.c_int
    return lib


def exact_unrestricted_flip_repair(a,b,*,radius=4,content_grouping=True):
    """Decide existence in the full product of two radius-r Hamming balls.

    On SAT return one independently verified witness (not all SAT models).
    On UNSAT audit both complete row counts before excluding this neighborhood.
    """
    a,b=tuple(normalize_binary_sequence(a)),tuple(normalize_binary_sequence(b));n=len(a)
    if (n!=len(b) or not 4<=n<=94 or n%2 or type(radius) is not int or not 0<=radius<=5
            or type(content_grouping) is not bool):
        raise ValueError('equal even binary rows length 4..94 and radius 0..5 required')
    if verify_pqcp(a,b).is_valid:raise ValueError('repair centers must be non-solutions')
    tick=perf_counter();masks=tuple(sum(v<<i for i,v in enumerate(row)) for row in (a,b))
    u=ctypes.c_uint64;output=(u*6)();stats=(u*7)();mask64=(1<<64)-1
    status=_library().pqcp_unrestricted_flip_join(masks[0]&mask64,masks[0]>>64,
        masks[1]&mask64,masks[1]>>64,n,radius,int(content_grouping),output,stats)
    if status not in (0,1):raise RuntimeError('native unrestricted repair failed: '+str(status))
    expected=sum(comb(n,r) for r in range(radius+1));witnesses=[]
    if not status and tuple(stats[:2])!=(expected,expected):
        raise RuntimeError('native unrestricted Hamming balls incompletely enumerated')
    if status:
        changed=(int(output[0])+(int(output[1])<<64),int(output[2])+(int(output[3])<<64))
        words=tuple(tuple((m>>i)&1 for i in range(n)) for m in changed)
        k,eta=int(output[4]),1 if output[5] else -1
        check=verify_pqcp(*words)
        if (not check.is_valid or tuple(check.profile)!=target_profile(n,k,eta)
                or any(sum(x!=y for x,y in zip(old,new))>radius for old,new in zip((a,b),words))):
            raise RuntimeError('unrestricted repair hit failed independent full target/ball audit')
        witnesses.append(dict(words=words,k=k,eta=eta))
    return dict(status='SAT' if status else 'UNSAT_UNRESTRICTED_PRODUCT_BALL',complete=not status,
                existence_decided=True,witnesses=witnesses,radius_per_row=radius,
                expected_configurations_per_row=expected,raw_product_configurations=expected**2,
                enumerated_per_row=tuple(stats[:2]),indexed_configurations=int(stats[2]),
                eligible_tested_configurations=int(stats[3]),exact_signature_lookups=int(stats[4]),
                mod8_candidate_pairs=int(stats[5]),full_coordinate_checks=int(stats[6]),
                content_grouping=content_grouping and n==86,
                elapsed=perf_counter()-tick,
                scope='At most radius arbitrary flips on EACH row, unrestricted parity contents, '
                      'all strict target positions/signs. SAT exports first witness only; '
                      'UNSAT excludes this product ball, never general L86 PQCPs.')
