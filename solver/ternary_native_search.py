"""Integer compressed navigation, bounded annealing; a miss is never UNSAT."""
import ctypes
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from time import perf_counter

from .compressed_repair import _validated_reference
from .structured_energy import integer_periodic_autocorrelation_profile as pacf


@lru_cache(maxsize=1)
def _library():
    root=Path(__file__).resolve().parents[1]
    source=root/'csrc/ternary_anneal.c'
    build=root/'build';build.mkdir(exist_ok=True)
    suffix='.dylib' if sys.platform=='darwin' else '.so'
    path=build/('ternary_anneal_'+hashlib.sha256(source.read_bytes()).hexdigest()[:16]+suffix)
    if not path.exists():
        fd,temporary=tempfile.mkstemp(dir=build,suffix=suffix);os.close(fd)
        try:
            subprocess.run(['cc','-O3','-std=c11','-Wall','-Wextra','-Werror','-fPIC',
                '-dynamiclib' if sys.platform=='darwin' else '-shared',str(source),'-lm','-o',temporary],
                check=True,capture_output=True,timeout=30)
            os.replace(temporary,path)
        finally:
            if os.path.exists(temporary):os.unlink(temporary)
    lib=ctypes.CDLL(str(path));ptr=ctypes.POINTER(ctypes.c_int)
    lib.pqcp_ternary_anneal.argtypes=[ptr,ptr,ptr,ctypes.c_int,ctypes.c_uint64,ctypes.c_uint64,
                                    ctypes.c_int,ctypes.c_int,ptr,ptr,ctypes.POINTER(ctypes.c_uint64)]
    lib.pqcp_ternary_anneal.restype=ctypes.c_int
    return lib


def search_ternary_native(a,b,target,*,seed=123,moves=65536,period=8192,audit_interval=1024,
                          binary_content_pruning=True):
    started=perf_counter();a,b,d,wanted=_validated_reference(a,b,target)
    if d>47 or type(moves) is not int or not 0<=moves<=2**31-1:
        raise ValueError('requires L<=94 and integer moves in 0..2^31-1')
    if type(seed) is not int or any(type(v) is not int or not 1<=v<=2**31-1 for v in (period,audit_interval)):
        raise ValueError('seed must be integer; period and audit_interval must be positive int32')
    if type(binary_content_pruning) is not bool:
        raise ValueError('binary_content_pruning must be boolean')
    bounds=[]
    for e,o in ((target.a_even_ones,target.a_odd_ones),(target.b_even_ones,target.b_odd_ones)):
        bounds.extend((max(0,e+o-d),min(e,o)))
    if d == 43 and binary_content_pruning:
        from .l86_binary_content import binary_content_bounds
        certified = binary_content_bounds(target)
        # Keep the coarse-only public API available for old/outside-domain
        # replay inputs. Production starts inside these full-feasible bounds.
        if certified is not None and all(lo <= w.count(-1) <= hi for w, (lo, hi) in zip((a, b), certified)):
            for side, (lo, hi) in enumerate(certified):
                bounds[2*side] = max(bounds[2*side], lo)
                bounds[2*side+1] = min(bounds[2*side+1], hi)
    lib=_library();output=(ctypes.c_int*(2*d))();best=ctypes.c_int();stats=(ctypes.c_uint64*5)()
    construction=perf_counter()-started;started=perf_counter()
    status=lib.pqcp_ternary_anneal((ctypes.c_int*(2*d))(*(a+b)),(ctypes.c_int*d)(*wanted),
        (ctypes.c_int*4)(*bounds),d,seed%(2**64),moves,period,audit_interval,output,ctypes.byref(best),stats)
    elapsed=perf_counter()-started
    if status not in (0,1):raise RuntimeError('native search validation/audit failed: {}'.format(status))
    words=(tuple(output[:d]),tuple(output[d:]));_validated_reference(*words,target)
    profile=tuple(x+y for x,y in zip(*(pacf(w) for w in words)))
    energy=sum((profile[u]-wanted[u])**2 for u in range(1,d//2+1))
    initial=tuple(x+y for x,y in zip(pacf(a),pacf(b)))
    initial_energy=sum((initial[u]-wanted[u])**2 for u in range(1,d//2+1))
    if energy!=best.value or energy>initial_energy or (status==1)!=(energy==0):
        raise RuntimeError('native best failed independent integer audit')
    return dict(status='SAT_COMPRESSED' if status else 'BOUNDED_MISS',words=words,profile=profile,
        energy=energy,initial_energy=initial_energy,construction_seconds=construction,solve_seconds=elapsed,
        proposals=stats[0],accepted=stats[1],transfer_proposals=stats[2],transfer_accepted=stats[3],audits=stats[4])
