"""Exact per-side two-swap neighborhood matching; bounded output, audited hits."""
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
    root = Path(__file__).resolve().parents[1]
    source = root/'csrc/ternary_mitm.c'
    build = root/'build'
    build.mkdir(exist_ok=True)
    suffix = '.dylib' if sys.platform == 'darwin' else '.so'
    path = build/('ternary_mitm_'+hashlib.sha256(source.read_bytes()).hexdigest()[:16]+suffix)
    if not path.exists():
        fd, temporary = tempfile.mkstemp(dir=build, suffix=suffix)
        os.close(fd)
        try:
            subprocess.run(['cc', '-O3', '-std=c11', '-Wall', '-Wextra', '-Werror', '-fPIC',
                            '-dynamiclib' if sys.platform == 'darwin' else '-shared',
                            str(source), '-o', temporary], check=True, capture_output=True, timeout=30)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    library = ctypes.CDLL(str(path))
    ptr = ctypes.POINTER(ctypes.c_int)
    library.pqcp_ternary_mitm.argtypes = [ptr, ptr, ctypes.c_int, ctypes.c_int, ptr, ptr,
                                        ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64)]
    library.pqcp_ternary_mitm.restype = ctypes.c_int
    return library


def repair_mitm(a, b, target, *, max_results=32):
    """All signatures retained. Cap means SAT_TRUNCATED, never local UNSAT.

    Exhaustion covers only N2(a) x N2(b), not the entire radius-eight ball.
    Multiple distinct words with equal PACFs are exported separately for lift.
    """
    started = perf_counter()
    a, b, d, wanted = _validated_reference(a, b, target)
    if d > 47 or type(max_results) is not int or not 1 <= max_results <= 4096:
        raise ValueError('requires L<=94 and max_results in 1..4096')
    lib = _library()
    output = (ctypes.c_int*(max_results*2*d))()
    count, entries, probes = ctypes.c_int(), ctypes.c_uint64(), ctypes.c_uint64()
    construction = perf_counter()-started
    started = perf_counter()
    status = lib.pqcp_ternary_mitm((ctypes.c_int*(2*d))(*(a+b)),
        (ctypes.c_int*d)(*wanted), d, max_results, output, ctypes.byref(count),
        ctypes.byref(entries), ctypes.byref(probes))
    elapsed = perf_counter()-started
    if status not in (0, 1):
        raise RuntimeError('MITM kernel failed with code {}'.format(status))
    pairs = []
    for i in range(count.value):
        offset = i*2*d
        pair = (tuple(output[offset:offset+d]), tuple(output[offset+d:offset+2*d]))
        exact = tuple(x+y for x,y in zip(*(pacf(w) for w in pair)))
        if exact != wanted or any(sorted(w) != sorted(old) or sum(x!=y for x,y in zip(w,old))>4
                                  for w,old in zip(pair,(a,b))):
            raise RuntimeError('MITM result failed independent integer audit')
        pairs.append(pair)
    return dict(status='SAT_TRUNCATED' if status else 'SAT_COMPLETE' if pairs else 'UNSAT_NEIGHBORHOOD',
                complete=status == 0, words=pairs, construction_seconds=construction, solve_seconds=elapsed,
                entries=entries.value, probes=probes.value, max_results=max_results)
