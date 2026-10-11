"""Exact fixed-weight binary bracelet generation for sparse support matching."""
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


@lru_cache(maxsize=1)
def _library():
    root = Path(__file__).resolve().parents[1]
    source = root/'csrc/sparse_supports.c'
    build = root/'build'
    build.mkdir(exist_ok=True)
    suffix = '.dylib' if sys.platform == 'darwin' else '.so'
    path = build/('sparse_supports_'+hashlib.sha256(source.read_bytes()).hexdigest()[:16]+suffix)
    if not path.exists():
        fd, temporary = tempfile.mkstemp(dir=build, suffix=suffix)
        os.close(fd)
        try:
            subprocess.run(['cc', '-O3', '-std=c11', '-Wall', '-Wextra', '-Werror', '-fPIC',
                '-dynamiclib' if sys.platform == 'darwin' else '-shared', str(source), '-o', temporary],
                check=True, capture_output=True, timeout=30)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    lib = ctypes.CDLL(str(path))
    lib.pqcp_sparse_bracelets.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint64)]
    lib.pqcp_sparse_bracelets.restype = ctypes.c_int
    return lib


def sparse_bracelets(d, weight, *, capacity=None):
    if type(d) is not int or not 3 <= d <= 47 or not d % 2 or type(weight) is not int or not 1 <= weight < d:
        raise ValueError('odd half-length 3..47 and nontrivial weight required')
    if capacity is None:
        # Every representative has bit zero occupied. This is an exact safe
        # buffer bound independent of orbit-period exceptions.
        capacity = comb(d-1, weight-1)
    if type(capacity) is not int or not 1 <= capacity <= 2**23:
        raise ValueError('output capacity in 1..2**23 required')
    started = perf_counter()
    masks = (ctypes.c_uint64*capacity)()
    signatures = (ctypes.c_uint32*capacity)()
    stats = (ctypes.c_uint64*2)()
    status = _library().pqcp_sparse_bracelets(d, weight, capacity, masks, signatures, stats)
    if status not in (0, 1):
        raise RuntimeError('sparse bracelet generation failed')
    stored = min(int(stats[1]), capacity)
    return dict(records=tuple(zip(masks[:stored], signatures[:stored])),
        complete=status == 0, candidates=int(stats[0]), total=int(stats[1]), elapsed=perf_counter()-started)
