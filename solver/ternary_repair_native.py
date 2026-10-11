"""Exact content-preserving ternary radius-four repair, via two swaps.

For each side, draw an edge old_symbol -> new_symbol at every changed
position. Fixed contents make a balanced directed graph on three vertices,
without loops. At <=4 changed positions, its cycles can only be one 2-cycle,
one 3-cycle, or two 2-cycles. Each is realizable with <=2 transpositions.
This holds across both sides as well, because counts are fixed per side.
Conversely two swaps change at most four positions and preserve contents.
Thus exhaustion proves precisely local radius-four UNSAT, not global UNSAT.
"""

import ctypes
from dataclasses import replace
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from time import perf_counter

from .compressed_repair import CompressedRepairResult, _validated_reference
from .structured_energy import integer_periodic_autocorrelation_profile


@lru_cache(maxsize=1)
def _library():
    root = Path(__file__).resolve().parents[1]
    source = root/'csrc/ternary_repair.c'
    build = root/'build'
    build.mkdir(exist_ok=True)
    suffix = '.dylib' if sys.platform == 'darwin' else '.so'
    path = build/('ternary_repair_'+hashlib.sha256(source.read_bytes()).hexdigest()[:16]+suffix)
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
    library.pqcp_ternary_repair.argtypes = [ptr, ptr, ctypes.c_int, ctypes.c_uint64, ptr,
                                          ctypes.POINTER(ctypes.c_uint64)]
    library.pqcp_ternary_repair.restype = ctypes.c_int
    return library


def repair_two_swaps(a, b, target, *, max_evaluations=0):
    """Return (audited result, evaluations); zero budget means full enumeration."""
    started = perf_counter()
    a, b, d, wanted = _validated_reference(a, b, target)
    if d > 47:
        raise ValueError('native radius-four repair supports L<=94')
    if type(max_evaluations) is not int or not 0 <= max_evaluations < 2**64:
        raise ValueError('max_evaluations must be a nonnegative uint64')
    library = _library()
    words_buffer = (ctypes.c_int*(2*d))(*(a+b))
    wanted_buffer = (ctypes.c_int*d)(*wanted)
    output = (ctypes.c_int*(2*d))()
    evaluations = ctypes.c_uint64()
    construction = perf_counter()-started
    started = perf_counter()
    status = library.pqcp_ternary_repair(words_buffer, wanted_buffer, d, max_evaluations,
                                        output, ctypes.byref(evaluations))
    result = CompressedRepairResult('UNKNOWN', construction, perf_counter()-started, 4)
    if status == 0:
        return replace(result, status='UNSAT_LOCAL'), evaluations.value
    if status == -1:
        return replace(result, reason='evaluation budget exhausted'), evaluations.value
    if status != 1:
        raise RuntimeError('native ternary repair kernel rejected input')
    words = (tuple(output[:d]), tuple(output[d:]))
    distance = sum(x != y for w, r in zip(words, (a, b)) for x, y in zip(w, r))
    exact = tuple(sum(v) for v in zip(*(integer_periodic_autocorrelation_profile(w) for w in words)))
    if (exact != wanted or distance > 4 or
            any(sorted(w) != sorted(r) for w, r in zip(words, (a, b)))):
        raise RuntimeError('native result failed independent integer audit')
    return replace(result, status='SAT_COMPRESSED', words=words, distance=distance), evaluations.value
