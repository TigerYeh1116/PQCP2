"""Native exact local energy selection; independent Python audits at boundary."""
import ctypes
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile


@lru_cache(maxsize=1)
def _library():
    root = Path(__file__).resolve().parents[1]
    source = root/'csrc/half_shift_projection.c'
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    build = root/'build'
    build.mkdir(exist_ok=True)
    suffix = '.dylib' if sys.platform == 'darwin' else '.so'
    path = build/('half_shift_projection_'+digest+suffix)
    if not path.exists():
        fd, temporary = tempfile.mkstemp(dir=build, suffix=suffix)
        os.close(fd)
        try:
            subprocess.run(['cc', '-O3', '-std=c11', '-Wall', '-Wextra', '-Werror', '-fPIC',
                '-dynamiclib' if sys.platform == 'darwin' else '-shared', str(source), '-o', temporary],
                check=True, capture_output=True, timeout=30)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary): os.unlink(temporary)
    lib = ctypes.CDLL(str(path))
    p = ctypes.POINTER(ctypes.c_int)
    lib.pqcp_half_shift_project.argtypes = [p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, p, ctypes.POINTER(ctypes.c_int64)]
    lib.pqcp_half_shift_project.restype = ctypes.c_int
    return lib


def project_energy_ties(words, target, counts, reverse_order=False, *, compression=True):
    length = target.L
    if (type(length) is not int or not 6 <= length <= 94 or length % 4 != 2 or len(words) != 2
            or type(target.k) is not int or not 1 <= target.k < length//2
            or type(target.eta) is not int or target.eta not in (-1, 1)
            or any(len(w) != length or any(type(v) is not int or v not in (0, 1) for v in w) for w in words)
            or len(counts) != 2 or any(type(n) is not int or not 0 <= n <= length//2 for n in counts)
            or type(reverse_order) is not bool or type(compression) is not bool):
        raise ValueError('bounded odd-half-length binary input and integral contents required')
    raw = (ctypes.c_int*(2*length))(*(tuple(words[0])+tuple(words[1])))
    out = (ctypes.c_int*(2*length))()
    numerator = ctypes.c_int64()
    status = _library().pqcp_half_shift_project(raw, length, target.k, target.eta,
        *counts, int(reverse_order), int(compression), out, ctypes.byref(numerator))
    if status != 0 or numerator.value % 16:
        raise RuntimeError('native half-shift energy construction failed: {}'.format(status))
    return (tuple(out[:length]), tuple(out[length:])), numerator.value//16
