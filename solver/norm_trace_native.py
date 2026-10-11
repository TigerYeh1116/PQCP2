"""Native integer field arithmetic; complete zero-partner fallback lives in Python."""
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
    source = root / 'csrc/norm_trace.c'
    build = root / 'build'
    build.mkdir(exist_ok=True)
    suffix = '.dylib' if sys.platform == 'darwin' else '.so'
    path = build / ('norm_trace_' + hashlib.sha256(source.read_bytes()).hexdigest()[:16] + suffix)
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
    lib = ctypes.CDLL(str(path))
    word = ctypes.c_uint32
    pointer = ctypes.POINTER(word)
    lib.pqcp_field_norm_trace.argtypes = [word, word, word, word, ctypes.c_int,
                                         pointer, pointer, pointer]
    lib.pqcp_field_norm_trace.restype = ctypes.c_int
    return lib


@lru_cache(maxsize=8)
def _tables(factor):
    from .norm_trace_completion import _frobenius_columns
    from .skew_norm_completion import _skew_field_roots
    degree = factor.bit_length()-1
    word = ctypes.c_uint32
    columns = (word * degree)(*_frobenius_columns(factor))
    roots = (word * (1 << degree))(*([0xffffffff] * (1 << degree)))
    for norm, pair in _skew_field_roots(factor).items():
        if len(pair) != 2 or pair[0] ^ pair[1] != 1:
            raise RuntimeError('skew root table failed exact two-root invariant')
        roots[norm] = pair[0]
    return columns, roots


def field_norm_trace_roots(n, h, w, factor):
    degree = factor.bit_length()-1
    if (degree not in (2, 4, 10, 14) or type(factor) is not int
            or any(type(v) is not int or not 0 <= v < 1 << degree for v in (n, h, w)) or not w):
        raise ValueError('supported field and nonzero literal partner required')
    columns, roots = _tables(factor)
    output = (ctypes.c_uint32 * 2)()
    count = _library().pqcp_field_norm_trace(n, h, w, factor, degree, columns, roots, output)
    if count < 0:
        raise RuntimeError(f'native norm/trace field audit failed: {count}')
    return tuple(output[i] for i in range(count))
