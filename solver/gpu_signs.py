"""Batch C-sign feasibility on the Apple GPU (csrc/gpu_ternary_signs.m).

prepare_pair builds the same GF(2) affine sign space as
ternary_signs_native.enumerate_signs; GpuSigns.check enumerates it on the
GPU for many support pairs at once. A GPU hit is only a candidate: callers
re-solve it with the exact CPU solver. A miss is a complete enumeration of
that pair's affine space, the same decision as the CPU kernel.
"""
import ctypes
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from .ternary_support_search import support_sign_system


@lru_cache(maxsize=1)
def _library():
    if sys.platform != 'darwin':
        raise RuntimeError('Metal GPU signs need macOS')
    root = Path(__file__).resolve().parents[1]
    source = root/'csrc/gpu_ternary_signs.m'
    build = root/'build'
    build.mkdir(exist_ok=True)
    path = build/('gpu_ternary_signs_'+hashlib.sha256(source.read_bytes()).hexdigest()[:16]+'.dylib')
    if not path.exists():
        fd, temporary = tempfile.mkstemp(dir=build, suffix='.dylib')
        os.close(fd)
        try:
            subprocess.run(['clang', '-O2', '-fobjc-arc', '-Wall', '-Werror', '-dynamiclib',
                            '-framework', 'Foundation', '-framework', 'Metal', str(source), '-o', temporary],
                           check=True, capture_output=True, timeout=60)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    lib = ctypes.CDLL(str(path))
    u32, u64, i32 = ctypes.c_uint32, ctypes.c_uint64, ctypes.c_int32
    lib.pqcp_gpu_create.argtypes = [ctypes.c_char_p, ctypes.c_int]
    lib.pqcp_gpu_create.restype = ctypes.c_void_p
    lib.pqcp_gpu_check.argtypes = [ctypes.c_void_p, u32, u32, u32, u32, ctypes.POINTER(u64), ctypes.POINTER(u64),
                                   ctypes.POINTER(u64), ctypes.POINTER(u32), ctypes.POINTER(i32), ctypes.POINTER(i32),
                                   ctypes.POINTER(u32), ctypes.POINTER(u64)]
    lib.pqcp_gpu_check.restype = ctypes.c_int
    lib.pqcp_gpu_destroy.argtypes = [ctypes.c_void_p]
    return lib


def prepare_pair(masks, d, wanted, negative):
    """The CPU kernel's affine space and edge-cut targets, or None if already UNSAT."""
    system = support_sign_system(masks, d, wanted, negative)
    if not system.consistent:
        return None
    free_count = len(system.free_columns)
    initial, toggles = [0, 0], [0]*(2*free_count)
    for (s, i), (mask, constant) in zip(system.variables, system.expressions()):
        initial[s] |= constant << i
        for bit in range(free_count):
            if mask >> bit & 1:
                toggles[2*bit+s] |= 1 << i
    for side in (0, 1):  # Forced-cardinality contradiction, exactly as enumerate_signs.
        variable = 0
        for bit in range(free_count):
            variable |= toggles[2*bit+side]
        fixed = bin(initial[side] & ~variable).count('1')
        if not fixed <= negative[side] <= fixed+bin(variable).count('1'):
            return None
    every = (1 << d)-1
    required = []
    for u in range(1, d//2+1):
        m = sum(bin(s & (((s << u) | (s >> (d-u))) & every)).count('1') for s in masks)
        if m < abs(wanted[u]) or (m-wanted[u]) % 2:
            return None
        required.append((m-wanted[u])//2)
    return tuple(masks), tuple(initial), tuple(toggles), free_count, tuple(required), tuple(negative)


class GpuSigns:
    def __init__(self, d, chunk_bits=12):
        self.d, self.chunk_bits = d, chunk_bits
        self.lib = _library()
        error = ctypes.create_string_buffer(512)
        self.handle = self.lib.pqcp_gpu_create(error, len(error))
        if not self.handle:
            raise RuntimeError('Metal setup failed: '+error.value.decode(errors='replace'))

    def check(self, prepared):
        """prepared: list of prepare_pair results (not None). Returns (hit flags, witness masks)."""
        n, d = len(prepared), self.d
        maxF = max(p[3] for p in prepared)
        u64, u32, i32 = ctypes.c_uint64, ctypes.c_uint32, ctypes.c_int32
        support = (u64*(2*n))(*(m for p in prepared for m in p[0]))
        initial = (u64*(2*n))(*(m for p in prepared for m in p[1]))
        toggles = (u64*(2*n*max(maxF, 1)))()
        for k, p in enumerate(prepared):
            for j, t in enumerate(p[2]):
                toggles[2*maxF*k+j] = t
        freec = (u32*n)(*(p[3] for p in prepared))
        required = (i32*(n*(d//2)))(*(r for p in prepared for r in p[4]))
        negative = (i32*(2*n))(*(c for p in prepared for c in p[5]))
        hits, witness = (u32*n)(), (u64*(2*n))()
        status = self.lib.pqcp_gpu_check(self.handle, n, d, maxF, self.chunk_bits, support, initial, toggles,
                                         freec, required, negative, hits, witness)
        if status:
            raise RuntimeError('GPU sign check failed: {}'.format(status))
        return [bool(h) for h in hits], [(witness[2*k], witness[2*k+1]) for k in range(n)]

    def __del__(self):
        if getattr(self, 'handle', None):
            self.lib.pqcp_gpu_destroy(self.handle)
            self.handle = None
