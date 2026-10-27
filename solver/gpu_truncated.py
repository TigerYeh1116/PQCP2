"""Apple GPU version of the per-support truncated C scan (csrc/gpu_truncated.m).

The CPU prepares, per C_A support, its kept sign patterns (top_signs) and its
weight-matched C_B supports (family_supports); pack_job turns them into flat bytes in a
worker process. The GPU then runs the single-row elimination and every kept pattern
for all C_B supports of a batch of C_A supports at once (about 5x the 12-core CPU scan
on the M6). Hits are the same exact solutions truncated_scan reports.
"""
from array import array
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
    if sys.platform != 'darwin':
        raise RuntimeError('Metal needs macOS')
    root = Path(__file__).resolve().parents[1]
    source = root/'csrc/gpu_truncated.m'
    build = root/'build'
    build.mkdir(exist_ok=True)
    path = build/('gpu_truncated_'+hashlib.sha256(source.read_bytes()).hexdigest()[:16]+'.dylib')
    if not path.exists():
        fd, temporary = tempfile.mkstemp(dir=build, suffix='.dylib')
        os.close(fd)
        try:
            subprocess.run(['clang', '-O2', '-fobjc-arc', '-Wall', '-Werror', '-dynamiclib', '-framework', 'Foundation',
                            '-framework', 'Metal', str(source), '-o', temporary], check=True, capture_output=True, timeout=60)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    lib = ctypes.CDLL(str(path))
    u32, u64, i32, P = ctypes.c_uint32, ctypes.c_uint64, ctypes.c_int32, ctypes.POINTER
    lib.pqcp_trunc_gpu_create.argtypes = [ctypes.c_char_p, ctypes.c_int]
    lib.pqcp_trunc_gpu_create.restype = ctypes.c_void_p
    lib.pqcp_trunc_gpu_scan.argtypes = [ctypes.c_void_p, u32, u32, u32, u32, P(u64), P(u32), P(i32), P(i32), P(u32),
                                        P(u32), P(u64), P(ctypes.c_int8), P(u32), P(u32), P(u64), ctypes.c_int, P(u64)]
    lib.pqcp_trunc_gpu_scan.restype = ctypes.c_int
    lib.pqcp_trunc_gpu_destroy.argtypes = [ctypes.c_void_p]
    return lib


def pack_job(d, supports_b, wanted, negatives_b, patterns):
    """One C_A support's GPU input: supports_b as raw uint64 bytes (family_supports(raw=True))."""
    H = d//2
    q1 = array('I', (sum(((p[u] >> 1) & 1) << u for u in range(H)) for p in patterns))
    q0 = sum((patterns[0][u] & 1) << u for u in range(H))
    return dict(sb=supports_b, T=array('i', (wanted[u] for u in range(1, H+1))).tobytes(), nB=negatives_b,
                M=len(patterns), Q1=q1.tobytes(), Q0=q0, RA=array('b', (v for p in patterns for v in p)).tobytes())


class GpuTruncated:
    def __init__(self, d):
        self.d, self.lib = d, _library()
        error = ctypes.create_string_buffer(512)
        self.handle = self.lib.pqcp_trunc_gpu_create(error, len(error))
        if not self.handle:
            raise RuntimeError('Metal setup failed: '+error.value.decode(errors='replace'))

    def scan(self, jobs, max_out=1 << 16):
        """jobs: pack_job dicts -> ([[(S_B, s, x)] per job], wide kernels skipped)."""
        H, n_sa = self.d//2, len(jobs)
        mmax = max(job['M'] for job in jobs)
        counts = [len(job['sb'])//8 for job in jobs]
        n = sum(counts)
        sai = array('I')
        for j, c in enumerate(counts):
            sai.extend(array('I', [j])*c)
        u32, u64, i32, i8 = ctypes.c_uint32, ctypes.c_uint64, ctypes.c_int32, ctypes.c_int8
        sb = (u64*n).from_buffer_copy(b''.join(job['sb'] for job in jobs))
        Q1 = (u32*(n_sa*mmax)).from_buffer_copy(b''.join(job['Q1']+bytes(4*(mmax-job['M'])) for job in jobs))
        RA = (i8*(n_sa*mmax*H)).from_buffer_copy(b''.join(job['RA']+bytes(H*(mmax-job['M'])) for job in jobs))
        T = (i32*(n_sa*H)).from_buffer_copy(b''.join(job['T'] for job in jobs))
        nB = (i32*n_sa)(*(job['nB'] for job in jobs))
        M = (u32*n_sa)(*(job['M'] for job in jobs))
        Q0 = (u64*n_sa)(*(job['Q0'] for job in jobs))
        index, s_out, x_out, stats = (u32*max_out)(), (u32*max_out)(), (u64*max_out)(), (u64*1)()
        found = self.lib.pqcp_trunc_gpu_scan(self.handle, n, n_sa, self.d, mmax, sb,
                                             (u32*n).from_buffer(sai), T, nB, M, Q1, Q0, RA,
                                             index, s_out, x_out, max_out, stats)
        if found < 0 or found > max_out:
            raise RuntimeError('GPU truncated scan failed: {}'.format(found))
        out = [[] for _ in jobs]
        for i in range(found):
            out[sai[index[i]]].append((sb[index[i]], s_out[i], x_out[i]))
        return out, stats[0]

    def __del__(self):
        if getattr(self, 'handle', None):
            self.lib.pqcp_trunc_gpu_destroy(self.handle)
            self.handle = None
