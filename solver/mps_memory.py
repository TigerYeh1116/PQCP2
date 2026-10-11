"""Bound temporary Objective-C objects without clearing useful tensor caches.

Long-lived Python threads have no Cocoa event-loop pool to drain objects
autoreleased outside PyTorch's internal pools. Foundation pools bound those
lifetimes without touching mathematical state. CPU/CUDA paths are no-ops.
Creation and draining always use the SAME thread.
"""
from contextlib import contextmanager
import ctypes
from functools import lru_cache, wraps
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading

_local = threading.local()


@lru_cache(maxsize=1)
def _library():
    root = Path(__file__).resolve().parents[1]
    source = root/'csrc/mps_autorelease.m'
    build = root/'build'; build.mkdir(exist_ok=True)
    path = build/('mps_autorelease_'+hashlib.sha256(source.read_bytes()).hexdigest()[:16]+'.dylib')
    if not path.exists():
        fd, temporary = tempfile.mkstemp(dir=build, suffix='.dylib'); os.close(fd)
        try:
            subprocess.run(['cc', '-O2', '-Wall', '-Wextra', '-Werror', '-fno-objc-arc',
                            '-dynamiclib', '-framework', 'Foundation', str(source), '-o', temporary],
                           check=True, capture_output=True, timeout=30)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary): os.unlink(temporary)
    lib = ctypes.CDLL(str(path))
    lib.pqcp_autorelease_create.argtypes = []
    lib.pqcp_autorelease_create.restype = ctypes.c_void_p
    lib.pqcp_autorelease_drain.argtypes = [ctypes.c_void_p]
    lib.pqcp_autorelease_drain.restype = None
    return lib


@contextmanager
def autorelease_pool(device):
    if sys.platform != 'darwin' or getattr(device, 'type', device) != 'mps':
        yield
        return
    # A bounded refresh already owns a pool. Avoid redundant per-step
    # draining while commands are still in flight; release once at its end.
    if getattr(_local, 'depth', 0):
        yield
        return
    library = _library()
    pool = library.pqcp_autorelease_create()
    if not pool: raise RuntimeError('failed to create MPS autorelease pool')
    _local.depth = 1
    try:
        yield
    finally:
        try:
            library.pqcp_autorelease_drain(pool)
        finally:
            _local.depth = 0


def bounded_mps_temporaries(method):
    @wraps(method)
    def run(self, *args, **kwargs):
        if getattr(self.device, 'type', self.device) != 'mps':
            return method(self, *args, **kwargs)
        with autorelease_pool(self.device):
            return method(self, *args, **kwargs)
    return run
