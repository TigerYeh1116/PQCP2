"""Optional exact C scan, with bounded calls that release the Python GIL.

The native kernel changes neither Gray order nor PACF conditions. Libraries
are keyed by source hash and built atomically. Missing toolchains/platform
support only disable acceleration; they never constitute UNSAT evidence.
"""

import ctypes
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import warnings
from typing import Optional, Sequence, Tuple


class NativeUnavailable(RuntimeError):
    """The Python reference backend remains usable."""


@lru_cache(maxsize=1)
def _library():
    root = Path(__file__).resolve().parents[1]
    source = root/'csrc/affine_scan.c'
    if sys.platform not in ('darwin','linux'):
        raise NativeUnavailable('native affine kernel supports macOS/Linux')
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    suffix = '.dylib' if sys.platform=='darwin' else '.so'
    build = root/'build'
    path = build/('pqcp_affine_'+digest+suffix)
    if not path.exists():
        compiler = shutil.which('cc')
        if compiler is None:
            raise NativeUnavailable('C compiler unavailable')
        build.mkdir(parents=True,exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix='affine_',suffix=suffix,dir=str(build))
        os.close(fd)
        try:
            flags = ['-dynamiclib'] if sys.platform=='darwin' else ['-shared']
            run = subprocess.run([compiler,'-O3','-std=c11','-fPIC',*flags,
                                  str(source),'-o',temporary],capture_output=True,text=True,timeout=30)
            if run.returncode:
                raise NativeUnavailable('affine kernel compilation failed: '+run.stderr)
            os.replace(temporary,path)
        except subprocess.TimeoutExpired as error:
            raise NativeUnavailable('affine kernel compilation timed out') from error
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    lib = ctypes.CDLL(str(path.resolve()))  # NOT PyDLL: scanning releases GIL
    up = ctypes.POINTER(ctypes.c_uint64)
    lib.pqcp_affine_scan.argtypes = [ctypes.c_int,ctypes.c_int,up,up,
        ctypes.POINTER(ctypes.c_int),ctypes.c_uint64,ctypes.c_uint64,up,up]
    lib.pqcp_affine_scan.restype = ctypes.c_int
    return lib


@lru_cache(maxsize=1)
def optional_library():
    """Auto mode warns once and keeps the exact Python fallback available."""
    try:
        return _library()
    except (NativeUnavailable,OSError) as error:
        warnings.warn('Native affine acceleration unavailable; using Python: '+str(error),
                      RuntimeWarning,stacklevel=2)
        return None


class NativeAffineScan:
    """Prepared immutable arrays, reused over time-bounded Gray chunks."""

    def __init__(self, L: int, base: Tuple[int,int], deltas: Sequence[Tuple[int,int]],
                 wanted: Sequence[int], lib=None) -> None:
        if type(L) is not int or not 4<=L<=128 or L%2 or len(deltas)>62:
            raise ValueError('unsupported native scan dimensions')
        if len(wanted)!=L//2-1 or any(type(x) is not int or not 0<=x<=2*L for x in wanted):
            raise ValueError('invalid exact mismatch target')
        def pack(pair):
            if len(pair)!=2 or any(type(w) is not int or not 0<=w<(1<<L) for w in pair):
                raise ValueError('invalid native bitset')
            mask=(1<<64)-1
            return pair[0]&mask,pair[0]>>64,pair[1]&mask,pair[1]>>64
        words = tuple(v for delta in deltas for v in pack(delta))
        self.base=(ctypes.c_uint64*4)(*pack(base))
        self.deltas=(ctypes.c_uint64*max(1,len(words)))(*words)
        self.wanted=(ctypes.c_int*len(wanted))(*wanted)
        self.L,self.free_bits=L,len(deltas)
        self.lib=_library() if lib is None else lib

    def scan(self, start: int, end: int) -> Tuple[int,Optional[Tuple[int,int]]]:
        """Return (next cursor, first matching pair or None), with exact bounds."""
        if (type(start) is not int or type(end) is not int
                or not 0<=start<=end<=(1<<self.free_bits)):
            raise ValueError('invalid native scan interval')
        next_index=ctypes.c_uint64()
        found=(ctypes.c_uint64*4)()
        code=self.lib.pqcp_affine_scan(self.L,self.free_bits,self.base,self.deltas,self.wanted,
                                       start,end,ctypes.byref(next_index),found)
        cursor=next_index.value
        if (code not in (0,1) or not start<=cursor<=end
                or (code==0 and cursor!=end) or (code==1 and cursor==start)):
            raise RuntimeError('native affine kernel returned invalid status/cursor')
        pair=None if code==0 else (found[0]|(found[1]<<64),found[2]|(found[3]<<64))
        return cursor,pair
