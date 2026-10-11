"""Exact anchor equivalence for ALL strict full-target locations.

Fixed-C_A completion depends only on its integer norm and support: C_B
norm/counts and all G domains follow from these. Homometric C_A words with
the same support may therefore share a whole-anchor existence decision.
Independent A rotation and a COMMON odd-unit decimation give the remaining
equivalences. A k-specific or UNKNOWN result cannot close another anchor.
"""
import ctypes
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from .coarse_row_completion import validate_coarse_row


@lru_cache(maxsize=1)
def library():
    root=Path(__file__).resolve().parents[1];source=root/'csrc/coarse_anchor_orbits.c'
    build=root/'build';build.mkdir(exist_ok=True)
    suffix='.dylib' if sys.platform=='darwin' else '.so'
    digest=hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    path=build/('coarse_anchor_orbits_'+digest+suffix)
    if not path.exists():
        fd,temporary=tempfile.mkstemp(dir=build,suffix=suffix);os.close(fd)
        try:
            subprocess.run(['cc','-O3','-std=c11','-Wall','-Wextra','-Werror','-fPIC',
                '-dynamiclib' if sys.platform=='darwin' else '-shared',str(source),'-o',temporary],
                check=True,capture_output=True,timeout=30)
            os.replace(temporary,path)
        finally:
            if os.path.exists(temporary):os.unlink(temporary)
    result=ctypes.CDLL(str(path));u=ctypes.c_uint64;i=ctypes.c_int
    result.pqcp_coarse_anchor_key.argtypes=[u,u,i,ctypes.POINTER(i),ctypes.POINTER(u),ctypes.POINTER(i)]
    result.pqcp_coarse_anchor_key.restype=i
    result.pqcp_coarse_word_key.argtypes=[u,u,i,ctypes.POINTER(u)]
    result.pqcp_coarse_word_key.restype=i
    return result


def coarse_anchor_key(row):
    row=validate_coarse_row(row);d=len(row)
    support=sum((v!=0)<<j for j,v in enumerate(row));minus=sum((v==-1)<<j for j,v in enumerate(row))
    i=ctypes.c_int;u=ctypes.c_uint64;profile=(i*d)();mask=u();action=(i*2)()
    if library().pqcp_coarse_anchor_key(support,minus,d,profile,ctypes.byref(mask),action):
        raise RuntimeError('native coarse-anchor canonicalization failed')
    return tuple(map(int,profile)),int(mask.value)


def coarse_anchor_key_with_action(row):
    """Diagnostic action: C'_j=C_{unit*(j+offset) mod d}."""
    row=validate_coarse_row(row);d=len(row)
    support=sum((v!=0)<<j for j,v in enumerate(row));minus=sum((v==-1)<<j for j,v in enumerate(row))
    i=ctypes.c_int;u=ctypes.c_uint64;profile=(i*d)();mask=u();action=(i*2)()
    if library().pqcp_coarse_anchor_key(support,minus,d,profile,ctypes.byref(mask),action):
        raise RuntimeError('native coarse-anchor canonicalization failed')
    return (tuple(map(int,profile)),int(mask.value)),tuple(map(int,action))


def coarse_word_orbit_key(row):
    """Literal signed-word isometry key; unlike homometry, preserves L1 balls."""
    row=validate_coarse_row(row);d=len(row)
    support=sum((v!=0)<<j for j,v in enumerate(row));minus=sum((v==-1)<<j for j,v in enumerate(row))
    u=ctypes.c_uint64;key=(u*2)()
    if library().pqcp_coarse_word_key(support,minus,d,key):
        raise RuntimeError('native coarse-word orbit canonicalization failed')
    return tuple(map(int,key))
