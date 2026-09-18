"""Fixed dual-compression rectangle completion, accelerated by a small C kernel.

For L=4n, n odd, cell (r,c) is the unique i with i mod 4=r and i mod n=c.
Column sums fix factor-4 compression; row sums fix factor-n compression.
Switching a checkerboard 2x2 rectangle preserves every margin and therefore
ordinary and even/odd weights. It does NOT generally preserve factor-2
compression or the full PACF. The latter is the integer search energy.

This bounded heuristic may find a verified solution, but a miss proves
nothing. It never supplies an UNSAT result or restricts subsequent Z3 lifts.
Initial matrices use only the prescribed margins, not previously found PQCPs.
"""

import ctypes
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
from time import perf_counter
from typing import Optional, Sequence, Tuple

from .correlation import full_correlation_profile, normalize_binary_sequence
from .cross_compression import binary_margins_feasible
from .structured_energy import compress_binary_sequence, full_error_energy, target_profile
from .target_profiles import TargetContentProfile, pair_content
from .verifier import verify_pqcp

Word = Tuple[int, ...]


def crt_position(n: int, row: int, column: int) -> int:
    """Exact CRT position; never row-major or adjacent-pair expansion."""
    if type(n) is not int or n < 1 or n % 2 == 0:
        raise ValueError("CRT rectangle needs positive odd n")
    if type(row) is not int or type(column) is not int or not 0 <= row < 4 or not 0 <= column < n:
        raise ValueError("invalid CRT cell")
    return column + n * (((row-column)*pow(n,-1,4)) % 4)


def initialize_from_margins(columns: Sequence[int], rows: Sequence[int], seed: int) -> Word:
    """Realize labeled sign margins by bipartite Havel--Hakimi.

    Process column degrees in decreasing order and connect each to the rows
    of greatest remaining degree. This standard degree construction preserves
    realizability for graphical margins. Random ties diversify starts; the
    resulting distribution is NOT claimed to be uniform over all matrices.
    """
    ca, cross = tuple(columns), tuple(rows)
    n = len(ca)
    if (not n or n % 2 == 0 or len(cross) != 4
            or any(type(x) is not int or x not in (-4,-2,0,2,4) for x in ca)
            or any(type(x) is not int or abs(x)>n or (n-x)%2 for x in cross)):
        raise ValueError("invalid factor-4 / factor-n margins")
    if type(seed) is not int:
        raise ValueError("seed must be integer")
    degrees = [(4-x)//2 for x in ca]
    remaining = [(n-x)//2 for x in cross]
    if not binary_margins_feasible(remaining,degrees):
        raise ValueError("infeasible binary matrix margins")
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    order.sort(key=lambda c: degrees[c], reverse=True)
    bits = [0]*(4*n)
    for column in order:
        indices = list(range(4))
        rng.shuffle(indices)
        indices.sort(key=lambda r: remaining[r],reverse=True)
        for row in indices[:degrees[column]]:
            if remaining[row] <= 0:
                raise RuntimeError("margin construction invariant failed")
            bits[crt_position(n,row,column)] = 1
            remaining[row] -= 1
    if any(remaining):
        raise RuntimeError("margin construction left unmatched degrees")
    return tuple(bits)


@lru_cache(maxsize=4)
def _library(build_directory: Optional[str] = None):
    """Compile/cache by source hash on macOS/Linux; CDLL releases the GIL."""
    root = Path(__file__).resolve().parent.parent
    source = root/"csrc/rectangle_search.c"
    build = Path(build_directory) if build_directory else root/"build"
    signature = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    path = build/("pqcp_rectangle_"+signature+suffix)
    if not path.exists():
        compiler = shutil.which("cc")
        if compiler is None:
            raise RuntimeError("rectangle search needs a C compiler; use --no-z3-rectangle-search to disable")
        build.mkdir(parents=True,exist_ok=True)
        fd, name = tempfile.mkstemp(prefix="rectangle_",suffix=suffix,dir=str(build))
        os.close(fd)
        try:
            flags = ["-dynamiclib"] if sys.platform == "darwin" else ["-shared"]
            result = subprocess.run([compiler,"-O3","-std=c11","-fPIC",*flags,
                                     str(source),"-o",name,"-lm"],capture_output=True,text=True)
            if result.returncode:
                raise RuntimeError("rectangle kernel compilation failed: "+result.stderr)
            os.replace(name,path)
        finally:
            if os.path.exists(name):
                os.unlink(name)
    lib = ctypes.CDLL(str(path.resolve()))
    ip = ctypes.POINTER(ctypes.c_int)
    lib.pqcp_rectangle_delta.argtypes = [ctypes.c_int,ip,*([ctypes.c_int]*4),ip]
    lib.pqcp_rectangle_delta.restype = ctypes.c_int
    lib.pqcp_rectangle_search.argtypes = [ctypes.c_int,ctypes.c_int,ctypes.c_int,ip,ip,
                                        ctypes.c_uint64,ctypes.c_int,ctypes.c_int,ip,
                                        ctypes.POINTER(ctypes.c_int64)]
    lib.pqcp_rectangle_search.restype = ctypes.c_int
    return lib


def rectangle_correlation_delta(sequence: Sequence[int], rows: Tuple[int,int],
                                columns: Tuple[int,int]) -> Word:
    """C integer delta for a legal switch; all cyclic shifts including L/2.

    Delta P(u)=-2*sum_{i: exactly one of i,i+u flipped} x[i]*x[i+u].
    Internal rectangle edges have BOTH endpoints flipped and contribute zero.
    This handles overlapping edges without a double-count correction guess.
    """
    bits = normalize_binary_sequence(sequence)
    L = len(bits)
    if L < 12 or L > 4096 or L % 8 != 4 or len(rows) != 2 or len(columns) != 2:
        raise ValueError("C rectangles require L=4n with odd n>=3 and two rows/columns")
    for row in rows:
        for col in columns:
            crt_position(L//4,row,col)
    array = ctypes.c_int*L
    out = array()
    result = _library().pqcp_rectangle_delta(L,array(*bits),*rows,*columns,out)
    if result:
        raise ValueError("not a legal checkerboard rectangle")
    return tuple(out)


@dataclass(frozen=True)
class RectangleResult:
    """Independently audited best candidate and bounded C operation counts."""

    a: Word
    b: Word
    profile: Word
    initial_energy: int
    best_energy: int
    moves: int
    evaluations: int
    accepted: int
    no_legal_moves: bool
    verified: bool
    elapsed: float


def search_rectangles(a: Sequence[int], b: Sequence[int], target: TargetContentProfile,
                      seed: int, moves: int = 1000, samples: int = 8) -> RectangleResult:
    """One bounded C restart in this pair's fixed row/column margins.

    Maintains exact full-profile integers via O(L) four-flip deltas. Energy is
    raw E_full=sum_{u=1}^{L/2}(S(u)-T(u))^2, not pqcp_objective's score.
    On return independently audit both compressions, content, profile, energy;
    E=0 additionally requires the unchanged verifier and exact target PASS.
    """
    started = perf_counter()
    a,b = normalize_binary_sequence(a),normalize_binary_sequence(b)
    L = len(a)
    if len(b) != L or L < 12 or L > 4096 or L % 8 != 4:
        raise ValueError("C rectangles require L=4n with odd n>=3")
    if not isinstance(target,TargetContentProfile) or target.L != L:
        raise ValueError("target length mismatch")
    desired = target_profile(L,target.k,target.eta)
    content = (target.a_even_ones,target.a_odd_ones,target.b_even_ones,target.b_odd_ones)
    if pair_content(a,b) != content:
        raise ValueError("candidate does not match target content")
    if (type(seed) is not int or type(moves) is not int or not 0 <= moves <= 2**31-1
            or type(samples) is not int or not 1 <= samples <= 1024):
        raise ValueError("invalid rectangle seed/moves/samples")
    compressions = tuple(compress_binary_sequence(word,f) for word in (a,b) for f in (4,L//4))
    initial = full_error_energy(full_correlation_profile(a,b),target.k,target.eta)
    array = ctypes.c_int*L
    aa,bb,profile,stats = array(*a),array(*b),array(),(ctypes.c_int64*6)()
    code = _library().pqcp_rectangle_search(L,target.k,target.eta,aa,bb,
                                          seed % 2**64,moves,samples,profile,stats)
    if code:
        raise RuntimeError("rectangle kernel returned error {}".format(code))
    a,b = tuple(aa),tuple(bb)
    observed = tuple(full_correlation_profile(a,b))
    energy = full_error_energy(observed,target.k,target.eta)
    if (tuple(profile) != observed or stats[3] != initial or stats[4] != energy
            or energy > initial or pair_content(a,b) != content
            or tuple(compress_binary_sequence(word,f) for word in (a,b) for f in (4,L//4)) != compressions):
        raise RuntimeError("rectangle kernel failed independent compression/profile/energy audit")
    verified = False
    if energy == 0:
        verified = verify_pqcp(a,b).is_valid and observed == desired
        if not verified:
            raise RuntimeError("rectangle E=0 failed independent verifier")
    return RectangleResult(a,b,observed,initial,energy,int(stats[0]),int(stats[1]),
                           int(stats[2]),bool(stats[5]),verified,perf_counter()-started)
