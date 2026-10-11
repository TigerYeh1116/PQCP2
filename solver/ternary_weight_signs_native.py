"""Exact cardinality-conditioned native enumeration of coarse sign solutions."""
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
    source = root/'csrc/ternary_weight_signs.c'
    build = root/'build'
    build.mkdir(exist_ok=True)
    suffix = '.dylib' if sys.platform == 'darwin' else '.so'
    path = build/('ternary_weight_signs_'+hashlib.sha256(source.read_bytes()).hexdigest()[:16]+suffix)
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
    pointer = ctypes.POINTER(ctypes.c_uint64)
    ipointer = ctypes.POINTER(ctypes.c_int)
    lib.pqcp_weight_signs.argtypes = [pointer, pointer, pointer, ctypes.c_int, ctypes.c_int,
        pointer, ipointer, ctypes.c_int, pointer, ctypes.c_int, ctypes.c_int,
        ipointer, ipointer, ctypes.c_uint64, pointer, pointer]
    lib.pqcp_weight_signs.restype = ctypes.c_int
    lib.pqcp_weight_pattern_count.argtypes = [ctypes.c_int, ctypes.c_int, pointer, ipointer, ctypes.c_int]
    lib.pqcp_weight_pattern_count.restype = ctypes.c_uint64
    return lib


def compatible_weight_patterns(plan, negative):
    """Exact count of projected patterns with this row's prescribed weight."""
    u, integer = ctypes.c_uint64, ctypes.c_int
    masks = (u*len(plan.checks))(*(m for m, _ in plan.checks))
    rhs = (integer*len(plan.checks))(*(r for _, r in plan.checks))
    result = int(_library().pqcp_weight_pattern_count(len(plan.pattern_variables), negative, masks, rhs, len(plan.checks)))
    if result == 2**64-1:
        raise RuntimeError('projected cardinality count failed')
    return result


def enumerate_weight_signs(system, plan, masks, d, wanted, negative, max_assignments, *,
                           conditioned_hybrid=None, hybrid_tail=None):
    """All constant-weight projected patterns and all their affine preimages.

    Budget counts tested preimages, not the old unconstrained affine space.
    UNKNOWN cannot eliminate any untested support or sign completion.
    """
    if (not system.consistent or type(d) is not int or not 3 <= d <= 47 or not d % 2
            or type(max_assignments) is not int or not 1 <= max_assignments <= 2**31
            or len(plan.expressions) != len(system.variables)
            or system.variables != tuple((s, i) for s in range(2) for i in range(d) if masks[s] >> i & 1)
            or any(not 0 <= n <= bin(mask).count('1') for n, mask in zip(negative, masks))):
        raise ValueError('valid support system, conditioning plan, counts and assignment budget required')
    started = perf_counter()
    pattern_count = len(plan.pattern_variables)
    if conditioned_hybrid is not None and type(conditioned_hybrid) is not bool:
        raise ValueError('conditioned_hybrid must be boolean or None (automatic)')
    if hybrid_tail is not None and (type(hybrid_tail) is not int or not 0 <= hybrid_tail <= 8):
        raise ValueError('hybrid_tail must be None or an integer in 0..8')
    if conditioned_hybrid is None:
        conditioned_hybrid = d == 43 and 12 <= plan.free_count <= 24 and comb(pattern_count, negative[plan.side]) <= 64
    filtered = conditioned_hybrid and plan.free_count <= 24
    base = [0, 0]
    patterns = [0]*(2*pattern_count)
    free = [0]*(2*plan.free_count)
    for (side, position), (gm, tm, constant) in zip(system.variables, plan.expressions):
        base[side] |= constant << position
        for bit in range(pattern_count):
            if tm >> bit & 1:
                patterns[2*bit+side] |= 1 << position
        for bit in range(plan.free_count):
            if gm >> bit & 1:
                free[2*bit+side] |= 1 << position
    if filtered:
        from .ternary_system_native import _library as planned_library
        lib = planned_library()
    else:
        lib = _library()
    array = ctypes.c_uint64
    integer = ctypes.c_int
    output = (array*2)()
    stats = (array*9)()
    construction = perf_counter()-started
    started = perf_counter()
    arguments = ((array*2)(*masks), (array*2)(*base),
        (array*len(patterns))(*patterns), pattern_count, plan.side,
        (array*len(plan.checks))(*(mask for mask, _ in plan.checks)),
        (integer*len(plan.checks))(*(rhs for _, rhs in plan.checks)), len(plan.checks),
        (array*len(free))(*free), plan.free_count, d, (integer*d)(*wanted),
        (integer*2)(*negative))
    if filtered:
        status = lib.pqcp_weight_quadratic_signs(*arguments, hybrid_tail or 6, max_assignments, output, stats)
    else:
        status = lib.pqcp_weight_signs(*arguments, max_assignments, output, stats)
    elapsed = perf_counter()-started
    if status not in (0, 1, 2):
        raise RuntimeError('cardinality-conditioned sign enumeration failed: {}'.format(status))
    words = None
    if status == 1:
        words = tuple(tuple(0 if not masks[s] >> i & 1 else -1 if output[s] >> i & 1 else 1
                            for i in range(d)) for s in range(2))
    extra = dict(prefix_branches=int(stats[5]), projected_candidates=int(stats[6]), quadratic_matches=int(stats[7]),
                 maximum_tail_nullity=int(stats[8]), conditioned_hybrid=True) if filtered else {}
    return dict(status=('UNSAT_FIXED_SUPPORT', 'SAT_COMPRESSED', 'UNKNOWN')[status], words=words,
        reason='assignment cap' if status == 2 else None, construction_seconds=construction,
        solve_seconds=elapsed, assignments=int(stats[2]), count_matches=int(stats[3]),
        correlation_checks=int(stats[4]), conditioned_side=plan.side,
        projection_rank=plan.projection_rank, branch_free_variables=plan.free_count,
        weight_patterns=int(stats[0]), compatible_patterns=int(stats[1]),
        raw_affine_size=1 << len(system.free_columns), **extra)
