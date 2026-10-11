"""Integer bit-parallel enumeration of an exact GF(2) sign parametrization."""
import ctypes
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from time import perf_counter


@lru_cache(maxsize=1)
def _library():
    root = Path(__file__).resolve().parents[1]
    source = root/'csrc/ternary_signs.c'
    build = root/'build'; build.mkdir(exist_ok=True)
    suffix = '.dylib' if sys.platform == 'darwin' else '.so'
    path = build/('ternary_signs_'+hashlib.sha256(source.read_bytes()).hexdigest()[:16]+suffix)
    if not path.exists():
        fd, temporary = tempfile.mkstemp(dir=build, suffix=suffix); os.close(fd)
        try:
            subprocess.run(['cc', '-O3', '-std=c11', '-Wall', '-Wextra', '-Werror', '-fPIC',
                '-dynamiclib' if sys.platform == 'darwin' else '-shared', str(source), '-o', temporary],
                check=True, capture_output=True, timeout=30)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary): os.unlink(temporary)
    lib = ctypes.CDLL(str(path)); ptr = ctypes.POINTER(ctypes.c_uint64); iptr = ctypes.POINTER(ctypes.c_int)
    lib.pqcp_ternary_signs.argtypes = [ptr, ptr, ptr, ctypes.c_int, ctypes.c_int, iptr, iptr,
                                     ctypes.c_uint64, ptr, ptr]
    lib.pqcp_ternary_signs.restype = ctypes.c_int
    return lib


def enumerate_signs(system, masks, d, wanted, negative, max_assignments, *, quadratic=None, hybrid_tail=None,
                    cardinality_pruning=True):
    """Caller audits hits. Exhaustion is ALL free assignments, cap is UNKNOWN."""
    if (not system.consistent or not 3 <= d <= 47 or d % 2 == 0
            or type(max_assignments) is not int or not 1 <= max_assignments <= 2**31):
        raise ValueError('consistent odd-half system and budget in 1..2**31 required')
    started = perf_counter()
    free_count = len(system.free_columns)
    if quadratic is not None and type(quadratic) is not bool:
        raise ValueError('quadratic must be boolean or None (automatic)')
    if hybrid_tail is not None and (type(hybrid_tail) is not int or not 0 <= hybrid_tail <= 8):
        raise ValueError('hybrid_tail must be None or an integer in 0..8')
    if type(cardinality_pruning) is not bool:
        raise ValueError('cardinality_pruning must be boolean')
    if hybrid_tail is None:
        hybrid_tail = 6 if quadratic is None and d == 43 and 16 <= free_count <= 24 else 0
    filtered = (d == 43 and 16 <= free_count <= 24 if quadratic is None else
                quadratic and free_count <= 24)
    initial = [0, 0]; toggles = [0]*(2*free_count)
    for (s, i), (mask, constant) in zip(system.variables, system.expressions()):
        initial[s] |= constant << i
        for bit in range(free_count):
            if mask >> bit & 1: toggles[2*bit+s] |= 1 << i
    if cardinality_pruning:
        for side in (0, 1):
            variable = 0
            for bit in range(free_count):
                variable |= toggles[2*bit+side]
            fixed = bin(initial[side] & ~variable).count('1')
            if not fixed <= negative[side] <= fixed+bin(variable).count('1'):
                return dict(status='UNSAT_FIXED_SUPPORT', words=None, assignments=0,
                    reason='GF2 forced cardinality contradiction', construction_seconds=perf_counter()-started,
                    solve_seconds=0.0, count_matches=0, correlation_checks=0)
    if filtered:
        from .ternary_system_native import _library as planned_library
        lib = planned_library()
    else:
        lib = _library()
    kernel = lib.pqcp_quadratic_signs if filtered else lib.pqcp_ternary_signs
    arr = ctypes.c_uint64
    output = (arr*2)(); stats = (arr*7)()
    construction = perf_counter()-started; started = perf_counter()
    arguments = ((arr*2)(*masks), (arr*2)(*initial), (arr*len(toggles))(*toggles),
                 free_count, d, (ctypes.c_int*d)(*wanted), (ctypes.c_int*2)(*negative))
    if filtered and hybrid_tail:
        status = lib.pqcp_quadratic_hybrid_signs(*arguments, hybrid_tail, max_assignments, output, stats)
    else:
        status = kernel(*arguments, max_assignments, output, stats)
    elapsed = perf_counter()-started
    if status not in (0, 1, 2): raise RuntimeError('native sign enumeration failed: {}'.format(status))
    words = None
    if status == 1:
        words = tuple(tuple(0 if not masks[s] >> i & 1 else -1 if output[s] >> i & 1 else 1
                            for i in range(d)) for s in range(2))
    extra = dict(quadratic_matches=int(stats[1])) if filtered else {}
    if filtered and hybrid_tail:
        extra.update(prefix_branches=int(stats[4]), projected_candidates=int(stats[5]),
                     maximum_tail_nullity=int(stats[6]), hybrid_tail=hybrid_tail)
    return dict(status=('UNSAT_FIXED_SUPPORT', 'SAT_COMPRESSED', 'UNKNOWN')[status], words=words,
                reason='assignment cap' if status == 2 else None,
                construction_seconds=construction, solve_seconds=elapsed,
                assignments=stats[0], count_matches=stats[2] if filtered else stats[1],
                correlation_checks=stats[3] if filtered else stats[2], **extra)


def prepare_sign_enumeration(system, masks, negative, *, conditioning=True, cardinality_pruning=True):
    """Return an exact conditioning plan and a conservative assignment bound."""
    plan = None
    upper = 1 << len(system.free_columns)
    if conditioning:
        from math import comb
        choices = sorted((comb(bin(mask).count('1'), n), side)
                         for side, (mask, n) in enumerate(zip(masks, negative)))
        if choices[0][0] <= 4096:
            from .affine_weight import condition_sign_row
            projected = condition_sign_row(system, choices[0][1])
            from .ternary_weight_signs_native import compatible_weight_patterns
            compatible = (compatible_weight_patterns(projected, negative[choices[0][1]])
                          if cardinality_pruning else choices[0][0])
            if (compatible == 0 or (projected.free_count >= 7 and
                    5*compatible <= 4*(1 << projected.projection_rank))):
                plan = projected
                upper = compatible*(1 << plan.free_count)
    return plan, upper


def solve_sign_system(system, masks, d, wanted, negative, max_assignments, *, conditioning=True,
                      prepared=None, quadratic=None, hybrid_tail=None, conditioned_hybrid=None, cardinality_pruning=True):
    """Choose exact cardinality conditioning or raw/mod-8 enumeration.

    quadratic=None uses mod-8 filtering for L43 raw spaces with 16..24 free
    bits. False restores the former raw kernel; budget caps stay UNKNOWN.
    """
    started = perf_counter()
    plan, _ = prepared if prepared is not None else prepare_sign_enumeration(
        system, masks, negative, conditioning=conditioning, cardinality_pruning=cardinality_pruning)
    preparation = perf_counter()-started
    if plan is None:
        out = enumerate_signs(system, masks, d, wanted, negative, max_assignments,
                              quadratic=quadratic, hybrid_tail=hybrid_tail, cardinality_pruning=cardinality_pruning)
    else:
        from .ternary_weight_signs_native import enumerate_weight_signs
        enabled = False if conditioned_hybrid is None and quadratic is False else conditioned_hybrid
        out = enumerate_weight_signs(system, plan, masks, d, wanted, negative, max_assignments,
                                     conditioned_hybrid=enabled, hybrid_tail=hybrid_tail)
    out['construction_seconds'] += preparation
    return out
