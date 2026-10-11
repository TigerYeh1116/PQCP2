"""Native exact GF2/cardinality plan construction using the audited sign kernels."""
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
    sources = [root/'csrc'/name for name in ('ternary_system.c', 'ternary_signs.c', 'ternary_weight_signs.c',
                                          'ternary_quadratic_signs.c')]
    digest = hashlib.sha256(b''.join(hashlib.sha256(p.read_bytes()).digest() for p in sources)).hexdigest()[:16]
    build = root/'build'
    build.mkdir(exist_ok=True)
    suffix = '.dylib' if sys.platform == 'darwin' else '.so'
    path = build/('ternary_system_'+digest+suffix)
    if not path.exists():
        fd, temporary = tempfile.mkstemp(dir=build, suffix=suffix)
        os.close(fd)
        try:
            subprocess.run(['cc', '-O3', '-std=c11', '-Wall', '-Wextra', '-Werror', '-fPIC',
                '-dynamiclib' if sys.platform == 'darwin' else '-shared', *(str(p) for p in sources), '-o', temporary],
                check=True, capture_output=True, timeout=30)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    lib = ctypes.CDLL(str(path))
    p = ctypes.POINTER(ctypes.c_uint64)
    ip = ctypes.POINTER(ctypes.c_int)
    lib.pqcp_affine_sign_plan.argtypes = [p, ctypes.c_int, ip, ip, ctypes.c_int, ctypes.c_int, p, p, p, p, ip, ip]
    lib.pqcp_affine_sign_plan.restype = ctypes.c_int
    lib.pqcp_ternary_signs.argtypes = [p, p, p, ctypes.c_int, ctypes.c_int, ip, ip, ctypes.c_uint64, p, p]
    lib.pqcp_ternary_signs.restype = ctypes.c_int
    lib.pqcp_quadratic_signs.argtypes = lib.pqcp_ternary_signs.argtypes
    lib.pqcp_quadratic_signs.restype = ctypes.c_int
    lib.pqcp_quadratic_hybrid_signs.argtypes = [p, p, p, ctypes.c_int, ctypes.c_int, ip, ip,
                                              ctypes.c_int, ctypes.c_uint64, p, p]
    lib.pqcp_quadratic_hybrid_signs.restype = ctypes.c_int
    lib.pqcp_quadratic_orbit_hybrid_signs.argtypes = lib.pqcp_quadratic_hybrid_signs.argtypes
    lib.pqcp_quadratic_orbit_hybrid_signs.restype = ctypes.c_int
    if hasattr(lib,'pqcp_quadratic_count_hybrid_signs'):
        lib.pqcp_quadratic_count_hybrid_signs.argtypes=lib.pqcp_quadratic_hybrid_signs.argtypes
        lib.pqcp_quadratic_count_hybrid_signs.restype=ctypes.c_int
    qp = ctypes.POINTER(ctypes.c_uint32)
    lib.pqcp_quadratic_plan.argtypes = [p, p, p, ctypes.c_int, ctypes.c_int, ip, ip, qp, qp, qp]
    lib.pqcp_quadratic_plan.restype = ctypes.c_int
    lib.pqcp_weight_signs.argtypes = [p, p, p, ctypes.c_int, ctypes.c_int, p, ip, ctypes.c_int,
        p, ctypes.c_int, ctypes.c_int, ip, ip, ctypes.c_uint64, p, p]
    lib.pqcp_weight_signs.restype = ctypes.c_int
    lib.pqcp_weight_pattern_count.argtypes = [ctypes.c_int, ctypes.c_int, p, ip, ctypes.c_int]
    lib.pqcp_weight_pattern_count.restype = ctypes.c_uint64
    lib.pqcp_weight_quadratic_signs.argtypes = lib.pqcp_weight_signs.argtypes[:-3]+[ctypes.c_int]+lib.pqcp_weight_signs.argtypes[-3:]
    lib.pqcp_weight_quadratic_signs.restype = ctypes.c_int
    return lib


def prepare_native_system(masks, d, wanted, negative, *, conditioning=True, force_side=None, cardinality_pruning=True):
    if (type(d) is not int or not 3 <= d <= 47 or not d % 2 or len(masks) != 2
            or any(type(m) is not int or not 0 <= m < 1 << d for m in masks)
            or sum(bin(m).count('1') for m in masks) > 47 or len(wanted) != d
            or any(type(v) is not int or abs(v) > 2*d for v in wanted)
            or len(negative) != 2 or any(type(n) is not int or not 0 <= n <= bin(m).count('1')
                                       for n, m in zip(negative, masks)) or type(conditioning) is not bool
            or type(cardinality_pruning) is not bool
            or (force_side is not None and (type(force_side) is not int or force_side not in (0, 1)))):
        raise ValueError('odd supports with <=47 variables, integral targets and counts required')
    started = perf_counter()
    u = ctypes.c_uint64
    integer = ctypes.c_int
    lib = _library()
    mask_array, wanted_array, negative_array = (u*2)(*masks), (integer*d)(*wanted), (integer*2)(*negative)
    choices = sorted((comb(bin(m).count('1'), n), side) for side, (m, n) in enumerate(zip(masks, negative)))
    selected = choices[0][1] if conditioning and choices[0][0] <= 4096 else -1
    if force_side is not None:
        if comb(bin(masks[force_side]).count('1'), negative[force_side]) > 4096:
            raise ValueError('forced conditioning exceeds pattern budget')
        selected = force_side
    base, toggles, patterns, checks, rhs, meta = (u*2)(), (u*94)(), (u*94)(), (u*96)(), (integer*96)(), (integer*8)()
    status = lib.pqcp_affine_sign_plan(mask_array, d, wanted_array, negative_array, selected, int(cardinality_pruning),
                                     base, toggles, patterns, checks, rhs, meta)
    if status < 0:
        raise RuntimeError('native GF2 plan construction failed')
    compatible = None
    if status and selected >= 0 and cardinality_pruning:
        compatible = int(lib.pqcp_weight_pattern_count(meta[5], negative[selected], checks, rhs, meta[6]))
        if compatible == 2**64-1:
            raise RuntimeError('native projected cardinality count failed')
        if compatible == 0:
            status = 0
            meta[7] = 3
    cost_patterns = choices[0][0] if compatible is None else compatible
    if status and selected >= 0 and force_side is None and (meta[4] < 7 or 5*cost_patterns > 4*(1 << meta[3])):
        selected = -1
        status = lib.pqcp_affine_sign_plan(mask_array, d, wanted_array, negative_array, selected, int(cardinality_pruning),
                                         base, toggles, patterns, checks, rhs, meta)
        if status < 0:
            raise RuntimeError('native raw GF2 plan construction failed')
    return dict(consistent=bool(status), masks=tuple(masks), d=d, base=base, toggles=toggles,
        patterns=patterns, checks=checks, rhs=rhs, meta=meta, selected=selected,
        wanted=wanted_array, negative=negative_array, mask_array=mask_array,
        cardinality_contradiction=meta[7] in (2, 3), compatible_pattern_count=compatible,
        estimated_assignments=(0 if not status else
            ((comb(bin(masks[selected]).count('1'), negative[selected]) if compatible is None else compatible)*(1 << meta[4])
             if selected >= 0 else 1 << meta[2])),
        construction_seconds=perf_counter()-started)


def run_native_system(plan, max_assignments, *, quadratic=None, hybrid_tail=None, conditioned_hybrid=None,even_sign_orbit=False,weight_prefix=False):
    if type(max_assignments) is not int or not 1 <= max_assignments <= 2**31:
        raise ValueError('assignment budget in 1..2**31 required')
    if quadratic is not None and type(quadratic) is not bool:
        raise ValueError('quadratic must be boolean or None (automatic)')
    if hybrid_tail is not None and (type(hybrid_tail) is not int or not 0 <= hybrid_tail <= 8):
        raise ValueError('hybrid_tail must be an integer in 0..8')
    if conditioned_hybrid is not None and type(conditioned_hybrid) is not bool:
        raise ValueError('conditioned_hybrid must be boolean or None (automatic)')
    if type(even_sign_orbit) is not bool:raise ValueError('even_sign_orbit must be boolean')
    if type(weight_prefix) is not bool:raise ValueError('weight_prefix must be boolean')
    if (weight_prefix and quadratic is None and hybrid_tail is None and conditioned_hybrid is None
            and not even_sign_orbit):
        from .weight_prefix_native import run_weight_prefix
        result=run_weight_prefix(plan,max_assignments)
        if result is not None:return result
    common = dict(rank=plan['meta'][1], free_variables=plan['meta'][2], words=None,
                  construction_seconds=plan['construction_seconds'], solve_seconds=0.0, assignments=0)
    if not plan['consistent']:
        return dict(common, status='UNSAT_FIXED_SUPPORT',
                    reason='GF2 forced cardinality contradiction' if plan.get('cardinality_contradiction') else 'native GF2 / range obstruction')
    lib = _library()
    u = ctypes.c_uint64
    output, stats = (u*2)(), (u*9)()
    m = plan['meta']
    if conditioned_hybrid is None:
        conditioned_hybrid = (quadratic is not False and plan['d'] == 43 and plan['selected'] >= 0
            and 7 <= m[4] <= 24 and comb(m[5], plan['negative'][plan['selected']]) <= 4096)
    if hybrid_tail is None:
        hybrid_tail = 6 if quadratic is None and plan['d'] == 43 and 16 <= m[2] <= 24 else 0
    started = perf_counter()
    if plan['selected'] < 0:
        enabled = plan['d'] == 43 and 16 <= m[2] <= 24 if quadratic is None else quadratic
        filtered = enabled and m[2] <= 24
        kernel = lib.pqcp_quadratic_signs if filtered else lib.pqcp_ternary_signs
        if filtered and hybrid_tail:
            hybrid_kernel=lib.pqcp_quadratic_orbit_hybrid_signs if even_sign_orbit else lib.pqcp_quadratic_hybrid_signs
            bit_tail=(not even_sign_orbit and hasattr(lib,'pqcp_quadratic_bit_tail_signs') and
                      plan['d']==43 and 0<=m[2]<=24 and 1<=hybrid_tail<=6 and max_assignments>=1<<m[2])
            status = hybrid_kernel(plan['mask_array'], plan['base'], plan['toggles'], m[2], plan['d'],
                plan['wanted'], plan['negative'], hybrid_tail, max_assignments, output, stats)
            common.update(prefix_branches=int(stats[4]), projected_candidates=int(stats[5]),
                          maximum_tail_nullity=int(stats[6]), hybrid_tail=hybrid_tail)
            common.update(hybrid_backend='exact_bit_tail' if bit_tail else 'projected_linear_tail',
                          candidate_counter_scope='original_mod8_models' if bit_tail else 'projected_tail_models')
            if even_sign_orbit:
                common.update(even_sign_orbit_pruning=bool(stats[7]),orbit_side=int(stats[7])-1,
                              covered_sign_orbits=int(stats[0])//2 if stats[7] else 0,
                              orbit_tail_variables=int(stats[8]),orbit_reduced_free_variables=m[2]-bool(stats[7]))
        else:
            status = kernel(plan['mask_array'], plan['base'], plan['toggles'], m[2], plan['d'],
                plan['wanted'], plan['negative'], max_assignments, output, stats)
        if filtered:
            assignments, matches, correlations = stats[0], stats[2], stats[3]
            common['quadratic_matches'] = int(stats[1])
        else:
            assignments, matches, correlations = stats[0], stats[1], stats[2]
    else:
        arguments = (plan['mask_array'], plan['base'], plan['patterns'], m[5], plan['selected'],
            plan['checks'], plan['rhs'], m[6], plan['toggles'], m[4], plan['d'], plan['wanted'], plan['negative'])
        if conditioned_hybrid and m[4] <= 24:
            status = lib.pqcp_weight_quadratic_signs(*arguments, hybrid_tail or 6, max_assignments, output, stats)
            common.update(prefix_branches=int(stats[5]), projected_candidates=int(stats[6]),
                          quadratic_matches=int(stats[7]), maximum_tail_nullity=int(stats[8]),
                          conditioned_hybrid=True)
        else:
            status = lib.pqcp_weight_signs(*arguments, max_assignments, output, stats)
        assignments, matches, correlations = stats[2], stats[3], stats[4]
        common.update(conditioned_side=plan['selected'], projection_rank=m[3], branch_free_variables=m[4],
                      weight_patterns=int(stats[0]), compatible_patterns=int(stats[1]))
    if status not in (0, 1, 2):
        raise RuntimeError('native planned sign enumeration failed')
    words = None if status != 1 else tuple(tuple(0 if not plan['masks'][s] >> i & 1 else
        -1 if output[s] >> i & 1 else 1 for i in range(plan['d'])) for s in range(2))
    common.update(status=('UNSAT_FIXED_SUPPORT', 'SAT_COMPRESSED', 'UNKNOWN')[status], words=words,
        assignments=int(assignments), count_matches=int(matches), correlation_checks=int(correlations),
        solve_seconds=perf_counter()-started, reason='assignment cap' if status == 2 else None)
    return common
