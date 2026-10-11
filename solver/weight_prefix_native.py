"""Exact constant-weight prefixes with an incrementally evaluated small tail.

The production proposal targets L43 plans with at most 17 selected signs
and up to 14 free bits, using a shared <=64-member suffix table.
Explicit side selection supports small exhaustive
controls. Partial original budgets and unsupported tails return None.
"""
import ctypes
from math import comb
from time import perf_counter

from .structured_energy import integer_periodic_autocorrelation_profile as pacf
from .ternary_support_search import support_mask
from .ternary_system_native import _library

DEFAULT_COUNT_ELIMINATION=True
DEFAULT_AFFINE_PRUNING=False  # Exact, but the large matched run was slower.
DEFAULT_FIELD_REBASING=False  # Exact; no median throughput gain in the large run.
DEFAULT_MULTI_PIVOT=False  # Full tail-truth substitution; benchmark before enabling.
MAX_FREE_BITS=14


def prepare_weight_prefix(plan, budget, *, selected=None):
    if (type(budget) is not int or not 1 <= budget <= 2**31
            or (selected is not None and (type(selected) is not int or selected not in (0, 1)))):
        raise ValueError('bounded original budget and optional row 0/1 required')
    if not plan['consistent'] or budget < 1 << int(plan['meta'][2]):
        return None
    if plan['selected']>=0:
        if selected is not None and selected!=plan['selected']:return None
        if selected is None and plan['d']!=43:return None
        selected=plan['selected']
        prepared=dict(plan=plan,base=plan['base'],free=plan['toggles'],patterns=plan['patterns'],
                      checks=plan['checks'],rhs=plan['rhs'],meta=plan['meta'],selected=selected,
                      construction_seconds=0.0)
        width=int(plan['meta'][5]);free=int(plan['meta'][4])
        upper=comb(width,plan['negative'][selected])*(1<<free)
        if width>17 or free>MAX_FREE_BITS or width+free>24 or upper>budget:return None
        prepared['upper_assignments']=upper
        return prepared
    if selected is None:
        if plan['d'] != 43:
            return None
        choices = [(comb(bin(m).count('1'), plan['negative'][s]), s)
                   for s, m in enumerate(plan['masks']) if 8 <= bin(m).count('1') <= 17]
        if not choices:
            return None
        selected = min(choices)[1]
    width = bin(plan['masks'][selected]).count('1')
    if width > 17:
        return None
    tick = perf_counter()
    u, integer = ctypes.c_uint64, ctypes.c_int
    base, free, patterns, checks, rhs, meta = (u*2)(), (u*94)(), (u*94)(), (u*96)(), (integer*96)(), (integer*8)()
    status = _library().pqcp_affine_sign_plan(plan['mask_array'], plan['d'], plan['wanted'],
        plan['negative'], selected, 1, base, free, patterns, checks, rhs, meta)
    if status < 0:
        raise RuntimeError('a consistent raw plan failed its exact symbolic conditioning')
    if status == 0:return None  # An input prepared without tight counts can acquire an obstruction.
    if not 0 <= meta[4] <= MAX_FREE_BITS or width+meta[4]>24:
        return None
    upper = comb(width, plan['negative'][selected])*(1 << meta[4])
    if upper > budget:
        return None
    return dict(plan=plan, base=base, free=free, patterns=patterns, checks=checks, rhs=rhs,
                meta=meta, selected=selected, upper_assignments=upper,
                construction_seconds=perf_counter()-tick)


def native_arguments(prepared):
    p = prepared
    plan = p['plan']
    return (plan['mask_array'], p['base'], p['patterns'], p['meta'][5], p['selected'],
            p['checks'], p['rhs'], p['meta'][6], p['free'], p['meta'][4], plan['d'],
            plan['wanted'], plan['negative'])


def run_weight_prefix(plan, max_assignments, *, selected=None,count_elimination=None,affine_pruning=None,field_rebasing=None,multi_pivot=None):
    if count_elimination is None:count_elimination=DEFAULT_COUNT_ELIMINATION
    if affine_pruning is None:affine_pruning=DEFAULT_AFFINE_PRUNING and count_elimination
    if field_rebasing is None:field_rebasing=DEFAULT_FIELD_REBASING and count_elimination
    if multi_pivot is None:multi_pivot=DEFAULT_MULTI_PIVOT and count_elimination
    if type(count_elimination) is not bool:raise ValueError('boolean count-elimination flag required')
    if type(affine_pruning) is not bool:raise ValueError('boolean affine-pruning flag required')
    if type(field_rebasing) is not bool:raise ValueError('boolean field-rebasing flag required')
    if type(multi_pivot) is not bool:raise ValueError('boolean multi-pivot flag required')
    if sum((affine_pruning,field_rebasing,multi_pivot))>1:raise ValueError('select one affine backend')
    if (affine_pruning or field_rebasing or multi_pivot) and not count_elimination:
        raise ValueError('affine pruning extends the exact count-elimination backend')
    prepared = prepare_weight_prefix(plan, max_assignments, selected=selected)
    if prepared is None:
        return None
    lib = _library()
    if not hasattr(lib, 'pqcp_weight_prefix_signs'):
        return None
    fn = (lib.pqcp_weight_prefix_multi_signs if multi_pivot else
          lib.pqcp_weight_prefix_rebased_signs if field_rebasing else
          lib.pqcp_weight_prefix_affine_signs if affine_pruning else
          lib.pqcp_weight_prefix_count_signs if count_elimination else lib.pqcp_weight_prefix_signs)
    fn.argtypes = lib.pqcp_weight_signs.argtypes+([ctypes.POINTER(ctypes.c_uint64)] if count_elimination else [])
    fn.restype = ctypes.c_int
    u = ctypes.c_uint64
    output, stats,extra = (u*2)(), (u*9)(),(u*5)()
    tick = perf_counter()
    args=(*native_arguments(prepared), max_assignments, output, stats)
    status = fn(*args,extra) if count_elimination else fn(*args)
    elapsed = perf_counter()-tick
    if status not in (0, 1):
        raise RuntimeError('complete constant-weight prefix enumeration failed: {}'.format(status))
    words = None if status == 0 else tuple(tuple(
        0 if not plan['masks'][s] >> i & 1 else -1 if output[s] >> i & 1 else 1
        for i in range(plan['d'])) for s in range(2))
    if words is not None and (tuple(support_mask(w) for w in words) != plan['masks']
            or tuple(w.count(-1) for w in words) != tuple(plan['negative'])
            or tuple(x+y for x, y in zip(pacf(words[0]), pacf(words[1]))) != tuple(plan['wanted'])):
        raise RuntimeError('weight-prefix witness failed independent all-shift integer audit')
    return dict(status='SAT_COMPRESSED' if status else 'UNSAT_FIXED_SUPPORT', words=words,
        rank=int(plan['meta'][1]), free_variables=int(plan['meta'][2]),
        assignments=int(stats[2]), count_matches=int(stats[3]), correlation_checks=int(stats[4]),
        weight_patterns=int(stats[0]), compatible_patterns=int(stats[1]), prefix_branches=int(stats[5]),
        projected_candidates=int(stats[6]), prefix_bit_changes=int(stats[7]), rejected_projection_patterns=int(stats[8]),
        conditioned_side=prepared['selected'], projection_rank=int(prepared['meta'][3]),
        count_eliminated_patterns=int(extra[0]),count_contradictory_patterns=int(extra[1]),
        affine_contradictory_patterns=int(extra[2]),
        syndrome_fields_rebased=field_rebasing,
        multi_pivot_elimination=multi_pivot,multi_pivot_patterns=int(extra[3]),multi_eliminated_bits=int(extra[4]),
        branch_free_variables=int(prepared['meta'][4]), hybrid_backend='exact_weight_prefix',
        candidate_counter_scope='Original modular models after exact selected-row weight and projected checks',
        construction_seconds=plan['construction_seconds']+prepared['construction_seconds'],
        solve_seconds=elapsed, reason=None)
