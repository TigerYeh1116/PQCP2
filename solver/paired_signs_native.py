"""Exact simultaneous bit-tail pruning of two raw affine sign cosets.

Only identical support/free bases and complete per-label budgets qualify.
Unsupported cases return None for the established independent solver.
The two labels keep separate integer targets and SAT/UNSAT decisions.
"""
import ctypes
from time import perf_counter

from .structured_energy import integer_periodic_autocorrelation_profile as pacf
from .ternary_support_search import support_mask
from .ternary_system_native import _library


def paired_arguments(plans, budget, tail=6):
    plans = tuple(plans)
    if (len(plans) != 2 or type(budget) is not int or not 1 <= budget <= 2**31
            or type(tail) is not int or not 1 <= tail <= 6):
        raise ValueError('two plans, bounded assignment budget and tail 1..6 required')
    a, b = plans
    q = int(a['meta'][2])
    if (not a['consistent'] or not b['consistent'] or a['selected'] >= 0 or b['selected'] >= 0
            or a['masks'] != b['masks'] or a['d'] != b['d'] or q != int(b['meta'][2])
            or not 0 <= q <= 24 or budget < 1 << q
            or tuple(a['toggles'][:2*q]) != tuple(b['toggles'][:2*q])):
        return None
    u = ctypes.c_uint64
    integer = ctypes.c_int
    return (a['mask_array'], (u*4)(*a['base'], *b['base']), a['toggles'], q, a['d'],
            (integer*(2*a['d']))(*a['wanted'], *b['wanted']),
            (integer*4)(*a['negative'], *b['negative']), tail)


def run_native_pair(plans, max_assignments, *, tail=6):
    plans = tuple(plans)
    args = paired_arguments(plans, max_assignments, tail)
    if args is None:
        return None
    lib = _library()
    if not hasattr(lib, 'pqcp_quadratic_paired_bit_tail_signs'):
        return None
    u = ctypes.c_uint64
    integer = ctypes.c_int
    function = lib.pqcp_quadratic_paired_bit_tail_signs
    p = ctypes.POINTER(u)
    ip = ctypes.POINTER(integer)
    function.argtypes = [p, p, p, integer, integer, ip, ip, integer, u, p, ip, p]
    function.restype = integer
    output, statuses, stats = (u*4)(), (integer*2)(), (u*14)()
    tick = perf_counter()
    code = function(*args, max_assignments, output, statuses, stats)
    elapsed = perf_counter()-tick
    if code or any(status not in (0, 1) for status in statuses):
        raise RuntimeError('paired exact enumeration failed: {}'.format(code))
    results = []
    for k, plan in enumerate(plans):
        words = None if statuses[k] == 0 else tuple(tuple(
            0 if not plan['masks'][s] >> i & 1 else -1 if output[2*k+s] >> i & 1 else 1
            for i in range(plan['d'])) for s in range(2))
        if words is not None and (tuple(support_mask(w) for w in words) != plan['masks']
                or tuple(w.count(-1) for w in words) != tuple(plan['negative'])
                or tuple(x+y for x, y in zip(pacf(words[0]), pacf(words[1]))) != tuple(plan['wanted'])):
            raise RuntimeError('paired label failed independent all-shift integer audit')
        st = stats[7*k:7*k+7]
        results.append(dict(status='SAT_COMPRESSED' if statuses[k] else 'UNSAT_FIXED_SUPPORT',
            words=words, rank=int(plan['meta'][1]), free_variables=int(plan['meta'][2]),
            construction_seconds=plan['construction_seconds'], solve_seconds=elapsed,
            solve_timing_scope='Joint two-label wall time; do not sum across labels',
            assignments=int(st[0]), quadratic_matches=int(st[1]), count_matches=int(st[2]),
            correlation_checks=int(st[3]), prefix_branches=int(st[4]), projected_candidates=int(st[5]),
            maximum_tail_nullity=int(st[6]), hybrid_tail=min(tail, int(plan['meta'][2])),
            hybrid_backend='exact_paired_bit_tail', candidate_counter_scope='original_mod8_models', reason=None))
    return tuple(results)
