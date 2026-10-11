"""Exact binary lift via alternating modulation of odd-half differences.

For odd d, h_i=(-1)^i g_i transforms N_h(u)=(-1)^u P_g(u).
g lives exactly on compression zeros, with negative count e-n_c per side.
Thus the existing periodic sign kernel is an exhaustive binary lift solver.
"""
from time import perf_counter

from .compressed_repair import _validated_reference
from .structured_energy import integer_periodic_autocorrelation_profile as pacf, target_profile
from .target_profiles import pair_content
from .ternary_signs_native import solve_sign_system
from .ternary_support_search import support_sign_system
from .verifier import verify_pqcp


def difference_sign_problem(coarse_masks, target):
    """Lift sign constraints depend only on literal supports and parity content."""
    d = target.L//2
    if (target.L != 2*d or not 3 <= d <= 47 or d % 2 == 0 or len(coarse_masks) != 2
            or any(type(m) is not int or not 0 <= m < 1 << d for m in coarse_masks)
            or sum(bin(m).count('1') for m in coarse_masks) != d):
        raise ValueError('odd-half supports of total weight d required')
    counts = ((target.a_even_ones, target.a_odd_ones), (target.b_even_ones, target.b_odd_ones))
    negatives = []
    for mask, (e, o) in zip(coarse_masks, counts):
        weight = bin(mask).count('1')
        difference = weight-(d-e-o)
        if difference % 2:
            raise ValueError('support parity violates prescribed row sum')
        n = difference//2
        z = d-weight
        if not 0 <= n <= weight or not 0 <= e-n <= z or not 0 <= o-n <= z:
            raise ValueError('support contents cannot realize parity counts')
        negatives.append(e-n)
    full_target = target_profile(2*d, target.k, target.eta)
    wanted = tuple((1 if u % 2 == 0 else -1)*(full_target[u]-full_target[u+d])//4
                   for u in range(d))
    masks = tuple(((1 << d)-1) ^ m for m in coarse_masks)
    return masks, wanted, tuple(negatives)


def reconstruct_odd_pair(coarse_words, difference_words, target):
    """Combine two independently solved sign pairs and perform the full audit."""
    d = target.L//2
    signs = []
    for c, g in zip(coarse_words, difference_words):
        h = tuple((1 if i % 2 == 0 else -1)*g[i] for i in range(d))
        signs.append(tuple(c[i]+h[i] for i in range(d))+tuple(c[i]-h[i] for i in range(d)))
    if any(v not in (-1, 1) for w in signs for v in w):
        raise RuntimeError('native orientation reconstruction is not binary')
    words = tuple(tuple(int(v == -1) for v in w) for w in signs)
    check = verify_pqcp(*words)
    if (not check.is_valid or check.profile != target_profile(2*d, target.k, target.eta)
            or pair_content(*words) != (target.a_even_ones, target.a_odd_ones,
                                       target.b_even_ones, target.b_odd_ones)):
        raise RuntimeError('native lift failed full independent correlation/content audit')
    return words


def solve_odd_lift_native(compressed_a, compressed_b, target, *, max_assignments=2**23,
                          conditioning=True, native_preprocessing=True, binary_content_pruning=True):
    started = perf_counter()
    raw = (tuple(compressed_a), tuple(compressed_b))
    if any(type(v) is not int or v not in (-2, 0, 2) for w in raw for v in w):
        raise ValueError('integer factor-two compressed words required')
    a, b, d, coarse_target = _validated_reference(*(tuple(v//2 for v in w) for w in raw), target)
    if (d > 47 or type(max_assignments) is not int or not 1 <= max_assignments <= 2**31
            or type(conditioning) is not bool or type(native_preprocessing) is not bool
            or type(binary_content_pruning) is not bool):
        raise ValueError('odd half-length <=47 and assignment budget in 1..2**31 required')
    if tuple(x+y for x, y in zip(pacf(a), pacf(b))) != coarse_target:
        raise ValueError('exact folded correlation target required before binary lift')
    coarse_masks = tuple(sum(1 << i for i, v in enumerate(w) if v) for w in (a, b))
    masks, wanted, negative = difference_sign_problem(coarse_masks, target)
    first_negative = a.count(-1)
    if binary_content_pruning and d == 43 and (first_negative <= 1 or first_negative >= 15):
        from .l86_binary_content import binary_content_bounds
        bounds = binary_content_bounds(target)
        if bounds is not None and any(not lo <= w.count(-1) <= hi for w, (lo, hi) in zip((a, b), bounds)):
            return dict(status='LIFT_UNSAT', words=None, verified=False, rank=0, free_variables=d,
                        wanted=wanted, masks=masks, negative_counts=negative, assignments=0,
                        content_pruned=True, reason='recorded complete L86 binary-content proof',
                        elapsed=perf_counter()-started)
    if native_preprocessing and d == 43:
        from .ternary_system_native import prepare_native_system, run_native_system
        plan = prepare_native_system(masks, d, wanted, negative, conditioning=conditioning)
        rank, free = plan['meta'][1], plan['meta'][2]
        consistent = plan['consistent']
    else:
        system = support_sign_system(masks, d, wanted, negative)
        rank, free, consistent = system.rank, len(system.free_columns), system.consistent
    common = dict(words=None, verified=False, rank=rank,
                  free_variables=free, wanted=wanted, masks=masks,
                  negative_counts=negative, assignments=0)
    if not consistent:
        return dict(common, status='LIFT_UNSAT', reason='GF2 / integer obstruction', elapsed=perf_counter()-started)
    out = (run_native_system(plan, max_assignments) if native_preprocessing and d == 43 else
           solve_sign_system(system, masks, d, wanted, negative, max_assignments, conditioning=conditioning))
    common['sign_result'] = out
    common['assignments'] = out['assignments']
    if out['words'] is None:
        return dict(common, status='LIFT_UNSAT' if out['status'] == 'UNSAT_FIXED_SUPPORT' else 'LIFT_UNKNOWN',
                    elapsed=perf_counter()-started)
    words = reconstruct_odd_pair((a, b), out['words'], target)
    return dict(common, status='LIFT_SAT', words=words, verified=True, elapsed=perf_counter()-started)
