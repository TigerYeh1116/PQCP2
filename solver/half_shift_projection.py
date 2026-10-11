"""Minimum Hamming projection with exact parity contents, for odd half-length.

An antipodal pair has one even and one odd coordinate. Let n be its count of
11 pairs. Keeping e/o ones fixed, changing n by delta needs at least 2*|delta|
bit flips and is attained by same-parity swaps between 10/01 or 11/00 pairs.
Enumerate allowed n_A+n_B to minimize the sum of these two exact distances.
This is an observation candidate constructor, not a continuous-loss projector.
"""
from dataclasses import dataclass

from .correlation import normalize_binary_sequence
from .target_profiles import pair_content


@dataclass(frozen=True)
class HalfShiftProjection:
    words: tuple
    distance: int
    negative_counts: tuple
    original_negative_counts: tuple


def _pair_types(word):
    d = len(word)//2
    groups = {key: [] for key in ((0, 0), (1, 0), (0, 1), (1, 1))}
    for i in range(d):
        even, odd = (i, i+d) if i % 2 == 0 else (i+d, i)
        groups[word[even], word[odd]].append((even, odd))
    return groups


def _change_negative_count(word, groups, desired):
    delta = desired-len(groups[1, 1])
    if delta == 0:
        return word
    # Every move flips one even donor 1 -> 0 and one even receiver 0 -> 1.
    donors, receivers = (groups[1, 0], groups[0, 1]) if delta > 0 else (groups[1, 1], groups[0, 0])
    if abs(delta) > min(len(donors), len(receivers)):
        raise RuntimeError('a content-feasible pair-type move could not be constructed')
    bits = list(word)
    for donor, receiver in zip(donors[:abs(delta)], receivers[:abs(delta)]):
        bits[donor[0]], bits[receiver[0]] = 0, 1
    return tuple(bits)


def minimum_half_shift_projection(a, b, target, *, negative_pairs=None):
    """Return a closest binary pair at half-shift zero, or None if infeasible.

    ``negative_pairs`` optionally restricts the finite set using separately
    proved content conditions. No empirical score is used to delete cases.
    Ties choose the smallest n_A; the minimal metric here is Hamming distance.
    It is NOT the nearest Euclidean projection of the original continuous
    parameters, so it must never replace BatchedPQCP.project in its loss.
    """
    a, b = normalize_binary_sequence(a), normalize_binary_sequence(b)
    d = target.L//2
    if (target.L < 6 or target.L % 2 or not d % 2 or len(a) != target.L or len(b) != target.L
            or pair_content(a, b) != (target.a_even_ones, target.a_odd_ones,
                                      target.b_even_ones, target.b_odd_ones)):
        raise ValueError('exact target parity contents and odd half-length required')
    groups = _pair_types(a), _pair_types(b)
    current = tuple(len(g[1, 1]) for g in groups)
    sums = ((target.a_even_ones, target.a_odd_ones), (target.b_even_ones, target.b_odd_ones))
    numerator = sum(e+o for e, o in sums)-d
    if numerator % 2:
        return None
    total = numerator//2
    ranges = tuple((max(0, e+o-d), min(e, o)) for e, o in sums)
    candidates = tuple((n, total-n) for n in range(ranges[0][0], ranges[0][1]+1)
                       if ranges[1][0] <= total-n <= ranges[1][1])
    if negative_pairs is not None:
        allowed = tuple(negative_pairs)
        if any(len(row) != 2 or any(type(n) is not int for n in row) or row not in candidates for row in allowed):
            raise ValueError('negative_pairs must be exact-content half-shift feasible pairs')
        candidates = tuple(row for row in candidates if row in allowed)
    if not candidates:
        return None
    distance, chosen = min((2*sum(abs(x-y) for x, y in zip(current, row)), row) for row in candidates)
    words = tuple(_change_negative_count(w, g, n) for w, g, n in zip((a, b), groups, chosen))
    if (sum(x != y for old, new in zip((a, b), words) for x, y in zip(old, new)) != distance
            or sum(sum(w[i] != w[i+d] for i in range(d)) for w in words) != d
            or pair_content(*words) != pair_content(a, b)):
        raise RuntimeError('minimum-distance half-shift construction failed integer audit')
    return HalfShiftProjection(words, distance, chosen, current)


def energy_aware_minimum_half_shift_projection(a, b, target, *, negative_pairs=None, compression=True):
    """Keep exact minimal Hamming distance, improve ties by exact integer energy.

    Greedy swap energies do not certify globally optimal energy among all ties.
    The original deterministic closest pair is included, so final energy cannot
    exceed that baseline. All minimal-count pairs and both row orders are tried.
    """
    from .half_shift_projection_native import project_energy_ties
    from .correlation import full_correlation_profile
    from .structured_energy import structured_energy_breakdown
    if type(compression) is not bool:
        raise ValueError('compression must be boolean')
    allowed = None if negative_pairs is None else tuple(negative_pairs)
    base = minimum_half_shift_projection(a, b, target, negative_pairs=allowed)
    if base is None or base.distance == 0:
        return base
    a, b = normalize_binary_sequence(a), normalize_binary_sequence(b)
    d = target.L//2
    if allowed is None:
        total = sum(base.negative_counts)
        ea, oa, eb, ob = pair_content(a, b)
        allowed = tuple((n, total-n) for n in range(max(0, ea+oa-d), min(ea, oa)+1)
                        if max(0, eb+ob-d) <= total-n <= min(eb, ob))
    def energy(words):
        return structured_energy_breakdown(full_correlation_profile(*words), target.k, target.eta,
            (2,) if compression else ()).weighted_total({2: 2} if compression else {})
    best, best_energy = base, energy(base.words)
    for counts in allowed:
        if 2*sum(abs(x-y) for x, y in zip(base.original_negative_counts, counts)) != base.distance:
            continue
        for reverse in (False, True):
            words, exact_energy = project_energy_ties((a, b), target, counts, reverse, compression=compression)
            if exact_energy < best_energy:
                best = HalfShiftProjection(words, base.distance, counts, base.original_negative_counts)
                best_energy = exact_energy
    if (energy(best.words) != best_energy or pair_content(*best.words) != pair_content(a, b)
            or sum(sum(w[i] != w[i+d] for i in range(d)) for w in best.words) != d
            or sum(x != y for old, new in zip((a, b), best.words) for x, y in zip(old, new)) != base.distance):
        raise RuntimeError('native energy-aware minimum-Hamming result failed independent audit')
    return best
