"""Exact two-row meet-in-the-middle repair with parity-preserving swaps.

Each row changes at most one pair of opposite signs at equal-parity indices.
The two row PACFs add independently, so their integer deltas can be joined
against a target residual without evaluating the Cartesian product. A miss
excludes ONLY this explicitly bounded neighborhood, never the full problem.
"""
from collections import defaultdict
from itertools import combinations
from time import perf_counter

from .correlation import full_correlation_profile, normalize_binary_sequence
from .structured_energy import target_profile
from .target_profiles import pair_content, target_content_profiles
from .verifier import verify_pqcp


def row_swap_deltas(bits):
    row = tuple(1-2*v for v in normalize_binary_sequence(bits))
    length = len(row)
    if length < 4 or length % 2:
        raise ValueError('even binary length at least four required')
    groups = defaultdict(list)
    groups[(0,)*(length//2)].append(())
    for p, q in combinations(range(length), 2):
        if p % 2 != q % 2 or row[p] == row[q]:
            continue
        delta = []
        for u in range(1, length//2+1):
            value = -2*row[p]*(row[(p+u) % length]+row[(p-u) % length])
            value -= 2*row[q]*(row[(q+u) % length]+row[(q-u) % length])
            value += 4*row[p]*row[q]*(int((p+u) % length == q)+int((q+u) % length == p))
            if value % 4:
                raise RuntimeError('binary swap delta failed integer lattice')
            delta.append(value//4)
        groups[tuple(delta)].append((p, q))
    return dict(groups)


def exact_two_row_swap_repair(a, b, *, targets=None):
    started = perf_counter()
    a, b = tuple(normalize_binary_sequence(a)), tuple(normalize_binary_sequence(b))
    if len(a) != len(b):
        raise ValueError('equal row lengths required')
    length, content = len(a), pair_content(a, b)
    feasible = tuple(p for p in target_content_profiles(length, decimation_reduced=False)
        if (p.a_even_ones, p.a_odd_ones, p.b_even_ones, p.b_odd_ones) == content)
    chosen = feasible if targets is None else tuple(targets)
    if any(p not in feasible for p in chosen):
        raise ValueError('target must preserve the center parity contents')
    profile = tuple(full_correlation_profile(a, b))
    left, right = row_swap_deltas(a), row_swap_deltas(b)
    witnesses, seen, lookups = [], set(), 0
    for target in chosen:
        wanted = target_profile(length, target.k, target.eta)
        difference = tuple(x-y for x, y in zip(wanted[1:length//2+1], profile[1:length//2+1]))
        if any(value % 4 for value in difference):
            continue
        residual = tuple(value//4 for value in difference)
        for first, swaps_a in left.items():
            second = tuple(x-y for x, y in zip(residual, first))
            lookups += 1
            for swap_a in swaps_a:
                for swap_b in right.get(second, ()):
                    words = tuple(tuple(v ^ int(i in swap) for i, v in enumerate(row))
                                  for row, swap in zip((a, b), (swap_a, swap_b)))
                    if words in seen:
                        continue
                    check = verify_pqcp(*words)
                    if not check.is_valid or tuple(check.profile) != wanted or pair_content(*words) != content:
                        raise RuntimeError('meet-in-the-middle hit failed independent integer verifier')
                    seen.add(words)
                    witnesses.append(dict(words=words, swaps=(swap_a, swap_b), k=target.k, sign=target.eta))
    return dict(status='SAT' if witnesses else 'EXHAUSTIVE_NEIGHBORHOOD_MISS', witnesses=witnesses,
        row_configurations=tuple(sum(map(len, index.values())) for index in (left, right)),
        profile_groups=(len(left), len(right)), targets=len(chosen), exact_lookups=lookups,
        complete=True, elapsed=perf_counter()-started,
        scope='At most one opposite-sign equal-parity swap per row, with the selected fixed-content targets only.')
