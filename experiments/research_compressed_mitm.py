"""Read-only feasibility study, NOT a production search backend.

Count factor-4 fixed-content spaces exactly without generating their words.
On L <= 12 only, exhaustively compare compressed hash joins, middle joins,
and independently recomputed binary correlations. No known result files,
random seeds, or GPU are used. The optional --pilot-44 additionally tries
four bounded existing Z3 lifts. Ordered/labeled counts are not counts of
inequivalent solutions. Nothing is appended to L.txt.
"""

import argparse
from collections import defaultdict
from dataclasses import asdict
from functools import lru_cache
from itertools import product
import json
from math import factorial, prod
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from solver.correlation import periodic_autocorrelation
from solver.cross_compression import (
    binary_margins_feasible, compatible_cross_compressions, middle_matches_cross,
)
from solver.frequency_constraints import quarter_frequency_feasible
from solver.structured_energy import (
    compress_binary_sequence, fold_profile,
    integer_periodic_autocorrelation_profile as pacf, target_profile,
)
from solver.target_profiles import canonical_target_content_profiles
from solver.verifier import verify_pqcp


def contents(n, symbols=5):
    """All weak compositions; no sequence permutation is constructed."""
    if symbols == 1:
        yield (n,)
    else:
        for count in range(n + 1):
            for tail in contents(n - count, symbols - 1):
                yield (count,) + tail


def word_count(content):
    """Exact multinomial count of labeled words with this content."""
    result = factorial(sum(content))
    for count in content:
        result //= factorial(count)
    return result


def prime_necklace_count(content):
    """Burnside for prime length: nonconstant words have full period."""
    n = sum(content)
    if n < 2 or any(n % q == 0 for q in range(2, int(n**0.5) + 1)):
        raise ValueError("prime compressed length required")
    return 1 if max(content) == n else word_count(content) // n


def content_necklaces(content):
    """Research-only fixed-content FKM over q=-2,-1,0,1,2.

    This is deliberately separate from the production ternary FKM API.
    Small brute-force tests audit both completeness and cyclic uniqueness.
    """
    n = sum(content)
    word, used = [0] * (n + 1), [0] * 5

    def visit(pos, period):
        if pos > n:
            if n % period == 0 and tuple(used) == content:
                yield tuple(x - 2 for x in word[1:])
            return
        first = word[pos - period]
        for symbol in range(first, 5):
            if used[symbol] == content[symbol]:
                continue
            word[pos] = symbol
            used[symbol] += 1
            yield from visit(pos + 1, period if symbol == first else pos)
            used[symbol] -= 1

    yield from visit(1, 1)


def coarse_row_pairs(profile):
    """All four-entry factor-n pairs with exact parity sums and folded PACF.

    This is an independent enumeration, not the production bounded pool.
    Norm <= target[0] is safe since the partner's squared norm is nonnegative.
    """
    n = profile.L // 4
    desired = fold_profile(target_profile(profile.L, profile.k, profile.eta), n)
    sides = []
    for even, odd in ((profile.a_even_ones, profile.a_odd_ones),
                      (profile.b_even_ones, profile.b_odd_ones)):
        words = []
        for d0, d1 in product(range(n + 1), repeat=2):
            degrees = (d0, d1, even - d0, odd - d1)
            if not all(0 <= d <= n for d in degrees):
                continue
            row = tuple(n - 2*d for d in degrees)
            if sum(x*x for x in row) <= desired[0]:
                words.append(row)
        sides.append(words)
    return list(signature_join(sides[0], sides[1], desired))


def signature_join(left, right, desired):
    """Exact integer hash join; retain ALL words with the same signature."""
    buckets = defaultdict(list)
    for word in right:
        buckets[pacf(word)].append(word)
    for word in left:
        needed = tuple(t - x for t, x in zip(desired, pacf(word)))
        for other in buckets.get(needed, ()):
            yield word, other


@lru_cache(maxsize=None)
def margin_compatible(content, row):
    n = sum(content)
    columns = tuple(degree for count, degree in zip(content, (4, 3, 2, 1, 0))
                    for _ in range(count))
    return binary_margins_feasible(tuple((n-x)//2 for x in row), columns)


def content_space(profile):
    """Exact counts AFTER sum/norm/Gale--Ryser tests, BEFORE long PACF/PSD.

    Work with q = c4/2 in {-2,-1,0,1,2}. The pair norm is fold_4(T)[0]/4.
    A histogram survives only if a partner histogram has complementary norm
    and both admit a common exact factor-n target pair. This is a necessary
    condition, never a claim that its permutations admit a PQCP partner.
    Independent rotations by multiples of 4 cycle all n columns for odd n,
    preserve mod-4 row counts, and justify the necklace count at prime n.
    """
    n = profile.L // 4
    if profile.L % 4 or n % 2 == 0 or n < 3:
        raise ValueError("requires L=4n, odd n >= 3")
    desired = fold_profile(target_profile(profile.L, profile.k, profile.eta), 4)
    assert desired[0] % 4 == 0
    budget = desired[0] // 4
    rows = coarse_row_pairs(profile)
    assert bool(rows) == quarter_frequency_feasible(profile)
    histograms = tuple(contents(n))
    sides = []
    for side, weight in enumerate(profile.weight_pair):
        total = (profile.L - 2*weight)//2
        choices = []
        for hist in histograms:
            if sum(c*x for c, x in zip(hist, (-2,-1,0,1,2))) != total:
                continue
            norm = sum(c*x*x for c, x in zip(hist, (-2,-1,0,1,2)))
            mask = sum(1 << i for i, pair in enumerate(rows)
                       if margin_compatible(hist, pair[side]))
            choices.append((hist, norm, mask))
        sides.append(choices)
    live = [set(), set()]
    pair_count = 0
    for ha, qa, ma in sides[0]:
        for hb, qb, mb in sides[1]:
            if qa + qb == budget and ma & mb:
                live[0].add(ha)
                live[1].add(hb)
                pair_count += 1
    return dict(
        target=asdict(profile), q_pair_norm=budget,
        factor_n_row_pairs=len(rows), compatible_content_pairs=pair_count,
        sides=[dict(
            sum_only_words=sum(word_count(h) for h, _, _ in choices),
            surviving_contents=len(kept),
            labeled_words=sum(word_count(h) for h in kept),
            cyclic_representatives=sum(prime_necklace_count(h) for h in kept),
        ) for choices, kept in zip(sides, live)],
    )


def pilot_44(lift_jobs=4, timeout_ms=250):
    """Exhaustive coarse join for ONE canonical L44 profile, bounded lifts.

    No PSD rejection is used: the measurement isolates content + exact MITM.
    The full signature itself enforces all compressed target frequencies.
    Lift timeouts are UNKNOWN, never evidence of nonexistence. First-match
    selection is deterministic, not representative of all matched jobs.
    No global decimation/reflection/complement job canonicalization here.
    """
    from solver.multilevel_lift import solve_multilevel_lift

    start = perf_counter()
    profile = next(p for p in canonical_target_content_profiles(44)
                   if p.k == 2 and quarter_frequency_feasible(p))
    n = 11
    desired = tuple(x // 4 for x in fold_profile(target_profile(44, 2, -1), 4))
    rows = coarse_row_pairs(profile)
    # Norm/content masks are cheap; enumerate histograms, not full words.
    sides = []
    for side, weight in enumerate(profile.weight_pair):
        choices = []
        for hist in contents(n):
            if sum(c*x for c, x in zip(hist, (-2,-1,0,1,2))) != (44-2*weight)//2:
                continue
            mask = sum(1 << i for i, pair in enumerate(rows)
                       if margin_compatible(hist, pair[side]))
            choices.append((hist, sum(c*x*x for c,x in zip(hist,(-2,-1,0,1,2))), mask))
        sides.append(choices)
    surviving = [set(), set()]
    for ha, qa, ma in sides[0]:
        for hb, qb, mb in sides[1]:
            if qa + qb == desired[0] and ma & mb:
                surviving[0].add(ha)
                surviving[1].add(hb)
    masks = [{h:m for h, _, m in side} for side in sides]
    buckets = defaultdict(list)
    generated = [0, 0]
    for hist in sorted(surviving[1]):
        for b in content_necklaces(hist):
            generated[1] += 1
            buckets[pacf(b)[:n//2+1]].append((b, masks[1][hist]))
    built = perf_counter()
    matched = cross_matched = 0
    selected = []
    for hist in sorted(surviving[0]):
        for a in content_necklaces(hist):
            generated[0] += 1
            needed = tuple(t-x for t,x in zip(desired[:n//2+1], pacf(a)))
            for b, mask_b in buckets.get(needed, ()):
                matched += 1
                if masks[0][hist] & mask_b:
                    cross_matched += 1
                    if len(selected) < lift_jobs:
                        selected.append((tuple(2*x for x in a), tuple(2*x for x in b)))
    joined = perf_counter()
    estimate = content_space(profile)
    assert generated == [s["cyclic_representatives"] for s in estimate["sides"]]
    lifts = []
    for ca, cb in selected:
        cross = compatible_cross_compressions(ca, cb, profile.k, profile.eta, profile)
        assert cross
        result = solve_multilevel_lift(ca, cb, profile.k, profile.eta,
                                      timeout_ms=timeout_ms, content_profile=profile,
                                      max_middle_models=4)
        if result.status == "SAT":
            audit = verify_pqcp(result.a, result.b)
            assert audit.is_valid and audit.profile == target_profile(44, profile.k, profile.eta)
        lifts.append(dict(ca=ca, cb=cb, status=result.status, verified=result.verified,
                          middle_models=result.middle_models,
                          binary_checks=result.binary_checks, reason=result.reason,
                          A=result.a, B=result.b))
    result = dict(target=asdict(profile), generated=generated, buckets=len(buckets),
                largest_bucket=max(map(len, buckets.values()), default=0),
                integer_signature_dimensions=n//2+1,
                exact_signature_matches=matched, matches_after_cross=cross_matched,
                build_seconds=built-start, generate_and_join_seconds=joined-built,
                lift_seconds=perf_counter()-joined, lift_timeout_ms=timeout_ms,
                lift_results=lifts)
    if selected:
        result["middle_mitm_probe"] = middle_mitm_probe(*selected[0], profile)
    return result


def middle_mitm_probe(ca, cb, profile, max_raw_per_side=100000, binary_jobs=4):
    """Bounded research alternative to asking Z3 to find middle patterns.

    Enumerate all per-column splits if BOTH raw counts fit the guard, enforce
    parity sums, then integer hash-join the middle PACFs. The join is exact
    within this coarse pair. Only the first four binary children are tried;
    unattempted or UNKNOWN children mean the parent remains unresolved.
    """
    from solver.compressed_lift import solve_compressed_lift

    start = perf_counter()
    options = [tuple(tuple(x for x in (-2,0,2) if v-x in (-2,0,2))
                     for v in word) for word in (ca,cb)]
    counts = [prod(map(len, side)) for side in options]
    if max(counts) > max_raw_per_side:
        return dict(raw_counts=counts, status="DEFERRED_BY_ENUMERATION_BUDGET")
    desired = target_profile(profile.L, profile.k, profile.eta)
    middle_target = fold_profile(desired, 2)
    sides = []
    for word, side, even, odd in zip(
            (ca,cb), options, (profile.a_even_ones,profile.b_even_ones),
            (profile.a_odd_ones,profile.b_odd_ones)):
        kept = []
        for first in product(*side):
            middle = first + tuple(v-x for v,x in zip(word,first))
            if (sum(middle[::2]) == profile.L//2-2*even
                    and sum(middle[1::2]) == profile.L//2-2*odd):
                kept.append(middle)
        sides.append(kept)
    cross = compatible_cross_compressions(ca, cb, profile.k, profile.eta, profile)
    match_count = cross_count = 0
    selected = []
    for a, b in signature_join(*sides, middle_target):
        match_count += 1
        if any(middle_matches_cross(a, ra) and middle_matches_cross(b, rb)
               for ra, rb in cross):
            cross_count += 1
            if len(selected) < binary_jobs:
                selected.append((a,b))
    joined = perf_counter()
    outcomes = []
    for a, b in selected:
        result = solve_compressed_lift(a,b,profile.k,profile.eta,
                                      timeout_ms=250,content_profile=profile,
                                      cross_compressions=cross)
        if result.status == "SAT":
            audit = verify_pqcp(result.a, result.b)
            assert audit.is_valid and audit.profile == desired
        outcomes.append(dict(status=result.status,verified=result.verified,
                             A=result.a,B=result.b))
    return dict(raw_counts=counts, content_filtered_counts=list(map(len,sides)),
                exact_middle_matches=match_count, cross_matches=cross_count,
                enumerate_and_join_seconds=joined-start,
                binary_lift_seconds=perf_counter()-joined,
                binary_outcomes=outcomes, unattempted_middle_pairs=cross_count-len(selected),
                scope="complete middle join; binary lifts bounded, no parent UNSAT claim")


def small_exhaustive(length):
    """Classify ALL labeled coarse pairs for L=4,8,12, with no solution input.

    The full-binary reference uses correlation.py, not compressed PACF.
    Every binary word is enumerated; grouping only avoids repeating identical
    PACF pair comparisons. One independent verifier call audits each distinct
    feasible (coarse A, coarse B, target) witness. Counts include all ordered
    pairs and target cases, with no symmetry/content representative reduction.
    """
    if length not in (4, 8, 12):
        raise ValueError("bounded research supports only L=4,8,12")
    start = perf_counter()
    n = length // 4
    coarse_words = tuple(product((-4,-2,0,2,4), repeat=n))
    middle_words = tuple(product((-2,0,2), repeat=2*n))
    binary_index = defaultdict(dict)
    for word in product((0, 1), repeat=length):
        signature = tuple(periodic_autocorrelation(word, u) for u in range(length))
        coarse = compress_binary_sequence(word, 4)
        # Independently audit composition and aliasing for EVERY binary word.
        middle = compress_binary_sequence(word, 2)
        assert tuple(middle[i] + middle[i+n] for i in range(n)) == coarse
        assert pacf(coarse) == fold_profile(signature, 4)
        assert pacf(middle) == fold_profile(signature, 2)
        count, witness = binary_index[signature].get(coarse, (0, word))
        binary_index[signature][coarse] = (count + 1, witness)
    totals = defaultdict(int)
    for k, eta in product(range(1, length//2), (-1, 1)):
        desired = target_profile(length, k, eta)
        coarse = set(signature_join(coarse_words, coarse_words, fold_profile(desired, 4)))
        middle = set()
        for a, b in signature_join(middle_words, middle_words, fold_profile(desired, 2)):
            middle.add((tuple(a[i]+a[i+n] for i in range(n)),
                        tuple(b[i]+b[i+n] for i in range(n))))
        feasible = set()
        for sig, aa in binary_index.items():
            bb = binary_index.get(tuple(t-x for t, x in zip(desired, sig)), {})
            for ca, (count_a, a) in aa.items():
                for cb, (count_b, b) in bb.items():
                    if (ca, cb) not in feasible:
                        check = verify_pqcp(a, b)
                        assert check.is_valid and check.profile == desired
                    feasible.add((ca, cb))
                    totals["ordered_binary_solutions"] += count_a * count_b
        assert feasible <= middle <= coarse
        totals["coarse_matches"] += len(coarse)
        totals["coarse_with_middle_lift"] += len(middle)
        totals["coarse_with_binary_lift"] += len(feasible)
        if n >= 3 and n % 2:
            cross = {pair for pair in coarse
                     if compatible_cross_compressions(*pair, k, eta)}
            assert feasible <= cross
            totals["coarse_with_cross_match"] += len(cross)
            totals["coarse_with_middle_and_cross"] += len(middle & cross)
    return dict(L=length, elapsed_seconds=perf_counter()-start,
                exhaustive_binary_words=2**length,
                all_assertions_passed=True, **totals)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--small-lengths", nargs="*", type=int, default=[4, 8, 12],
                        choices=[4, 8, 12])
    parser.add_argument("--count-lengths", nargs="*", type=int, default=[44, 68],
                        choices=[44, 68])
    parser.add_argument("--pilot-44", action="store_true",
                        help="Enumerate ~235k short necklaces; four bounded Z3 lifts")
    args = parser.parse_args()
    start = perf_counter()
    report = dict(
        scope="exact research counts; no production search, no L.txt writes",
        small=[small_exhaustive(n) for n in args.small_lengths],
        content_spaces=[content_space(p) for L in args.count_lengths
                        for p in canonical_target_content_profiles(L)],
    )
    report["elapsed_seconds"] = perf_counter() - start
    if args.pilot_44:
        report["pilot_44"] = pilot_44()
        report["elapsed_seconds"] = perf_counter() - start
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
