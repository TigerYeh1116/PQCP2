"""Exhaust all length-6 input pairs and independent nearest feasible sets."""
from collections import defaultdict
from itertools import product
import random

import pytest

from solver.half_shift_projection import minimum_half_shift_projection
from solver.target_profiles import TargetContentProfile, pair_content, canonical_target_content_profiles


def half_correlation(a, b):
    d = len(a)//2
    return sum((1-2*w[i])*(1-2*w[(i+d) % len(w)]) for w in (a, b) for i in range(len(w)))


def negatives(word):
    return sum(word[i] == word[i+len(word)//2] == 1 for i in range(len(word)//2))


def test_all_length6_pairs_match_literal_minimum_hamming_oracle():
    by_content = defaultdict(list)
    pairs = tuple(product(tuple(product((0, 1), repeat=6)), repeat=2))
    for a, b in pairs:
        if half_correlation(a, b) == 0:
            by_content[pair_content(a, b)].append((a, b))
    for a, b in pairs:
        contents = pair_content(a, b)
        target = TargetContentProfile(6, 1, -1, *contents)
        feasible = by_content[contents]
        result = minimum_half_shift_projection(a, b, target)
        if not feasible:
            assert result is None
            continue
        distance = min(sum(x != y for old, new in zip((a, b), words) for x, y in zip(old, new))
                       for words in feasible)
        assert result.words in feasible and result.distance == distance
        assert result.negative_counts == tuple(negatives(w) for w in result.words)
        assert minimum_half_shift_projection(*result.words, target).words == result.words
        # Also restrict to an independently selected content subtype, checking
        # that the finite minimization neither invents nor omits configurations.
        allowed = tuple(negatives(w) for w in feasible[0])
        restricted = [words for words in feasible if tuple(negatives(w) for w in words) == allowed]
        result = minimum_half_shift_projection(a, b, target, negative_pairs=(allowed,))
        assert result.words in restricted
        assert result.distance == min(sum(x != y for old, new in zip((a, b), words) for x, y in zip(old, new))
                                      for words in restricted)


def test_l86_random_pairs_preserve_contents_and_all_proved_count_bounds():
    from solver.torch_compressed_search import search_paired_contents
    rng = random.Random(17643)
    for target in canonical_target_content_profiles(86):
        allowed = tuple((a[0], b[0]) for a, b in search_paired_contents(target))
        for _ in range(128):
            words = []
            for e, o in ((target.a_even_ones, target.a_odd_ones), (target.b_even_ones, target.b_odd_ones)):
                bits = [0]*86
                for i in rng.sample(range(0, 86, 2), e)+rng.sample(range(1, 86, 2), o):
                    bits[i] = 1
                words.append(tuple(bits))
            result = minimum_half_shift_projection(*words, target, negative_pairs=allowed)
            assert half_correlation(*result.words) == 0
            assert pair_content(*result.words) == pair_content(*words)
            assert result.negative_counts in allowed
            assert result.distance == 2*min(sum(abs(x-y) for x, y in zip(result.original_negative_counts, row)) for row in allowed)
            assert minimum_half_shift_projection(*result.words, target, negative_pairs=allowed).distance == 0


def test_empty_restricted_set_and_invalid_input():
    t = TargetContentProfile(6, 1, -1, 1, 1, 1, 0)
    a, b = (1, 1, 0, 0, 0, 0), (1, 0, 0, 0, 0, 0)
    assert minimum_half_shift_projection(a, b, t, negative_pairs=()) is None
    with pytest.raises(ValueError):
        minimum_half_shift_projection(a, b, t, negative_pairs=((99, -99),))
    with pytest.raises(ValueError):
        minimum_half_shift_projection(b, a, t)


def test_native_energy_and_minimum_distance_against_independent_integer_pacf():
    from solver.half_shift_projection_native import project_energy_ties
    from solver.correlation import full_correlation_profile
    from solver.structured_energy import structured_energy_breakdown
    rng = random.Random(86543)
    for length in (6, 14, 22, 86, 94):
        d = length//2
        for _ in range(48):
            words = tuple(tuple(rng.randrange(2) for _ in range(length)) for _ in range(2))
            content = pair_content(*words)
            target = TargetContentProfile(length, 1+rng.randrange(d-1), rng.choice((-1, 1)), *content)
            counts = tuple(rng.randint(max(0, e+o-d), min(e, o)) for e, o in (content[:2], content[2:]))
            for reverse, compression in product((False, True), repeat=2):
                out, energy = project_energy_ties(words, target, counts, reverse, compression=compression)
                assert pair_content(*out) == content
                assert tuple(negatives(w) for w in out) == counts
                assert energy == structured_energy_breakdown(full_correlation_profile(*out),
                    target.k, target.eta, (2,) if compression else ()).weighted_total({2: 2} if compression else {})
                assert sum(x != y for old, new in zip(words, out) for x, y in zip(old, new)) == 2*sum(
                    abs(negatives(w)-n) for w, n in zip(words, counts))


def test_energy_ties_preserve_all_literal_minimum_distances_and_improve_baseline():
    from solver.half_shift_projection import energy_aware_minimum_half_shift_projection
    from solver.correlation import full_correlation_profile
    from solver.structured_energy import structured_energy_breakdown
    def energy(words, target):
        return structured_energy_breakdown(full_correlation_profile(*words), target.k, target.eta, (2,)).weighted_total({2: 2})
    rows = tuple(product((0, 1), repeat=6))
    for a, b in product(rows, repeat=2):
        target = TargetContentProfile(6, 2, 1, *pair_content(a, b))
        base = minimum_half_shift_projection(a, b, target)
        result = energy_aware_minimum_half_shift_projection(a, b, target)
        if base is None:
            assert result is None
            continue
        assert result.distance == base.distance
        assert sum(x != y for old, new in zip((a, b), result.words) for x, y in zip(old, new)) == base.distance
        assert half_correlation(*result.words) == 0
        assert energy(result.words, target) <= energy(base.words, target)
