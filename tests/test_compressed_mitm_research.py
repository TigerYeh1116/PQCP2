"""Bounded mathematical checks; pytest never runs the L44 pilot."""

from collections import defaultdict
from itertools import product

import pytest

from experiments.research_compressed_mitm import (
    content_necklaces, content_space, prime_necklace_count,
    signature_join, small_exhaustive, word_count, middle_mitm_probe,
)
from solver.target_profiles import canonical_target_content_profiles, TargetContentProfile
from solver.structured_energy import fold_profile, target_profile, integer_periodic_autocorrelation_profile


@pytest.mark.parametrize("n", (1, 2, 3, 4, 5))
def test_content_generator_against_all_five_letter_words(n):
    expected = defaultdict(set)
    labeled = defaultdict(int)
    for word in product((-2,-1,0,1,2), repeat=n):
        hist = tuple(word.count(x) for x in (-2,-1,0,1,2))
        expected[hist].add(min(word[i:]+word[:i] for i in range(n)))
        labeled[hist] += 1
    for hist, representatives in expected.items():
        observed = tuple(content_necklaces(hist))
        assert len(observed) == len(set(observed))
        assert set(observed) == representatives
        assert word_count(hist) == labeled[hist]
        if n in (2, 3, 5):
            assert prime_necklace_count(hist) == len(representatives)


def test_signature_join_retains_distinct_words_with_equal_pacf():
    words = ((1,0,0), (0,1,0))
    assert set(signature_join(words, words, (2,0,0))) == set(product(words, repeat=2))


@pytest.mark.parametrize("length,coarse,middle,binary", (
    (4,5,5,5), (8,16,16,16), (12,936,936,816),
))
def test_exhaustive_stages_have_no_false_negative(length, coarse, middle, binary):
    result = small_exhaustive(length)
    assert result["all_assertions_passed"]
    assert result["coarse_matches"] == coarse
    assert result["coarse_with_middle_lift"] == middle
    assert result["coarse_with_binary_lift"] == binary
    if length == 12:
        assert result["coarse_with_cross_match"] == 864


def test_L44_count_regression_and_frequency_impossibility():
    profiles = canonical_target_content_profiles(44)
    excluded = content_space(profiles[0])
    assert excluded["factor_n_row_pairs"] == 0
    assert all(s["cyclic_representatives"] == 0 for s in excluded["sides"])
    included = content_space(profiles[1])
    assert [s["cyclic_representatives"] for s in included["sides"]] == [119132,116193]


def test_research_guards_avoid_accidental_large_brute_force():
    with pytest.raises(ValueError):
        small_exhaustive(68)
    with pytest.raises(ValueError):
        prime_necklace_count((0,0,4,0,0))


def test_middle_probe_against_complete_ternary_space_without_z3_check():
    profile = TargetContentProfile(12,3,-1,1,3,2,4)
    ca, cb = (4,0,0), (0,0,0)
    pools = []
    for coarse, even, odd in ((ca,1,3),(cb,2,4)):
        pools.append([w for w in product((-2,0,2),repeat=6)
                      if tuple(w[i]+w[i+3] for i in range(3)) == coarse
                      and sum(w[::2]) == 6-2*even
                      and sum(w[1::2]) == 6-2*odd])
    desired = fold_profile(target_profile(12,3,-1),2)
    expected = sum(tuple(x+y for x,y in zip(integer_periodic_autocorrelation_profile(a),
                                          integer_periodic_autocorrelation_profile(b))) == desired
                   for a in pools[0] for b in pools[1])
    result = middle_mitm_probe(ca,cb,profile,binary_jobs=0)
    assert result["content_filtered_counts"] == list(map(len,pools))
    assert result["exact_middle_matches"] == expected
    assert result["binary_outcomes"] == []
    assert result["unattempted_middle_pairs"] == result["cross_matches"]


def test_middle_probe_budget_is_deferred_not_unsat():
    profile = TargetContentProfile(12,3,-1,1,3,2,4)
    result = middle_mitm_probe((4,0,0),(0,0,0),profile,max_raw_per_side=1)
    assert result["status"] == "DEFERRED_BY_ENUMERATION_BUDGET"
