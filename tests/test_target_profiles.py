"""Mathematical identity, symmetry, and known-solution profile tests."""

import pytest

from solver.correlation import full_correlation_profile
from solver.target_profiles import (
    canonical_target_content_profiles,
    decimation_shift_representatives,
    pair_content,
    target_content_profiles,
)
from solver.weight_constraints import admissible_weight_pairs


KNOWN_L44_A = tuple(map(int, "10001001110010011010011110101000011000010000"))
KNOWN_L44_B = tuple(map(int, "11000000111101111000001011001010101101000100"))


def test_l44_decimation_representatives_match_unit_orbits():
    assert decimation_shift_representatives(44) == (1, 2, 4, 11)


@pytest.mark.parametrize("length", (44, 46, 68, 86, 94))
def test_profile_weight_projection_equals_existing_sum_identity(length):
    projected = {profile.weight_pair for profile in target_content_profiles(length)}
    assert projected == set(admissible_weight_pairs(length))


@pytest.mark.parametrize("length", (58, 90))
def test_impossible_lengths_have_no_target_content_profiles(length):
    assert target_content_profiles(length) == ()


def test_known_l44_solution_satisfies_rederived_ordinary_and_alternating_profile():
    correlation = full_correlation_profile(KNOWN_L44_A, KNOWN_L44_B)
    k = next(shift for shift in range(1, 22) if correlation[shift])
    eta = correlation[k] // 4
    content = pair_content(KNOWN_L44_A, KNOWN_L44_B)
    assert any(
        profile.k == k and profile.eta == eta
        and content == (
            profile.a_even_ones, profile.a_odd_ones,
            profile.b_even_ones, profile.b_odd_ones,
        )
        for profile in target_content_profiles(44)
    )


def test_canonical_profiles_reduce_but_do_not_invent_cases():
    complete = target_content_profiles(44)
    canonical = canonical_target_content_profiles(44)
    assert 0 < len(canonical) < len(complete)
    assert {(p.k, p.eta) for p in canonical} <= {(p.k, p.eta) for p in complete}
