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


@pytest.mark.parametrize("length", (4, 6, 8, 44, 46, 68, 86, 94))
def test_canonical_profiles_cover_every_complement_and_pair_swap_orbit(length):
    """Every decimation-reduced case maps to one retained symmetry orbit."""
    half = length // 2
    retained = canonical_target_content_profiles(length)
    retained_keys = {
        (p.k, p.eta, p.a_even_ones, p.a_odd_ones,
         p.b_even_ones, p.b_odd_ones)
        for p in retained
    }
    for profile in target_content_profiles(length, decimation_reduced=True):
        variants = []
        for complement_a in (False, True):
            for complement_b in (False, True):
                ae = half - profile.a_even_ones if complement_a else profile.a_even_ones
                ao = half - profile.a_odd_ones if complement_a else profile.a_odd_ones
                be = half - profile.b_even_ones if complement_b else profile.b_even_ones
                bo = half - profile.b_odd_ones if complement_b else profile.b_odd_ones
                for rotate_a in (False, True):
                    a_counts = (ao, ae) if rotate_a else (ae, ao)
                    for rotate_b in (False, True):
                        b_counts = (bo, be) if rotate_b else (be, bo)
                        variants.extend((a_counts + b_counts, b_counts + a_counts))
        representative = min(variants)
        assert (profile.k, profile.eta, *representative) in retained_keys


def test_l68_safe_filters_remove_only_mathematically_impossible_or_equivalent_cases():
    from solver.frequency_constraints import quarter_frequency_feasible

    complete = target_content_profiles(68, decimation_reduced=True)
    canonical = canonical_target_content_profiles(68)
    filtered = tuple(p for p in canonical if quarter_frequency_feasible(p))
    assert (len(complete), len(canonical), len(filtered)) == (128, 10, 8)
