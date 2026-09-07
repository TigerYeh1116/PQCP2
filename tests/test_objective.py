"""Exact integer objective tests for Project 2 target profiles."""

import pytest

from solver.objective import (
    evaluate_objective,
    pqcp_objective,
    pqcp_objective_breakdown,
    target_profile_l1_distance,
    target_pair_squared_energy,
)


def test_perfect_symmetric_target_profile_has_zero_objective():
    # Nonzero shifts 1 and 4 form one periodic-symmetry pair for L=5.
    profile = (10, 4, 0, 0, 4)
    result = evaluate_objective(profile)
    assert result.total == 0
    assert result.nonzero_shifts == (1, 4)
    assert pqcp_objective(profile) == 0


def test_objective_penalizes_extra_nonzero_shifts():
    result = evaluate_objective((8, 4, 4, 4))
    assert result.zero_shift_deviation == 0
    assert result.nonzero_count_deviation == 1
    assert result.total > 0


def test_objective_penalizes_magnitude_other_than_four():
    result = evaluate_objective((8, 2, 0, 2))
    assert result.zero_shift_deviation == 4
    assert result.total > 0


def test_objective_penalizes_origin_and_symmetry_deviations():
    result = evaluate_objective((7, 4, 0, 0, -4))
    assert result.origin_deviation == 3
    assert result.symmetry_deviation == 16
    assert result.total > 0


def test_l44_symmetric_target_profile_has_zero_objective():
    profile = [0] * 44
    profile[0] = 88
    profile[4] = -4
    profile[40] = -4
    assert pqcp_objective(profile) == 0
    assert target_pair_squared_energy(profile) == 0
    assert target_profile_l1_distance(profile) == 0


def test_target_l1_commits_to_exactly_one_nonzero_orbit():
    # Existing objective allows each +/-4 entry locally and then adds only a
    # count penalty.  Complete-target L1 must remove the extra orbit itself.
    profile = (12, 4, 4, 0, 4, 4)
    assert target_profile_l1_distance(profile) == 2


def test_target_pair_energy_penalizes_extra_sidelobes_and_half_shift():
    assert target_pair_squared_energy((8, 4, 4, 4)) > 0
    assert target_pair_squared_energy((8, 0, 4, 0)) > 0


def test_json_safe_breakdown_preserves_exact_objective_total():
    profile = (10, 4, 2, 4, 0)
    breakdown = pqcp_objective_breakdown(profile)
    assert breakdown["total"] == pqcp_objective(profile)
    assert set(breakdown) == {
        "target_value_distance", "nonzero_count_penalty", "s0_penalty",
        "symmetry_penalty", "total",
    }


@pytest.mark.parametrize("profile", [(), (2, 0.0), (2, True)])
def test_objective_rejects_invalid_profiles(profile):
    with pytest.raises(ValueError):
        evaluate_objective(profile)
