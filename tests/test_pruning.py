"""Brute-force safety tests for mathematical Project 2 pruning rules."""

from itertools import product

import pytest

from solver.correlation import full_correlation_profile
from solver.pruning import (
    partial_correlation_bounds,
    prune_complete_profile,
    prune_partial_assignment,
)


def _is_project2_target(profile):
    length = len(profile)
    nonzero = [shift for shift in range(1, length) if profile[shift] != 0]
    return (
        profile[0] == 2 * length
        and len(nonzero) == 2
        and all(abs(profile[shift]) == 4 for shift in nonzero)
        and all(profile[shift] == profile[(-shift) % length] for shift in range(1, length))
    )


@pytest.mark.parametrize("length", range(1, 9))
def test_complete_pruning_never_rejects_a_brute_force_target(length):
    words = tuple(product((0, 1), repeat=length))
    for a in words:
        for b in words:
            profile = full_correlation_profile(a, b)
            if _is_project2_target(profile):
                assert not prune_complete_profile(profile).should_prune


@pytest.mark.parametrize("length", range(1, 9))
def test_partial_pruning_never_rejects_partial_views_of_brute_force_target(length):
    words = tuple(product((0, 1), repeat=length))
    for a in words:
        for b in words:
            profile = full_correlation_profile(a, b)
            if _is_project2_target(profile):
                partial_a = tuple(bit if index % 2 == 0 else None for index, bit in enumerate(a))
                partial_b = tuple(bit if index % 2 == 1 else None for index, bit in enumerate(b))
                assert not prune_partial_assignment(partial_a, partial_b).should_prune


def test_complete_rules_reject_specific_necessary_condition_violations():
    assert "S[0] is not 2L" in prune_complete_profile((7, 4, 0, 4)).reasons
    assert "periodic symmetry S[u] = S[L-u] is violated" in prune_complete_profile((8, 4, 0, -4)).reasons
    assert "binary pair-correlation congruence modulo 4 is violated" in prune_complete_profile((8, 2, 0, 2)).reasons
    assert "a nonzero shift has magnitude other than 4" in prune_complete_profile((8, 8, 0, 8)).reasons
    assert "the actual nonzero-shift count is not two" in prune_complete_profile((8, 0, 0, 0)).reasons


def test_partial_bounds_are_conservative_for_a_known_completion():
    a = (0, None, 1, None)
    b = (None, 1, None, 0)
    completion_a = (0, 0, 1, 1)
    completion_b = (1, 1, 0, 0)
    profile = full_correlation_profile(completion_a, completion_b)
    bounds = partial_correlation_bounds(a, b)
    assert all(lower <= value <= upper
               for lower, value, upper in zip(bounds.lower, profile, bounds.upper))


def test_partial_rules_prune_only_proven_impossibilities():
    assert not prune_partial_assignment((None,) * 4, (None,) * 4).should_prune
    result = prune_partial_assignment((0, 0, 0, 0), (0, 0, 0, 0))
    assert result.should_prune
    assert "a shift bound cannot reach any allowed target value" in result.reasons


def test_partial_pruning_rejects_odd_lengths_by_proven_congruence():
    # For odd L, every binary pair S(u) is 2L mod 4, but 0 and +/-4 are 0 mod 4.
    assert prune_partial_assignment((None,) * 3, (None,) * 3).should_prune


def test_partial_validation_rejects_invalid_values():
    with pytest.raises(ValueError):
        partial_correlation_bounds((0, 2), (0, None))
