"""Brute-force safety tests for mathematical Project 2 pruning rules."""

from itertools import product

import pytest

from solver.correlation import full_correlation_profile
from solver.pruning import (
    fixed_content_partial_correlation_bounds,
    partial_correlation_bounds,
    prune_complete_profile,
    prune_fixed_content_partial_assignment,
    prune_partial_assignment,
)
from solver.target_profiles import TargetContentProfile, pair_content, target_content_profiles


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


@pytest.mark.parametrize("length", (4, 6, 8))
def test_fixed_content_bounds_contain_every_compatible_completion(length):
    target = target_content_profiles(length)[0]
    words = tuple(product((0, 1), repeat=length))
    a_words = tuple(word for word in words if (
        sum(word[0::2]), sum(word[1::2])
    ) == (target.a_even_ones, target.a_odd_ones))
    b_words = tuple(word for word in words if (
        sum(word[0::2]), sum(word[1::2])
    ) == (target.b_even_ones, target.b_odd_ones))
    assert a_words and b_words

    # Several independently shaped partial views are enough to exercise
    # known-known, boundary and unknown-unknown edge accounting.  Every
    # compatible completion is still checked exhaustively.
    masks = (
        tuple(index % 2 == 0 for index in range(length)),
        tuple(index % 3 == 0 for index in range(length)),
        tuple(index not in (1, length - 1) for index in range(length)),
    )
    for mask in masks:
        partial_a = tuple(bit if mask[index] else None
                          for index, bit in enumerate(a_words[-1]))
        partial_b = tuple(bit if mask[-1-index] else None
                          for index, bit in enumerate(b_words[-1]))
        bounds = fixed_content_partial_correlation_bounds(
            partial_a, partial_b, target
        )
        assert bounds.content_feasible
        completions_a = tuple(
            word for word in a_words
            if all(known is None or known == bit
                   for known, bit in zip(partial_a, word))
        )
        completions_b = tuple(
            word for word in b_words
            if all(known is None or known == bit
                   for known, bit in zip(partial_b, word))
        )
        for a in completions_a:
            for b in completions_b:
                profile = full_correlation_profile(a, b)
                assert all(
                    lower <= value <= upper
                    for lower, value, upper in zip(
                        bounds.lower, profile, bounds.upper
                    )
                )


def test_fixed_content_bounds_are_stricter_than_unknown_term_bounds():
    target = TargetContentProfile(4, 1, -1, 0, 2, 1, 1)
    a = (0, 1, 0, None)
    b = (None,) * 4
    coarse = partial_correlation_bounds(a, b)
    fixed = fixed_content_partial_correlation_bounds(a, b, target)
    assert fixed.content_feasible
    assert fixed.upper[1] < coarse.upper[1]
    assert fixed.lower[2] > coarse.lower[2]
    assert fixed.lower[0] == fixed.upper[0] == 8


def test_complete_fixed_content_assignment_returns_exact_profile():
    target = TargetContentProfile(4, 1, -1, 0, 2, 1, 1)
    a = (0, 1, 0, 1)
    b = (0, 0, 1, 1)
    bounds = fixed_content_partial_correlation_bounds(a, b, target)
    exact = tuple(full_correlation_profile(a, b))
    assert bounds.content_feasible
    assert bounds.lower == bounds.upper == exact


def test_fixed_content_pruning_detects_more_than_generic_partial_bounds():
    target = TargetContentProfile(6, 1, -1, 1, 2, 0, 2)
    a = (0, 0, 0, 1, 1, 1)
    b = (None,) * 6
    assert not prune_partial_assignment(a, b).should_prune
    result = prune_fixed_content_partial_assignment(a, b, target)
    assert result.should_prune
    assert "fixed-content shift bound" in result.reasons[0]
    assert prune_partial_assignment(a, b, target).should_prune


def test_impossible_remaining_content_is_pruned_before_correlation():
    target = TargetContentProfile(4, 1, -1, 0, 2, 1, 1)
    bounds = fixed_content_partial_correlation_bounds(
        (1, None, None, None), (None,) * 4, target
    )
    assert not bounds.content_feasible
    result = prune_fixed_content_partial_assignment(
        (1, None, None, None), (None,) * 4, target
    )
    assert result.should_prune
    assert "fixed parity content" in result.reasons[0]


@pytest.mark.parametrize("length", (4, 6, 8))
def test_no_partial_view_of_an_actual_solution_is_fixed_content_pruned(length):
    words = tuple(product((0, 1), repeat=length))
    for a in words:
        for b in words:
            profile = full_correlation_profile(a, b)
            if not _is_project2_target(profile):
                continue
            shift = next(u for u in range(1, length) if profile[u])
            target = TargetContentProfile(
                length, min(shift, length - shift), profile[shift] // 4,
                *pair_content(a, b)
            )
            partial_a = tuple(bit if index % 3 else None
                              for index, bit in enumerate(a))
            partial_b = tuple(bit if index % 3 == 1 else None
                              for index, bit in enumerate(b))
            assert not prune_fixed_content_partial_assignment(
                partial_a, partial_b, target
            ).should_prune
