"""Correctness and separation tests for the empirical difficulty score."""

from solver.correlation import full_correlation_profile
from solver.difficulty import lookahead_repair_difficulty, repair_difficulty
from solver.objective import target_profile_l1_distance


def test_verified_l4_pair_has_zero_repair_difficulty():
    # S=(8,-4,0,-4), a valid L=4 target.
    a = (0, 0, 0, 0)
    b = (0, 0, 1, 1)
    result = repair_difficulty(a, b)
    assert result.total == 0
    assert result.target_l1 == 0
    assert result.local_barrier == 0


def test_difficulty_preserves_state_and_reports_complete_neighborhood():
    a = (0, 0, 0, 1, 1, 1)
    b = (0, 0, 1, 0, 1, 1)
    before = full_correlation_profile(a, b)
    result = repair_difficulty(a, b)
    assert full_correlation_profile(a, b) == before
    assert result.target_l1 == target_profile_l1_distance(before)
    assert result.legal_moves == 18
    assert 0 <= result.improving_moves <= result.legal_moves
    assert result.total == result.target_l1 * (1 + result.local_barrier)


def test_lookahead_reports_exact_one_swap_cost_to_known_solution():
    solution_a = (1, 0, 0, 0, 0, 0)
    solution_b = (1, 1, 0, 0, 0, 0)
    perturbed_b = (1, 0, 0, 1, 0, 0)
    exact = lookahead_repair_difficulty(solution_a, solution_b, max_depth=2, beam_width=None)
    nearby = lookahead_repair_difficulty(solution_a, perturbed_b, max_depth=2, beam_width=None)
    assert exact.total == 0
    assert exact.solution_depth == 0
    assert nearby.total == 1
    assert nearby.best_depth == 1
    assert nearby.residual_target_l1 == 0
    assert nearby.solution_depth == 1


def test_lookahead_validation_rejects_invalid_controls():
    a = b = (0, 0, 1, 1)
    import pytest
    with pytest.raises(ValueError):
        lookahead_repair_difficulty(a, b, max_depth=-1)
    with pytest.raises(ValueError):
        lookahead_repair_difficulty(a, b, beam_width=0)


def test_lookahead_penalizes_exhausted_non_solution_probe():
    a = b = (0, 0, 0, 1, 1, 1)
    result = lookahead_repair_difficulty(a, b, max_depth=0, beam_width=1)
    assert result.solution_depth is None
    assert result.total == 1 + result.initial_target_l1
