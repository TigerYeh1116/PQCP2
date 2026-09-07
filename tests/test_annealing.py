"""Reproducibility and exactness tests for baseline simulated annealing."""

import pytest

from solver.annealing import AnnealingParameters, initialize_from_fkm, simulated_annealing
from solver.correlation import full_correlation_profile
from solver.fkm import is_cyclic_representative
from solver.objective import pqcp_objective


def _parameters(seed=123):
    return AnnealingParameters(
        max_iterations=80,
        initial_temperature=6.0,
        cooling_rate=0.97,
        min_temperature=0.1,
        seed=seed,
        restart_count=2,
        fkm_pool_size=16,
    )


def test_fkm_initialization_is_seeded_and_uses_cyclic_representatives():
    first = initialize_from_fkm(8, seed=19, pool_size=16)
    second = initialize_from_fkm(8, seed=19, pool_size=16)
    assert first == second
    assert all(is_cyclic_representative(sequence) for sequence in first)


def test_same_seed_and_parameters_produce_same_trajectory_and_result():
    first = simulated_annealing(6, _parameters(seed=123))
    second = simulated_annealing(6, _parameters(seed=123))
    assert first.best_a == second.best_a
    assert first.best_b == second.best_b
    assert first.best_score == second.best_score
    assert first.initial_scores == second.initial_scores
    assert first.accepted_moves == second.accepted_moves
    assert first.best_score_history == second.best_score_history


@pytest.mark.parametrize("length", [4, 6, 8])
def test_tiny_search_results_match_independent_full_objective(length):
    result = simulated_annealing(length, _parameters(seed=100 + length))
    profile = tuple(full_correlation_profile(result.best_a, result.best_b))
    assert result.best_profile == profile
    assert result.best_score == pqcp_objective(profile)


def test_best_score_history_is_monotone_nonincreasing():
    result = simulated_annealing(8, _parameters())
    assert all(after <= before for before, after in zip(
        result.best_score_history, result.best_score_history[1:]
    ))


def test_perfect_initial_pair_is_preserved_and_independently_verified():
    result = simulated_annealing(
        4,
        _parameters(),
        initial_pair=((0, 0, 0, 0), (0, 0, 1, 1)),
    )
    assert result.solved
    assert result.independently_verified
    assert result.best_score == 0
    assert result.best_a == (0, 0, 0, 0)
    assert result.best_b == (0, 0, 1, 1)
    assert result.iterations == 0


def test_legacy_annealing_keeps_initial_pair_weights_fixed():
    initial = ((0, 0, 0, 1, 1, 1), (0, 0, 1, 0, 1, 1))
    result = simulated_annealing(6, _parameters(seed=44), initial_pair=initial)
    assert sum(result.best_a) == sum(initial[0])
    assert sum(result.best_b) == sum(initial[1])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_iterations": -1},
        {"initial_temperature": 0},
        {"cooling_rate": 0},
        {"min_temperature": 9, "initial_temperature": 8},
        {"restart_count": 0},
    ],
)
def test_invalid_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        AnnealingParameters(**kwargs)
