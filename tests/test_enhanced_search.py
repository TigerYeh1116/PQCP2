"""Exactness, rollback, deterministic escape, and JSON-resume tests."""

import pytest

from solver.compression import CorrelationState
from solver.correlation import full_correlation_profile
from solver.enhanced_search import (
    EnhancedParameters,
    EnhancedSearch,
    TwoBitMove,
    load_enhanced_checkpoint,
    save_enhanced_checkpoint,
)
from solver.objective import pqcp_objective


def _search(length=6, seed=12, **kwargs):
    defaults = dict(stagnation_iterations=1000, two_bit_samples=8)
    defaults.update(kwargs)
    return EnhancedSearch.new(length, seed, EnhancedParameters(**defaults))


def _deterministic_state(search):
    state = search.state.to_dict()
    state.pop("escape_evaluation_seconds")
    state.pop("elapsed_seconds")
    return state


def _valid_move(search, kind):
    values = search.state.current_a if kind == "AA" else search.state.current_b
    return TwoBitMove(kind, values.index(0), values.index(1))


@pytest.mark.parametrize("kind", ["AA", "BB"])
def test_exact_two_bit_trials_restore_state_and_match_full_profile(kind):
    search = _search()
    move = _valid_move(search, kind)
    before = (search.state.current_a, search.state.current_b, search.state.current_score, search._correlation.profile)
    trial_score = search.trial_two_bit_move(move)
    state = CorrelationState(before[0], before[1])
    search_move = EnhancedSearch._apply_move
    search_move(search, move)
    expected_profile = tuple(full_correlation_profile(search._correlation.a, search._correlation.b))
    assert search._correlation.profile == expected_profile
    assert trial_score == pqcp_objective(expected_profile)
    search._rollback_move(move)
    assert (search._correlation.a, search._correlation.b, search.state.current_score, search._correlation.profile) == before


def test_rejected_two_bit_trial_keeps_current_objective_exact():
    search = _search()
    score = search.state.current_score
    search.trial_two_bit_move(_valid_move(search, "AA"))
    assert search.state.current_score == score
    assert search._correlation.profile == tuple(full_correlation_profile(search.state.current_a, search.state.current_b))


def test_same_seed_has_identical_enhanced_trajectory():
    first = _search(seed=55, stagnation_iterations=4, two_bit_samples=5)
    second = _search(seed=55, stagnation_iterations=4, two_bit_samples=5)
    first.run_steps(30)
    second.run_steps(30)
    assert _deterministic_state(first) == _deterministic_state(second)


def test_stagnation_trigger_and_global_best_survives_worse_escape():
    search = _search(stagnation_iterations=1, two_bit_samples=4, max_escape_fraction=1.0)
    original_best = search.state.best_score
    search._handle_stagnation()
    assert search.state.escape_triggers == 1
    assert search.state.sampled_moves == 4
    assert search.state.best_score <= original_best


def test_best_sampled_chooses_lowest_sampled_score(monkeypatch):
    search = _search()
    moves = iter((TwoBitMove("AA", 0, 1), TwoBitMove("BB", 0, 1), TwoBitMove("AA", 1, 2)))
    monkeypatch.setattr(search, "_random_two_bit_move", lambda: next(moves))
    scores = iter((9, 3, 5))
    monkeypatch.setattr(search, "trial_two_bit_move", lambda _move: next(scores))
    move, score = search._sample_best_move(3)
    assert move.kind == "BB" and score == 3


def test_checkpoint_resume_matches_continuous_steps(tmp_path):
    parameters = EnhancedParameters(stagnation_iterations=5, two_bit_samples=4)
    continuous = EnhancedSearch.new(6, 33, parameters)
    split = EnhancedSearch.new(6, 33, parameters)
    continuous.run_steps(30)
    split.run_steps(11)
    path = tmp_path / "enhanced.json"
    save_enhanced_checkpoint(path, split.state)
    resumed = EnhancedSearch.resume(path)
    resumed.run_steps(19)
    assert _deterministic_state(resumed) == _deterministic_state(continuous)
    assert load_enhanced_checkpoint(path).L == 6


@pytest.mark.parametrize("length", [4, 6, 8])
def test_tiny_lengths_preserve_objective_consistency(length):
    search = _search(length=length, stagnation_iterations=3, two_bit_samples=3)
    search.run_steps(12)
    assert search.state.current_score == pqcp_objective(search._correlation.profile)
    assert search._correlation.profile == tuple(full_correlation_profile(search.state.current_a, search.state.current_b))


def test_score_zero_initial_state_is_independently_verified():
    search = _search(length=4)
    a, b = (0, 0, 0, 0), (0, 0, 1, 1)
    correlation = CorrelationState(a, b)
    search._correlation = correlation
    search.state.current_a, search.state.current_b = a, b
    search.state.current_score = 0
    result = search.run(0)
    assert result.verified and result.state.finished


def test_invalid_same_bit_double_flip_is_rejected():
    with pytest.raises(ValueError):
        TwoBitMove("AA", 1, 1)


def test_cross_sequence_two_bit_move_is_rejected_because_it_changes_both_weights():
    with pytest.raises(ValueError, match="AA or BB"):
        TwoBitMove("AB", 0, 1)


def test_regular_and_escape_moves_preserve_both_hamming_weights():
    search = _search(length=8, seed=91, stagnation_iterations=3, two_bit_samples=5)
    expected = (sum(search.state.current_a), sum(search.state.current_b))
    for _ in range(50):
        search.step()
        assert (sum(search.state.current_a), sum(search.state.current_b)) == expected
