"""Bounded-run persistence, interruption, and verified-solution handling tests."""

import json

import pytest

from solver.checkpoint import SearchParameters
from solver.compression import CorrelationState
from solver.objective import pqcp_objective
from solver.search_runner import SearchRunner, append_verified_solution_if_new
from solver.target_profiles import canonical_target_content_profiles, pair_content


class AdvancingClock:
    """Deterministic monotonic clock for fast time/periodic-save tests."""

    def __init__(self, increment=0.1):
        self.value = 0.0
        self.increment = increment

    def __call__(self):
        self.value += self.increment
        return self.value


def test_time_budget_and_periodic_checkpoint(tmp_path):
    runner = SearchRunner.new(6, 123, SearchParameters(stagnation_iterations=None))
    checkpoint = tmp_path / "checkpoint.json"
    summary = runner.run(
        seconds=0.5,
        checkpoint_path=checkpoint,
        checkpoint_interval=0.1,
        clock=AdvancingClock(),
    )
    assert checkpoint.exists()
    assert summary.state.iteration > 0
    assert not summary.interrupted


def test_best_candidate_persistence_contains_exact_profile(tmp_path):
    runner = SearchRunner.new(4, 7, SearchParameters(stagnation_iterations=None))
    best_path = tmp_path / "best.json"
    runner.run(seconds=0, best_path=best_path)
    payload = json.loads(best_path.read_text(encoding="utf-8"))
    assert payload["score"] == runner.state.best_score
    assert payload["correlation_profile"] == list(CorrelationState(payload["a"], payload["b"]).profile)


def test_keyboard_interrupt_saves_checkpoint_and_best(tmp_path):
    runner = SearchRunner.new(4, 8, SearchParameters(stagnation_iterations=None))
    checkpoint = tmp_path / "checkpoint.json"
    best_path = tmp_path / "best.json"

    def interrupt():
        raise KeyboardInterrupt

    runner.step = interrupt
    summary = runner.run(seconds=1, checkpoint_path=checkpoint, best_path=best_path)
    assert summary.interrupted
    assert checkpoint.exists() and best_path.exists()


def test_verified_initial_solution_is_checked_and_reported(tmp_path):
    parameters = SearchParameters(stagnation_iterations=None, max_restarts=1)
    runner = SearchRunner.new(4, 9, parameters)
    a, b = (0, 0, 0, 0), (0, 0, 1, 1)
    state = CorrelationState(a, b)
    runner._correlation = state
    runner.state.current_a = a
    runner.state.current_b = b
    runner.state.current_score = pqcp_objective(state.profile)
    runner.state.best_a = a
    runner.state.best_b = b
    runner.state.best_score = 0
    found = []
    summary = runner.run(seconds=0, on_verified_solution=lambda x, y, _state: found.append((x, y)))
    assert summary.verified_solution_found
    assert found == [(a, b)]


def test_project_solution_append_is_new_then_duplicate_safe(tmp_path):
    a, b = (0, 0, 0, 0), (0, 0, 1, 1)
    assert append_verified_solution_if_new(4, a, b, tmp_path)
    content = (tmp_path / "4.txt").read_text(encoding="utf-8")
    assert content == (
        "\nL=4\nnonzero shifts=1,3\nnonzero PACS=4\n"
        "a=0000\nb=0011\n"
    )
    assert not append_verified_solution_if_new(4, a, b, tmp_path)
    assert not append_verified_solution_if_new(4, b, a, tmp_path)
    assert (tmp_path / "4.txt").read_text(encoding="utf-8") == content


@pytest.mark.parametrize("duplicate", [False, True])
def test_concurrent_solution_writers_cannot_lose_or_double_count_pairs(tmp_path, monkeypatch, duplicate):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    import solver.search_runner as module

    entered = threading.Event()
    release = threading.Event()
    second_started = threading.Event()
    original_write = module._atomic_write_text

    def paused_write(path, text):
        if not entered.is_set():
            entered.set()
            assert release.wait(5)
        original_write(path, text)

    monkeypatch.setattr(module, "_atomic_write_text", paused_write)
    a, b = (0, 0, 0, 0), (0, 0, 1, 1)
    other = (b, a) if duplicate else (a, (0, 1, 1, 0))

    def second_write():
        second_started.set()
        return append_verified_solution_if_new(4, *other, tmp_path)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(append_verified_solution_if_new, 4, a, b, tmp_path)
        try:
            assert entered.wait(5)
            second = pool.submit(second_write)
            assert second_started.wait(5)
        finally:
            release.set()
        assert first.result(timeout=5)
        assert second.result(timeout=5) is (not duplicate)
    text = (tmp_path / "4.txt").read_text()
    assert text.count("\na=") == (1 if duplicate else 2)
    assert "a=0000\nb=0011" in text
    if not duplicate:
        assert "a=0000\nb=0110" in text


def test_observation_callback_receives_restart_without_changing_run(tmp_path):
    runner = SearchRunner.new(4, 10, SearchParameters(stagnation_iterations=None, max_iterations_per_restart=1))
    events = []
    runner.run(
        seconds=0.5,
        checkpoint_path=tmp_path / "state.json",
        event_callback=lambda outcome, _state: events.append(outcome),
        clock=AdvancingClock(),
    )
    assert any(event.restart_info is not None for event in events)


def test_every_baseline_move_preserves_both_hamming_weights():
    runner = SearchRunner.new(8, 123, SearchParameters(stagnation_iterations=None))
    expected = (sum(runner.state.current_a), sum(runner.state.current_b))
    for _ in range(200):
        runner.step()
        assert (sum(runner.state.current_a), sum(runner.state.current_b)) == expected


def test_proposal_samples_must_be_a_positive_integer():
    for value in (0, -1, 1.5, True):
        with pytest.raises(ValueError):
            SearchParameters(proposal_samples=value)


def test_objective_energy_weight_must_be_a_positive_integer():
    for value in (0, -1, 1.5, True):
        with pytest.raises(ValueError):
            SearchParameters(objective_energy_weight=value)


def test_multisample_runner_is_deterministic_for_same_seed():
    parameters = SearchParameters(
        stagnation_iterations=None,
        acceptance_mode="objective_plus_target_pair",
        proposal_samples=8,
    )
    first = SearchRunner.new(8, 418, parameters)
    second = SearchRunner.new(8, 418, parameters)
    for _ in range(100):
        first.step()
        second.step()
    assert first.state.to_dict() == second.state.to_dict()


def test_profile_initialized_runner_preserves_all_four_parity_counts():
    profile = canonical_target_content_profiles(8)[0]
    encoded = ((profile.k, profile.eta, profile.a_even_ones,
                profile.a_odd_ones, profile.b_even_ones, profile.b_odd_ones),)
    parameters = SearchParameters(
        stagnation_iterations=None,
        target_content_profiles=encoded,
        preserve_alternating_content=True,
        acceptance_mode="fixed_target_multiscale",
        proposal_samples=4,
    )
    runner = SearchRunner.new(8, 777, parameters)
    expected = encoded[0][2:]
    for _ in range(100):
        runner.step()
        assert pair_content(runner.state.current_a, runner.state.current_b) == expected


def test_compressed_tiebreak_runner_is_deterministic_and_preserves_content():
    profile = canonical_target_content_profiles(44)[0]
    encoded = ((profile.k, profile.eta, profile.a_even_ones,
                profile.a_odd_ones, profile.b_even_ones, profile.b_odd_ones),)
    parameters = SearchParameters(
        stagnation_iterations=None,
        target_content_profiles=encoded,
        preserve_alternating_content=True,
        acceptance_mode="fixed_target_full_compressed_tiebreak",
        proposal_samples=6,
        fkm_seed_policy="phase_random",
    )
    first = SearchRunner.new(44, 6001, parameters)
    second = SearchRunner.new(44, 6001, parameters)
    expected = encoded[0][2:]
    for _ in range(50):
        first.step()
        second.step()
        assert pair_content(first.state.current_a, first.state.current_b) == expected
    assert first.state.to_dict() == second.state.to_dict()


def test_profile_runner_checkpoint_resume_is_deterministic(tmp_path):
    profiles = canonical_target_content_profiles(8)[:2]
    encoded = tuple((p.k, p.eta, p.a_even_ones, p.a_odd_ones,
                     p.b_even_ones, p.b_odd_ones) for p in profiles)
    parameters = SearchParameters(
        stagnation_iterations=20,
        target_content_profiles=encoded,
        preserve_alternating_content=True,
        acceptance_mode="fixed_target_full",
    )
    continuous = SearchRunner.new(8, 81, parameters)
    resumed = SearchRunner.new(8, 81, parameters)
    for _ in range(35):
        continuous.step()
    for _ in range(17):
        resumed.step()
    from solver.checkpoint import save_checkpoint
    path = tmp_path / "profile.json"
    save_checkpoint(path, resumed.state)
    resumed = SearchRunner.resume(path)
    for _ in range(18):
        resumed.step()
    assert continuous.state.to_dict() == resumed.state.to_dict()


def test_completion_miss_restart_uses_existing_deterministic_seed_schedule():
    parameters = SearchParameters(stagnation_iterations=None)
    first = SearchRunner.new(8, 321, parameters)
    second = SearchRunner.new(8, 321, parameters)
    info = first.restart_after_completion_miss()
    second._restart("test equivalent")
    assert info.reason == "bounded Z3 completion miss"
    assert first.state.restart_index == 1
    assert first.state.to_dict() == second.state.to_dict()
