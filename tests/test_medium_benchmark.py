"""Lightweight aggregation and persistence tests; no medium benchmark is run."""

import json

import pytest

from experiments.benchmark_medium_search import (
    aggregate,
    build_best_candidate_event,
    first_reach_time,
    paired_comparison,
    persist_best_candidate,
    run_one,
)
from solver.enhanced_search import (
    EnhancedParameters,
    EnhancedSearch,
    save_enhanced_checkpoint,
)
from solver.objective import pqcp_objective


def _record(seed, score, improvements, snapshots, verified=False):
    return {
        "seed": seed, "best_score": score, "improvements": improvements,
        "snapshots": snapshots, "verified": verified,
    }


def test_threshold_and_aggregate_statistics():
    records = [
        _record(1, 9, [{"elapsed": 0, "score": 30}, {"elapsed": 5, "score": 9}], {"1.0": 20, "5.0": 9}),
        _record(2, 14, [{"elapsed": 0, "score": 25}, {"elapsed": 2, "score": 14}], {"1.0": 25, "5.0": 14}),
        _record(3, 18, [{"elapsed": 0, "score": 18}], {"1.0": 18, "5.0": 18}),
    ]
    assert first_reach_time(records[0], 10) == 5
    assert first_reach_time(records[2], 15) is None
    result = aggregate(records, (1.0, 5.0))
    assert result["minimum"] == 9
    assert result["median"] == 14
    assert result["reached_15"] == {"count": 2, "median_time": 3.5}
    assert result["reached_10"] == {"count": 1, "median_time": 5}
    assert result["progression"][1.0]["median"] == 20


def test_paired_comparison_reports_wins_ties_losses():
    baseline = [
        _record(1, 12, [], {}), _record(2, 10, [], {}), _record(3, 9, [], {}),
    ]
    enhanced = [
        _record(1, 8, [], {}), _record(2, 10, [], {}), _record(3, 11, [], {}),
    ]
    assert paired_comparison(baseline, enhanced) == {
        "enhanced_wins": 1, "ties": 1, "baseline_wins": 1,
    }


def _search():
    return EnhancedSearch.new(
        4, 7, EnhancedParameters(stagnation_iterations=1000, two_bit_samples=0)
    )


def test_global_best_event_persists_sequences_profile_score_and_restart(tmp_path):
    search = _search()
    search.run_steps(1)
    event = build_best_candidate_event(search, "baseline", 1.25, 999999)
    path = persist_best_candidate(event, tmp_path)
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["A"] == list(search.state.best_a)
    assert saved["B"] == list(search.state.best_b)
    assert saved["restart_index"] == search.state.restart_index
    assert saved["profile"] == list(event["profile"])
    assert saved["score"] == pqcp_objective(saved["profile"])
    assert saved["objective_components"]["total"] == saved["score"]


def test_atomic_best_file_replacement_keeps_one_complete_latest_snapshot(tmp_path):
    search = _search()
    first = build_best_candidate_event(search, "baseline", 0.0, None)
    path = persist_best_candidate(first, tmp_path)
    search.run_steps(1)
    second = build_best_candidate_event(search, "baseline", 1.0, first["new_best_score"])
    assert persist_best_candidate(second, tmp_path) == path
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["iteration"] == second["iteration"]
    assert not list(tmp_path.glob("*.tmp"))


def test_short_run_history_contains_full_candidates_and_actual_restart(tmp_path, monkeypatch):
    # This is a persistence test, not authorization to write a tiny-L test
    # solution into the project's real result files.
    from experiments import benchmark_medium_search as benchmark
    monkeypatch.setattr(benchmark, "_record_verified_solution", lambda _event: None)
    record = run_one(4, "baseline", 9, 0.002, (), best_directory=tmp_path)
    assert record["improvements"]
    for event in record["improvements"]:
        assert len(event["A"]) == len(event["B"]) == 4
        assert event["objective_components"]["total"] == event["score"]
        assert "restart_index" in event
        assert event["best_candidate_path"]


def test_score_zero_event_uses_independent_verifier(monkeypatch):
    search = _search()
    a, b = (0, 0, 0, 0), (0, 0, 1, 1)
    search.state.best_a, search.state.best_b, search.state.best_score = a, b, 0
    calls = []
    from experiments import benchmark_medium_search as benchmark
    original = benchmark.verify_pqcp

    def checked(a_value, b_value):
        calls.append((a_value, b_value))
        return original(a_value, b_value)

    monkeypatch.setattr(benchmark, "verify_pqcp", checked)
    event = build_best_candidate_event(search, "baseline", 0.0, 1)
    assert event["verified"] and calls == [(a, b)]


def test_resumed_search_persists_a_new_improvement_with_its_restart_index(tmp_path):
    from solver.weight_constraints import canonical_weight_pairs
    parameters = EnhancedParameters(
        stagnation_iterations=1000, two_bit_samples=0,
        weight_pairs=canonical_weight_pairs(8),
    )
    original = EnhancedSearch.new(8, 0, parameters)
    original.step()
    checkpoint = tmp_path / "resume.json"
    save_enhanced_checkpoint(checkpoint, original.state)
    resumed = EnhancedSearch.resume(checkpoint)
    old_best = resumed.state.best_score
    for _ in range(20):
        resumed.step()
        if resumed.state.best_score < old_best:
            break
    assert resumed.state.best_score < old_best
    event = build_best_candidate_event(resumed, "baseline", 2.0, old_best)
    path = persist_best_candidate(event, tmp_path / "best")
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["score"] == resumed.state.best_score
    assert saved["restart_index"] == resumed.state.restart_index
