"""Lightweight tests for paired score<=8 benchmark aggregation."""

import experiments.benchmark_score8_hits as benchmark_module


def test_benchmark_counts_literal_200_percent_increase(monkeypatch):
    def fake_run(_length, seed, _seconds, proposal_samples):
        optimized = proposal_samples == 10
        hit = optimized or seed == 0
        return {
            "seed": seed,
            "proposal_samples": proposal_samples,
            "initial_a": [0, 1],
            "initial_b": [1, 0],
            "initial_score": 12,
            "hit": hit,
            "time_to_hit": 0.001 if hit else None,
            "best_score": 8 if hit else 10,
            "iterations": 10,
            "elapsed": 0.01,
            "verified": False,
        }

    monkeypatch.setattr(benchmark_module, "run_one", fake_run)
    result = benchmark_module.benchmark(2, (0, 1, 2), 1, 0.01, optimized_samples=10)
    assert result["baseline_hits"] == 1
    assert result["optimized_hits"] == 3
    assert result["hit_factor"] == 3.0
    assert result["hit_increase_percent"] == 200.0
    assert result["initial_mismatches"] == 0


def test_aggregation_rejects_nonidentical_paired_initial_states(monkeypatch):
    def fake_run(_length, _seed, _seconds, proposal_samples):
        return {
            "seed": 0,
            "proposal_samples": proposal_samples,
            "initial_a": [proposal_samples == 1, 0],
            "initial_b": [1, 0],
            "initial_score": 8,
            "hit": True,
            "time_to_hit": 0.0,
            "best_score": 8,
            "iterations": 0,
            "elapsed": 0.0,
            "verified": False,
        }

    monkeypatch.setattr(benchmark_module, "run_one", fake_run)
    try:
        benchmark_module.benchmark(2, (0,), 1, 0.01, optimized_samples=10)
    except AssertionError as error:
        assert "identical FKM" in str(error)
    else:
        raise AssertionError("mismatched paired initialization was not rejected")
