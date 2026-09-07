"""Lightweight aggregation tests for the score<4 hit benchmark."""

import experiments.benchmark_best_score_hits as benchmark_module


def test_hit_benchmark_counts_paired_runs_and_percentage(monkeypatch):
    def fake_run(_length, seed, _seconds, guided):
        hit = guided or seed == 0
        return {
            "seed": seed, "guided": guided, "hit": hit,
            "best_score": 3 if hit else 4, "iterations": 10,
            "elapsed": 0.01, "verified": False,
        }

    monkeypatch.setattr(benchmark_module, "run_one", fake_run)
    result = benchmark_module.benchmark(24, (0, 1, 2), repeats=2, seconds=0.01)
    assert result["baseline_hits"] == 2
    assert result["guided_hits"] == 6
    assert result["hit_factor"] == 3.0
    assert result["hit_increase_percent"] == 200.0
