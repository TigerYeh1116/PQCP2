"""Lightweight arithmetic-mean and novelty checks for the GCP benchmark."""

import experiments.benchmark_golay_seeding as benchmark_module


PAIRS = (
    ((0, 0, 0, 0), (0, 0, 1, 1)),
    ((0, 0, 0, 0), (0, 1, 1, 0)),
)


def test_benchmark_uses_arithmetic_mean_and_distinct_pairs(monkeypatch):
    monkeypatch.setattr(
        benchmark_module,
        "run_fkm_to_solution",
        lambda _length, seed, _steps: (2.0, PAIRS[seed]),
    )
    monkeypatch.setattr(
        benchmark_module,
        "run_golay_to_solution",
        lambda _length, seed: (0.2, PAIRS[seed]),
    )
    result = benchmark_module.benchmark(4, (0, 1), repeats=2, max_steps=10)
    assert result["fkm_mean"] == 2.0
    assert result["golay_mean"] == 0.2
    assert result["time_reduction_percent"] == 90.0
    assert result["unique_per_repeat"] == (2, 2)
