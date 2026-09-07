"""Lightweight aggregation test for the paired Z3 acceleration benchmark."""

import experiments.benchmark_z3_acceleration as benchmark_module


def test_benchmark_uses_arithmetic_mean_and_records_verified_pairs(monkeypatch):
    pair = ((0, 0, 0, 0), (0, 0, 1, 1))

    def fake_run(_length, seed, _steps, _timeout, optimized):
        return (1.0 if optimized else 2.0), pair[0], pair[1]

    monkeypatch.setattr(benchmark_module, "run_to_first_solution", fake_run)
    result = benchmark_module.benchmark(4, (1, 2), repeats=3, max_steps=10, timeout_ms=10)
    assert result["legacy_mean"] == 2.0
    assert result["optimized_mean"] == 1.0
    assert result["reduction_percent"] == 50.0
    assert len(result["legacy_times"]) == len(result["optimized_times"]) == 6
    assert result["verified_pair_count"] == 1
