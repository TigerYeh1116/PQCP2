"""Lightweight deterministic tests for the parallel-search benchmark."""

import pytest

from experiments.benchmark_parallel_search import (
    SEED_STRIDE,
    counterbalanced_counts,
    summarize,
    worker_seed,
)


def test_worker_seed_schedule_is_nested_and_nonoverlapping():
    assert [worker_seed(123, index) for index in range(3)] == [
        123, 123 + SEED_STRIDE, 123 + 2 * SEED_STRIDE,
    ]
    with pytest.raises(ValueError):
        worker_seed(123, -1)


def test_counterbalanced_counts_rotates_without_changing_members():
    counts = (1, 2, 4, 8)
    assert counterbalanced_counts(counts, 0) == counts
    assert counterbalanced_counts(counts, 1) == (2, 4, 8, 1)
    assert counterbalanced_counts(counts, 4) == counts


def test_parallel_summary_uses_one_worker_rate_as_speedup_baseline():
    rows = [
        {
            "L": 44, "workers": 1, "aggregate_iterations_per_second": 100.0,
            "parent_wall": 1.0, "minimum_best_score": 12,
            "median_best_score": 12, "verified": 0,
        },
        {
            "L": 44, "workers": 4, "aggregate_iterations_per_second": 350.0,
            "parent_wall": 1.0, "minimum_best_score": 8,
            "median_best_score": 10, "verified": 1,
        },
    ]
    result = summarize(rows)
    assert result[0]["speedup_vs_one_worker"] == 1.0
    assert result[1]["speedup_vs_one_worker"] == 3.5
    assert result[1]["verified"] == 1
