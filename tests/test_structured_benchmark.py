"""Lightweight aggregation tests; no long benchmark is run by pytest."""

from experiments.benchmark_structured_search import summarize


def test_structured_summary_groups_scores_and_solutions():
    rows = [
        {"L": 8, "mode": "full", "best_score": 0, "evaluations_per_second": 100.0},
        {"L": 8, "mode": "full", "best_score": 4, "evaluations_per_second": 200.0},
    ]
    result = summarize(rows)[0]
    assert result["min_score"] == 0
    assert result["median_score"] == 2
    assert result["solutions"] == 1
    assert result["mean_evaluations_per_second"] == 150.0
