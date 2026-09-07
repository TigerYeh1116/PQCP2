"""Tests for PyTorch pipeline benchmark comparison logic."""

from experiments.benchmark_torch_pipeline import paired, summarize


def test_summary_and_paired_comparison():
    rows = [
        {"L": 44, "seed": 1, "method": "sa", "best_score": 12, "verified": False,
         "iterations": 100, "torch_seconds": 0.0},
        {"L": 44, "seed": 1, "method": "torch_warm", "best_score": 8, "verified": False,
         "iterations": 80, "torch_seconds": 0.2},
        {"L": 44, "seed": 2, "method": "sa", "best_score": 8, "verified": False,
         "iterations": 100, "torch_seconds": 0.0},
        {"L": 44, "seed": 2, "method": "torch_warm", "best_score": 8, "verified": False,
         "iterations": 80, "torch_seconds": 0.2},
    ]
    assert paired(rows, "torch_warm") == {"torch_wins": 1, "ties": 1, "sa_wins": 0}
    summary = summarize(rows)
    assert len(summary) == 2
    assert next(row for row in summary if row["method"] == "torch_warm")["median_score"] == 8
