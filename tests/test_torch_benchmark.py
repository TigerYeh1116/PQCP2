"""Aggregation tests only; pytest never runs the expensive benchmark."""

from experiments.benchmark_torch_relaxation import summarize


def test_summary_separates_exact_recovery_from_score_improvement():
    rows = [
        {"kind": "controlled", "swaps": 1, "torch_verified": True,
         "sa_verified": False, "torch_score": 0, "sa_score": 8,
         "torch_distance_to_center": 0, "sa_distance_to_center": 6},
        {"kind": "real_elite", "initial_score": 10, "torch_score": 8,
         "torch_verified": False},
    ]
    result = summarize(rows)
    assert result["torch_verified_total"] == 1
    assert result["sa_verified_total"] == 0
    assert result["real_elite_improved"] == 1
    assert result["real_elite_verified"] == 0
