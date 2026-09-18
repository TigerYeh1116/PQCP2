"""Aggregation only; pytest never runs the timed MPS experiment."""

from experiments.benchmark_torch_mathematical_loss import summarize


def test_summary_uses_exact_discrete_outputs():
    rows = [
        {"L": 44, "mathematical_loss": "none", "projected_best_score": 12,
         "target_best": 40, "post_polish_best_score": 10, "epochs": 100,
         "new_solutions": 0},
        {"L": 44, "mathematical_loss": "none", "projected_best_score": 8,
         "target_best": 32, "post_polish_best_score": 6, "epochs": 120,
         "new_solutions": 1},
    ]
    row = summarize(rows)[0]
    assert row["projected_min"] == 8
    assert row["projected_median"] == 10
    assert row["target_median"] == 36
    assert row["post_polish_median"] == 8
    assert row["epochs_median"] == 110
    assert row["new_solutions"] == 1
