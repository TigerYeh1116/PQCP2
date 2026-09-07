"""Lightweight aggregation test for the Z3 removal experiment."""

from experiments.benchmark_z3_removal import method_order, summarize


def test_z3_removal_summary_keeps_methods_separate():
    rows = [
        {"L": 44, "method": "no_z3", "best_score": 8, "iterations": 100,
         "wall": 1.0, "completion_attempts": 0, "exact_z3_executed": False,
         "wall_overrun": 0.0, "verified": False},
        {"L": 44, "method": "z3", "best_score": 12, "iterations": 20,
         "wall": 1.0, "completion_attempts": 1, "exact_z3_executed": True,
         "wall_overrun": 0.0, "verified": False},
    ]
    result = summarize(rows)
    assert result[0]["method"] == "no_z3"
    assert result[0]["mean_iterations_per_second"] == 100
    assert result[1]["completion_attempts"] == 1
    assert result[1]["exact_z3_executions"] == 1


def test_method_order_is_deterministic_and_counterbalanced():
    orders = [method_order(44, seed) for seed in (0, 1, 2)]
    assert len(set(orders)) == 3
    assert all(set(order) == {"sa_only", "repair_only", "z3"} for order in orders)
