"""Aggregation-only tests; pytest does not execute final-repair benchmarks."""

from experiments.benchmark_final_repair import aggregate


def test_final_repair_aggregation_preserves_unknown_status():
    rows = [
        {"L": 44, "method": "beam", "solved": True, "elapsed": 1.0},
        {"L": 44, "method": "beam", "solved": False, "elapsed": 3.0},
        {"L": 44, "method": "z3", "solved": False, "elapsed": 5.0,
         "status": "UNKNOWN"},
    ]
    result = aggregate(rows)["44"]
    assert result["beam"]["solved"] == 1
    assert result["beam"]["success_rate"] == 0.5
    assert result["beam"]["median_seconds"] == 2.0
    assert result["z3"]["status_counts"]["UNKNOWN"] == 1
