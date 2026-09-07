"""Lightweight aggregation tests for guided Z3 performance measurement."""

from experiments.benchmark_guided_z3 import aggregate


def test_threefold_speed_is_literal_two_hundred_percent_increase():
    records = [
        {"method": "broad", "elapsed": 3.0},
        {"method": "guided", "elapsed": 1.0},
        {"method": "broad", "elapsed": 6.0},
        {"method": "guided", "elapsed": 2.0},
    ]
    result = aggregate(records)
    assert result["speed_factor"] == 3.0
    assert result["speed_increase_percent"] == 200.0
