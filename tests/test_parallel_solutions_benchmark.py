"""Lightweight tests for verified parallel time-to-solution aggregation."""

import pytest

from experiments.benchmark_parallel_solutions import (
    encoded_profiles,
    paired_speed_interval,
    summarize,
)


def test_encoded_profiles_are_available_for_small_exact_benchmark():
    profiles = encoded_profiles(28)
    assert profiles
    assert all(len(profile) == 6 for profile in profiles)


def test_solution_summary_uses_censored_time_conservatively():
    rows = [
        {"L": 28, "workers": 1, "solved": True,
         "time_to_solution": 0.5, "censored_time": 0.5},
        {"L": 28, "workers": 1, "solved": False,
         "time_to_solution": None, "censored_time": 2.0},
        {"L": 28, "workers": 8, "solved": True,
         "time_to_solution": 0.1, "censored_time": 0.1},
        {"L": 28, "workers": 8, "solved": True,
         "time_to_solution": 0.15, "censored_time": 0.15},
    ]
    result = summarize(rows)
    assert result[0]["verified_solutions"] == 1
    assert result[0]["restricted_mean_time"] == 1.25
    assert result[1]["verified_solutions"] == 2
    assert result[1]["speed_factor_vs_one_worker"] == 10.0


def test_paired_speed_interval_resamples_complete_trials():
    rows = []
    for trial, baseline, optimized in ((0, 1.0, 0.2), (1, 2.0, 0.4)):
        rows.extend((
            {"trial": trial, "workers": 1, "censored_time": baseline},
            {"trial": trial, "workers": 8, "censored_time": optimized},
        ))
    result = paired_speed_interval(rows, samples=100, seed=3)
    assert result["paired_trials"] == 2
    assert result["speed_factor"] == pytest.approx(5.0)
    assert result["lower"] == pytest.approx(5.0)
    assert result["upper"] == pytest.approx(5.0)
