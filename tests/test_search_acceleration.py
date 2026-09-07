"""Correctness gates for the optimized exact search hot path."""

from experiments.benchmark_search_acceleration import benchmark, run_to_first_solution
from solver.checkpoint import SearchParameters
from solver.compression import CorrelationState
from solver.objective import pqcp_objective
from solver.search_runner import SearchRunner


def test_incremental_score_equals_full_objective_over_long_l44_trajectory():
    runner = SearchRunner.new(44, 321, SearchParameters(stagnation_iterations=None))
    for _ in range(5000):
        runner.step()
        assert runner._correlation.score == pqcp_objective(runner._correlation.profile)


def test_reference_and_optimized_find_identical_first_verified_pair():
    reference = run_to_first_solution(20, 100, 5000, True)
    optimized = run_to_first_solution(20, 100, 5000, False)
    assert optimized == reference


def test_benchmark_records_one_paired_timing_per_seed_and_repeat():
    result = benchmark(20, (100,), repeats=2, max_steps=5000)
    assert len(result["reference_times"]) == len(result["optimized_times"]) == 2
    assert result["reference_mean"] > 0 and result["optimized_mean"] > 0
