"""Lightweight validation and aggregation tests for compressed-search ablation."""

from collections import Counter

import pytest

import experiments.benchmark_compressed_search as benchmark_module
from experiments.benchmark_compressed_search import (
    BenchmarkConfig,
    ENERGY_MODES,
    SEED_POLICIES,
    THRESHOLDS,
    counterbalanced_order,
    energy_key,
    initialize_runner,
    paired_initialization_audit,
    paired_wtl,
    parameters_for,
    run_case,
    summarize,
    threshold_summary,
)
from solver.correlation import full_correlation_profile
from solver.search_runner import SearchRunner
from solver.structured_energy import structured_energy_breakdown
from solver.target_profiles import canonical_target_content_profiles, pair_content
from solver.verifier import verify_pqcp


def _record(length, seed, energy, policy, score, hit8=None, elapsed=10.0):
    thresholds = {}
    for threshold in THRESHOLDS:
        hit = score <= threshold
        if threshold == 8 and hit8 is not None:
            hit = hit8
        thresholds[str(threshold)] = {
            "hit": hit,
            "time": 2.0 if hit else None,
            "iteration": 10 if hit else None,
            "evaluations": 100 if hit else None,
        }
    return {
        "L": length,
        "seed": seed,
        "energy_mode": energy,
        "seed_policy": policy,
        "initial_score": 100,
        "initial_full_energy": 200,
        "initialization_elapsed": 0.5,
        "best_score": score,
        "elapsed": elapsed,
        "iterations": 50,
        "proposal_evaluations": 500,
        "evaluations_per_second": 50.0,
        "verified": score == 0,
        "thresholds": thresholds,
    }


def test_config_requires_exactly_one_budget_and_valid_elite_count():
    with pytest.raises(ValueError, match="exactly one"):
        BenchmarkConfig((44,), (1,), iterations=None, seconds_per_run=None)
    with pytest.raises(ValueError, match="exactly one"):
        BenchmarkConfig((44,), (1,), iterations=10, seconds_per_run=1.0)
    with pytest.raises(ValueError, match="elite_count"):
        BenchmarkConfig((44,), (1,), iterations=10, candidate_count=4, elite_count=5)
    with pytest.raises(ValueError, match="stagnation_iterations"):
        BenchmarkConfig((44,), (1,), iterations=10, stagnation_iterations=0)
    config = BenchmarkConfig((44, 46), (1, 2), seconds_per_run=1.0)
    assert config.lengths == (44, 46)


def test_config_rejects_unpaired_duplicate_seeds_and_unknown_modes():
    with pytest.raises(ValueError, match="unique"):
        BenchmarkConfig((44,), (1, 1), iterations=10)
    with pytest.raises(ValueError, match="energy_modes"):
        BenchmarkConfig((44,), (1,), iterations=10, energy_modes=("unknown",))
    with pytest.raises(ValueError, match="seed_policies"):
        BenchmarkConfig((44,), (1,), iterations=10, seed_policies=("unknown",))


def test_paired_wtl_uses_common_length_seed_cells():
    rows = [
        _record(44, 1, "full", "legacy", 12),
        _record(44, 2, "full", "legacy", 8),
        _record(44, 3, "full", "legacy", 12),
        _record(44, 1, "full_e2", "legacy", 8),
        _record(44, 2, "full_e2", "legacy", 8),
        _record(44, 3, "full_e2", "legacy", 16),
    ]
    assert paired_wtl(
        rows, "full", "legacy", "full_e2", "legacy"
    ) == {"wins": 1, "ties": 1, "losses": 1, "pairs": 3}


def test_threshold_summary_counts_misses_as_censored_budget():
    rows = [
        _record(44, 1, "full", "legacy", 8, hit8=True, elapsed=10.0),
        _record(44, 2, "full", "legacy", 12, hit8=False, elapsed=10.0),
    ]
    result = threshold_summary(rows, 8)
    assert result["hits"] == 1 and result["hit_rate"] == 0.5
    assert result["restricted_mean_time"] == 6.0
    assert result["restricted_mean_evaluations"] == 300


def test_summary_reports_distribution_thresholds_and_two_paired_axes():
    rows = [
        _record(44, 1, "full", "legacy", 12),
        _record(44, 2, "full", "legacy", 8),
        _record(44, 1, "full", "phase_top_q", 8),
        _record(44, 2, "full", "phase_top_q", 12),
        _record(44, 1, "full_e2", "legacy", 8),
        _record(44, 2, "full_e2", "legacy", 8),
    ]
    summary = summarize(rows)
    indexed = {
        (row["energy_mode"], row["seed_policy"]): row for row in summary
    }
    full = indexed[("full", "legacy")]
    assert (full["minimum"], full["median"], full["mean"], full["maximum"]) == (
        8, 10.0, 10, 12,
    )
    assert full["thresholds"]["8"]["hits"] == 1
    assert indexed[("full_e2", "legacy")]["paired_energy_vs_full"] == {
        "wins": 1, "ties": 1, "losses": 0, "pairs": 2,
    }
    assert indexed[("full", "phase_top_q")]["paired_policy_vs_legacy"] == {
        "wins": 1, "ties": 0, "losses": 1, "pairs": 2,
    }


def test_paired_initialization_audit_detects_unfair_energy_start():
    rows = [
        {
            "L": 44, "seed": 1, "seed_policy": "legacy",
            "initial_A": [0, 1], "initial_B": [1, 0],
        },
        {
            "L": 44, "seed": 1, "seed_policy": "legacy",
            "initial_A": [0, 1], "initial_B": [1, 0],
        },
    ]
    assert paired_initialization_audit(rows)["mismatches"] == 0
    rows[1]["initial_B"] = [0, 1]
    with pytest.raises(AssertionError, match="different initial"):
        paired_initialization_audit(rows)


@pytest.mark.parametrize("energy_mode", ENERGY_MODES)
@pytest.mark.parametrize("seed_policy", SEED_POLICIES)
def test_parameters_map_every_factorial_cell_to_production_controls(
    energy_mode, seed_policy,
):
    expected_mode = {
        "full": "fixed_target_full",
        "compressed_tiebreak": "fixed_target_full_compressed_tiebreak",
        "full_e2": "fixed_target_full_e2",
        "multiscale": "fixed_target_multiscale",
    }[energy_mode]
    parameters = parameters_for(
        44, energy_mode, seed_policy, 10, 12, 3, 32,
        target_profile_index=2, stagnation_iterations=777,
    )
    assert parameters.acceptance_mode == expected_mode
    assert parameters.fkm_seed_policy == seed_policy
    assert parameters.fkm_candidate_count == 12
    assert parameters.fkm_elite_count == 3
    assert parameters.proposal_samples == 10
    assert parameters.stagnation_iterations == 777
    expected = canonical_target_content_profiles(44)[2]
    assert parameters.target_content_profiles[0] == (
        expected.k, expected.eta, expected.a_even_ones,
        expected.a_odd_ones, expected.b_even_ones, expected.b_odd_ones,
    )


@pytest.mark.parametrize("length", (44, 46))
@pytest.mark.parametrize("mode", ENERGY_MODES)
def test_energy_key_matches_exact_structured_components(length, mode):
    target = canonical_target_content_profiles(length)[0]
    runner = initialize_runner(
        length, 901, mode, "phase_top_q", 3, 8, 3, 24,
    )
    profile = full_correlation_profile(
        runner.state.current_a, runner.state.current_b
    )
    factors = (2, 4) if length % 4 == 0 else (2,)
    breakdown = structured_energy_breakdown(
        profile, target.k, target.eta, factors
    )
    if mode == "full":
        expected = (breakdown.full,)
    elif mode == "compressed_tiebreak":
        expected = (breakdown.full,) + tuple(
            breakdown.component(factor) for factor in factors
        )
    elif mode == "full_e2":
        expected = (breakdown.full + breakdown.component(2),)
    else:
        expected = (breakdown.full + 2 * breakdown.component(2)
                    + (4 * breakdown.component(4) if length % 4 == 0 else 0),)
    assert energy_key(profile, target, mode) == expected


@pytest.mark.parametrize("policy", SEED_POLICIES)
def test_energy_modes_receive_identical_production_initial_pair(policy):
    pairs = []
    for mode in ENERGY_MODES:
        runner = initialize_runner(44, 314, mode, policy, 2, 8, 3, 24)
        pairs.append((runner.state.current_a, runner.state.current_b))
    assert len(set(pairs)) == 1


def test_run_case_recomputes_profile_score_and_verifier_exactly():
    config = BenchmarkConfig(
        (8,), (71,), iterations=5,
        energy_modes=("compressed_tiebreak",),
        seed_policies=("phase_top_q",),
        proposal_samples=3, candidate_count=6, elite_count=2,
        fkm_pool_size=8,
    )
    row = run_case(
        config, 8, 71, "compressed_tiebreak", "phase_top_q",
        target_profile_index=1,
    )
    assert tuple(full_correlation_profile(row["best_A"], row["best_B"])) == tuple(
        row["best_profile"]
    )
    assert verify_pqcp(row["best_A"], row["best_B"]).is_valid == row["verified"]
    assert row["sampled_proposals"] == row["iterations"] * 3
    assert row["target_profile_index"] == 1


def test_initial_score_zero_is_verified_without_taking_a_step(monkeypatch):
    parameters = parameters_for(4, "full", "legacy", 1, 4, 1, 4)
    solution = ((0, 0, 0, 0), (0, 0, 1, 1))
    assert verify_pqcp(*solution).is_valid
    runner = SearchRunner.from_candidate(*solution, seed=5, parameters=parameters)
    monkeypatch.setattr(
        benchmark_module, "initialize_runner", lambda *args, **kwargs: runner
    )
    config = BenchmarkConfig(
        (4,), (5,), iterations=10,
        energy_modes=("full",), seed_policies=("legacy",),
        fkm_pool_size=4,
    )
    row = run_case(config, 4, 5, "full", "legacy")
    assert row["verified"] is True
    assert row["best_score"] == 0
    assert row["iterations"] == 0


def test_restart_keeps_seed_policy_and_exact_next_profile_content():
    runner = initialize_runner(
        44, 123, "compressed_tiebreak", "phase_top_q",
        2, 8, 3, 24,
    )
    runner.restart_after_completion_miss()
    assert runner.state.restart_index == 1
    assert runner.state.algorithm_parameters.fkm_seed_policy == "phase_top_q"
    target = runner._current_target_content_profile()
    assert target is not None
    assert pair_content(runner.state.current_a, runner.state.current_b) == (
        target.a_even_ones, target.a_odd_ones,
        target.b_even_ones, target.b_odd_ones,
    )


def test_counterbalance_covers_each_cell_at_every_position_twice():
    cells = tuple((mode, "legacy") for mode in ENERGY_MODES)
    position_counts = Counter()
    for design_index in range(2 * len(cells)):
        order = counterbalanced_order(cells, design_index)
        assert set(order) == set(cells)
        for position, cell in enumerate(order):
            position_counts[(cell, position)] += 1
    assert set(position_counts.values()) == {2}
