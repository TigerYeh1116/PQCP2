"""Lightweight aggregation tests; no timed search runs under pytest."""

from experiments.benchmark_reference_port import aggregate, method_parameters


def test_ablation_changes_one_or_the_complete_reference_parameter_set():
    legacy = method_parameters(44, "legacy")
    compressed = method_parameters(44, "compressed_seed")
    reference_seed = method_parameters(44, "reference_seed")
    multiscale = method_parameters(44, "multiscale")
    reference_full = method_parameters(44, "reference_full")
    reference = method_parameters(44, "reference")
    assert compressed.fkm_seed_policy == "compressed_top_q"
    assert compressed.acceptance_mode == legacy.acceptance_mode
    assert reference_seed.fkm_seed_policy == "compressed_a_random_b_top_q"
    assert reference_seed.acceptance_mode == legacy.acceptance_mode
    assert multiscale.fkm_seed_policy == legacy.fkm_seed_policy
    assert multiscale.acceptance_mode == "fixed_target_multiscale"
    assert reference_full.fkm_seed_policy == "compressed_a_random_b_top_q"
    assert reference_full.acceptance_mode == "fixed_target_full"
    assert reference.fkm_seed_policy == "compressed_a_random_b_top_q"
    assert reference.acceptance_mode == "fixed_target_multiscale"


def test_aggregate_scores_and_paired_directions():
    records = [
        {"L": 44, "method": "legacy", "seed": 1, "best_score": 10,
         "verified": False, "iterations_per_second": 100, "initialization_seconds": .1},
        {"L": 44, "method": "legacy", "seed": 2, "best_score": 8,
         "verified": False, "iterations_per_second": 100, "initialization_seconds": .1},
        {"L": 44, "method": "reference", "seed": 1, "best_score": 7,
         "verified": False, "iterations_per_second": 80, "initialization_seconds": .2},
        {"L": 44, "method": "reference", "seed": 2, "best_score": 9,
         "verified": False, "iterations_per_second": 80, "initialization_seconds": .2},
    ]
    result = aggregate(records)
    paired = next(row for row in result["paired_vs_legacy"] if row["method"] == "reference")
    assert paired == {"method": "reference", "wins": 1, "ties": 0, "losses": 1}
    reference = next(row for row in result["summaries"] if row["method"] == "reference")
    assert reference["median"] == 8
    assert reference["mean"] == 8
