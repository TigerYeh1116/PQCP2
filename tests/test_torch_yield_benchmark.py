"""No long benchmarks in pytest: test only accounting and configuration."""

from dataclasses import asdict

from experiments.benchmark_torch_yield import summarize, variant_config
from solver.torch_hybrid_runner import TorchHybridConfig


def test_censoring_is_not_silently_dropped():
    rows = [dict(L=44, variant="current", first_new_seconds=None, new_pairs=0, elapsed=10),
            dict(L=44, variant="current", first_new_seconds=2, new_pairs=2, elapsed=10)]
    row = summarize(rows)[0]
    assert row["mean_first_new"] is None
    assert row["successful_runs"] == 1
    assert row["seconds_per_new"] == 10
    assert row["new_per_second"] == 0.1


def test_variant_changes_are_explicit_and_keep_other_controls():
    base = TorchHybridConfig(44)
    old = variant_config(base, "current")
    fast = variant_config(base, "optimized")
    assert old.continuous_loss_backend == "profile" and not old.fast_observation
    assert old.initial_formation_seconds is None
    assert fast.continuous_loss_backend == "spectral" and fast.fast_observation
    assert fast.initial_formation_seconds == 8
    allowed = {"continuous_loss_backend", "fast_observation", "initial_formation_seconds"}
    assert {k for k, v in asdict(old).items() if v != asdict(fast)[k]} == allowed


def test_cli_keeps_new_paths_reversible():
    import main
    args = main.parse_args(["--L", "44", "--torch-fast-observation",
                            "--torch-observation-backend", "torch",
                            "--cuda-initial-seconds", "8", "--cuda-pair-seed-percent", "50"])
    assert args.torch_fast_observation and args.torch_observation_backend == "torch"
    assert args.cuda_initial_seconds == 8 and args.cuda_pair_seed_percent == 50
    assert args.torch_loss_backend == "profile"


def test_production_enables_only_proven_profile_reductions_by_default():
    import main

    config = TorchHybridConfig(68)
    assert config.continuous_frequency_pruning
    assert config.continuous_symmetry_pruning
    assert config.continuous_half_shift_seeding
    assert config.continuous_compression_pairing
    args = main.parse_args(["--L", "68"])
    assert (args.torch_frequency_pruning and args.torch_symmetry_pruning
            and args.torch_half_shift_seeding
            and args.torch_compression_pairing)
    rollback = main.parse_args([
        "--L", "68", "--no-torch-frequency-pruning",
        "--no-torch-symmetry-pruning", "--no-torch-half-shift-seeding",
        "--no-torch-compression-pairing",
    ])
    assert not rollback.torch_frequency_pruning
    assert not rollback.torch_symmetry_pruning
    assert not rollback.torch_half_shift_seeding
    assert not rollback.torch_compression_pairing


def test_mixed_variant_preserves_gpu_work_and_original_neighborhood():
    base = TorchHybridConfig(44)
    metal = variant_config(base, "metal")
    mixed = variant_config(base, "mixed")
    changed = {k for k, v in asdict(metal).items() if v != asdict(mixed)[k]}
    assert changed == {"completion_pair_seed_percent"}
    assert mixed.completion_pair_seed_percent == 50
    assert mixed.observation_backend == "torch"
    assert mixed.continuous_loss_backend == "profile"


def test_feasibility_variant_changes_only_continuous_iteration_from_mixed():
    import main
    base = TorchHybridConfig(44)
    mixed = asdict(variant_config(base, "mixed"))
    feasibility = asdict(variant_config(base, "feasibility"))
    assert {key for key in mixed if mixed[key] != feasibility[key]} == {
        "continuous_optimization_mode"
    }
    assert feasibility["continuous_optimization_mode"] == "douglas_rachford"
    args = main.parse_args(["--L", "44", "--torch-optimization-mode", "douglas_rachford"])
    assert args.torch_optimization_mode == "douglas_rachford"


def test_extended_feasibility_does_not_change_projection_or_cpu_search():
    base = TorchHybridConfig(44)
    short = asdict(variant_config(base, "feasibility"))
    longer = asdict(variant_config(base, "feasibility_long"))
    assert {key for key in short if short[key] != longer[key]} == {
        "continuous_steps_per_restart"
    }
    assert longer["continuous_steps_per_restart"] == 20000


def test_wide_only_increases_continuous_parallel_lanes():
    base = TorchHybridConfig(44)
    mixed = asdict(variant_config(base, "mixed"))
    wide = asdict(variant_config(base, "wide"))
    assert {key for key in mixed if mixed[key] != wide[key]} == {
        "continuous_batch_size"
    }
    assert wide["continuous_batch_size"] == 8192


def test_frequency_variant_only_adds_the_proven_content_filter_to_mixed():
    import main
    base = TorchHybridConfig(44)
    mixed = asdict(variant_config(base, "mixed"))
    filtered = asdict(variant_config(base, "frequency"))
    assert {key for key in mixed if mixed[key] != filtered[key]} == {
        "continuous_frequency_pruning"
    }
    assert filtered["continuous_frequency_pruning"]
    assert main.parse_args(["--L", "44", "--torch-frequency-pruning"]).torch_frequency_pruning


def test_new_math_ablation_changes_only_the_requested_loss_layer():
    import main
    base = TorchHybridConfig(44)
    current = asdict(variant_config(base, "current"))
    expected = {
        "math_psd": "psd_cap", "math_divisor": "divisor_lift",
        "math_lattice": "lattice", "math_variance": "variance", "math_combined": "combined",
        "math_lattice_bootstrap": "lattice_bootstrap",
    }
    for variant, mode in expected.items():
        candidate = asdict(variant_config(base, variant))
        assert {key for key in current if current[key] != candidate[key]} == {
            "continuous_mathematical_loss"
        }
        assert candidate["continuous_mathematical_loss"] == mode
    args = main.parse_args(["--L", "44", "--torch-mathematical-loss", "combined"])
    assert args.torch_mathematical_loss == "combined"
