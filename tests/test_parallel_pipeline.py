"""Tests for the conservative spawn-based production orchestrator."""

import json
import math

import pytest

from solver.parallel_pipeline import (
    ParallelPipelineConfig,
    ParallelWorkerError,
    SEED_STRIDE,
    _finalize_worker_messages,
    _merge_new_verified_files,
    _parallel_progress_snapshot,
    derived_worker_seed,
    isolated_worker_root,
    run_parallel_pipeline,
)
from solver.correlation import full_correlation_profile
from solver.objective import pqcp_objective


SOLUTION_A = (0, 0, 0, 0)
SOLUTION_B = (0, 0, 1, 1)


def _raw_success(config, worker_index, a=SOLUTION_A, b=SOLUTION_B):
    profile = full_correlation_profile(a, b)
    return {
        "ok": True,
        "worker_index": worker_index,
        "seed": derived_worker_seed(config.base_seed, worker_index),
        "worker_root": str(isolated_worker_root(config, worker_index)),
        "interrupted": False,
        "elapsed": 0.1,
        "iteration": 10,
        "restart_index": 2,
        "checkpoint_path": str(
            isolated_worker_root(config, worker_index) / "checkpoints" / "L4.json"
        ),
        "best": {
            "A": list(a), "B": list(b), "score": pqcp_objective(profile),
            "profile": profile, "source": "best",
        },
        "verified_candidates": [{
            "A": list(a), "B": list(b), "profile": profile,
            "source": "verified",
        }],
    }


def test_seed_schedule_is_stable_and_worker_roots_are_disjoint(tmp_path):
    config = ParallelPipelineConfig(L=44, workers=3, seconds=1, root=tmp_path)
    assert [derived_worker_seed(123, index) for index in range(3)] == [
        123, 123 + SEED_STRIDE, 123 + 2 * SEED_STRIDE,
    ]
    roots = [isolated_worker_root(config, index) for index in range(3)]
    assert len(set(roots)) == 3
    assert all(root.parent == roots[0].parent for root in roots)
    with pytest.raises(ValueError):
        derived_worker_seed(123, -1)

    infinite = ParallelPipelineConfig(
        L=44, workers=2, seconds=math.inf, root=tmp_path,
    )
    assert math.isinf(infinite.seconds)


def test_parent_reverifies_and_deduplicates_ab_exchange(tmp_path):
    config = ParallelPipelineConfig(
        L=4, workers=2, seconds=1, root=tmp_path,
        raise_on_worker_failure=False,
    )
    result = _finalize_worker_messages(
        config,
        (
            _raw_success(config, 0),
            _raw_success(config, 1, SOLUTION_B, SOLUTION_A),
        ),
        interrupted=False,
        elapsed=0.2,
    )
    assert result.global_best is not None and result.global_best.verified
    assert result.global_best.profile == tuple(
        full_correlation_profile(SOLUTION_A, SOLUTION_B)
    )
    assert result.new_solution_count == 1
    text = (tmp_path / "4.txt").read_text(encoding="utf-8")
    assert text.count("\nL=4\n") == 1
    # Replaying the same discoveries cannot append or corrupt the file.
    repeated = _finalize_worker_messages(
        config, (_raw_success(config, 0),), False, 0.1
    )
    assert repeated.new_solution_count == 0
    assert (tmp_path / "4.txt").read_text(encoding="utf-8") == text


def test_worker_failure_is_reported_after_successful_parent_merge(tmp_path):
    config = ParallelPipelineConfig(
        L=4, workers=2, seconds=1, root=tmp_path,
        raise_on_worker_failure=False,
    )
    failed = {
        "ok": False, "worker_index": 1,
        "seed": derived_worker_seed(config.base_seed, 1),
        "worker_root": str(isolated_worker_root(config, 1)),
        "error": "synthetic failure", "interrupted": False,
    }
    result = _finalize_worker_messages(
        config, (_raw_success(config, 0), failed), False, 0.1
    )
    assert len(result.failures) == 1
    assert result.failures[0].error == "synthetic failure"
    assert result.new_solution_count == 1

    strict = ParallelPipelineConfig(L=4, workers=2, seconds=1, root=tmp_path)
    partial = _finalize_worker_messages(
        strict, (_raw_success(strict, 0), failed), False, 0.1
    )
    with pytest.raises(ParallelWorkerError) as caught:
        if partial.failures:
            raise ParallelWorkerError(partial)
    assert caught.value.result.global_best is not None


def test_parent_rejects_child_score_or_profile_mismatch(tmp_path):
    config = ParallelPipelineConfig(
        L=4, workers=1, seconds=1, root=tmp_path,
        raise_on_worker_failure=False,
    )
    raw = _raw_success(config, 0)
    raw["best"]["score"] += 1
    result = _finalize_worker_messages(config, (raw,), False, 0.1)
    assert result.failures
    assert "parent recomputation" in result.failures[0].error


def test_two_worker_spawn_smoke_preserves_private_checkpoints(tmp_path):
    config = ParallelPipelineConfig(
        L=4,
        workers=2,
        seconds=0.05,
        base_seed=31,
        root=tmp_path,
        gcp=False,
        checkpoint_interval=0.02,
        progress_interval=1.0,
    )
    result = run_parallel_pipeline(config)
    assert result.start_method == "spawn"
    assert len(result.reports) == 2
    assert not result.failures
    assert [report.seed for report in result.reports] == [31, 31 + SEED_STRIDE]
    assert len({report.worker_root for report in result.reports}) == 2
    assert all(
        report.checkpoint_path is not None and report.checkpoint_path.exists()
        for report in result.reports
    )
    assert result.global_best is not None
    assert result.global_best.score == pqcp_objective(result.global_best.profile)
    payload = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert payload["start_method"] == "spawn"
    assert len(payload["workers_results"]) == 2


def test_reference_port_runs_in_isolated_parallel_workers(tmp_path):
    config = ParallelPipelineConfig(
        L=44, workers=2, seconds=0.05, base_seed=73, root=tmp_path,
        gcp=False, reference_port=True,
        checkpoint_interval=0.02, progress_interval=1.0,
    )
    result = run_parallel_pipeline(config)
    assert not result.failures
    assert all(
        report.checkpoint_path.name == "L44_reference.json"
        for report in result.reports
    )
    assert result.global_best is not None


def test_parallel_reference_port_rejects_conflicting_initializer():
    with pytest.raises(ValueError, match="cannot be combined"):
        ParallelPipelineConfig(
            L=44, workers=2, seconds=1,
            reference_port=True, gcp=True,
        )


def test_reference_progress_reads_reference_checkpoint_and_counts_samples(tmp_path):
    config = ParallelPipelineConfig(
        L=44, workers=1, seconds=1, base_seed=73, root=tmp_path,
        gcp=False, reference_port=True,
    )
    root = isolated_worker_root(config, 0)
    checkpoint = root / "checkpoints" / "L44_reference.json"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text(json.dumps({
        "state": {
            "iteration": 25,
            "restart_index": 2,
            "current_score": 12,
            "best_score": 8,
            "elapsed_seconds": 3.5,
            "algorithm_parameters": {"proposal_samples": 2},
        }
    }), encoding="utf-8")
    snapshot = _parallel_progress_snapshot(config)
    assert snapshot["reporting_workers"] == 1
    assert snapshot["restarts"] == 3
    assert snapshot["iterations"] == 25
    assert snapshot["swap_evaluations"] == 50


def test_parent_live_merge_writes_verified_worker_snapshot_once(tmp_path):
    config = ParallelPipelineConfig(L=4, workers=1, seconds=1, root=tmp_path)
    directory = isolated_worker_root(config, 0) / "results" / "verified"
    directory.mkdir(parents=True)
    profile = full_correlation_profile(SOLUTION_A, SOLUTION_B)
    (directory / "solution.json").write_text(json.dumps({
        "L": 4,
        "A": list(SOLUTION_A),
        "B": list(SOLUTION_B),
        "profile": profile,
        "verified": True,
    }), encoding="utf-8")
    processed = set()
    assert _merge_new_verified_files(config, processed) == 1
    assert _merge_new_verified_files(config, processed) == 0
    assert (tmp_path / "4.txt").read_text(encoding="utf-8").count("\nL=4\n") == 1


def test_parallel_worker_can_resume_its_private_checkpoint(tmp_path):
    initial_config = ParallelPipelineConfig(
        L=4, workers=1, seconds=0.03, base_seed=91, root=tmp_path,
        gcp=False, checkpoint_interval=0.01, progress_interval=1.0,
    )
    initial = run_parallel_pipeline(initial_config)
    resumed = run_parallel_pipeline(ParallelPipelineConfig(
        L=4, workers=1, seconds=0.03, base_seed=91, root=tmp_path,
        gcp=False, checkpoint_interval=0.01, progress_interval=1.0,
        resume=True,
    ))
    assert resumed.reports[0].iteration > initial.reports[0].iteration
    assert resumed.reports[0].checkpoint_path == initial.reports[0].checkpoint_path
