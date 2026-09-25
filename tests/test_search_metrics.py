"""Coarse observation counters never call a solver or change RNG state."""

import json
import random

import pytest

from solver.search_metrics import SearchMetrics


def test_unique_rate_retries_times_and_atomic_snapshot(tmp_path):
    now = [0.0]
    metrics = SearchMetrics(tmp_path, 86, 123, clock=lambda: now[0],
                            parameters={"polish_mode": "alternating"})
    rng = random.getstate()
    for key in ("same", "same", "different"):
        metrics.matched(key)
    metrics.backlog(2, 4)
    for _ in range(2):
        metrics.result(dict(status="UNKNOWN", z3_calls=1, construction_time=.2, solve_time=.4,
                            gf2_applied=True, gf2_rank=12, gf2_free_bits=20, gf2_preprocessing_time=.01))
    metrics.stage("worker_waiting", 3)
    assert not metrics.path.exists()
    now[0] = 60
    metrics.flush()
    first = json.loads(metrics.path.read_text())
    assert first["parameters"]["polish_mode"] == "alternating"
    assert first["unique_compressed_matches"] == 2
    assert first["unique_matches_per_minute"] == 2
    assert first["gf2_rank_turn_histogram"] == {"12": 2}
    assert first["counts"]["z3_calls"] == 2
    assert first["seconds"]["solve_time"] == .8
    assert first["peak_queued_backlog"] == 6
    metrics.backlog(0)
    metrics.flush(force=True)
    assert json.loads(metrics.path.read_text())["queued_backlog"] == 0
    assert random.getstate() == rng


def test_disabled_no_disk_and_failed_write_is_nonfatal(tmp_path, monkeypatch):
    off = SearchMetrics(tmp_path, 86, 123, enabled=False)
    off.stage("work", 1)
    off.flush(force=True)
    assert not off.path.exists()
    active = SearchMetrics(tmp_path, 86, 123)
    def fail(*args):
        raise OSError("disk unavailable")
    monkeypatch.setattr("solver.search_metrics.atomic_write_json", fail)
    with pytest.warns(UserWarning, match="disabled"):
        active.flush(force=True)
    assert not active.enabled


def test_cli_defaults_and_rollback():
    from main import parse_args
    args = parse_args(["--L", "86"])
    assert args.torch_polish_mode == "baseline" and args.search_metrics
    args = parse_args(["--L", "86", "--torch-polish-mode", "alternating", "--no-search-metrics"])
    assert args.torch_polish_phase_steps == 32 and not args.search_metrics


def test_invalid_mode_length_does_not_start_completion(monkeypatch):
    import main
    def forbidden(*args, **kwargs):
        pytest.fail("must validate before starting a background solver")
    monkeypatch.setattr(main, "_CCompletionBridge", forbidden)
    assert main.main(["--L", "44", "--torch-polish-mode", "alternating"]) != 0
