"""Background lift matchmaking, bounded memory and official persistence."""

from dataclasses import replace
import json
import threading
from types import SimpleNamespace

import pytest

import main
from solver.compressed_completion import CompressedLiftJob, CompressedLiftPool, run_compressed_lift_job
from solver.compressed_lift import solve_compressed_lift
from solver.correlation import full_correlation_profile
from solver.objective import pqcp_objective
from solver.structured_energy import compress_binary_sequence
from solver.target_profiles import TargetContentProfile, pair_content


A = tuple(map(int, "00000111"))
B = tuple(map(int, "00010011"))
TARGET = TargetContentProfile(8, 2, -1, *pair_content(A, B))


def job():
    return CompressedLiftJob(compress_binary_sequence(A, 2),
                             compress_binary_sequence(B, 2), TARGET, A, B)


def test_pool_exact_match_and_scheduled_deduplication():
    pool = CompressedLiftPool(8)
    matches = pool.offer(A, B, 2, -1)
    assert matches and job().key in {x.key for x in matches}
    for match in matches:
        pool.submit(match)
    assert pool.offer(A, B, 2, -1) == []


def test_pool_unsubmitted_job_is_not_forgotten():
    pool = CompressedLiftPool(8)
    assert pool.offer(A, B, 2, -1)
    assert pool.offer(A, B, 2, -1)  # queue-full omission is not an UNSAT claim


def test_pool_memory_is_bounded():
    pool = CompressedLiftPool(8, capacity=1, history_capacity=1)
    matches = pool.offer(A, B, 2, -1)
    assert all(len(p) <= 1 for p in pool.pools.values())
    assert sum(len(bucket) for bucket in pool.signature_index.values()) <= sum(
        len(p) for p in pool.pools.values())
    first = job()
    pool.submit(first)
    pool.submit(replace(first, compressed_a=first.compressed_a[::-1]))
    assert len(pool.submitted) <= 1


def test_no_match_is_empty_not_global_unsat():
    pool = CompressedLiftPool(8)
    assert pool.offer((0,) * 8, (0,) * 8, 1, -1) == []


def test_sat_persistence_full_profile_components_format_and_dedup(tmp_path):
    result = run_compressed_lift_job(job(), 1_000, tmp_path, 123)
    assert result["status"] == "LIFT_SAT" and result["new_solution_written"]
    snapshot = json.loads(open(result["snapshot"], encoding="utf-8").read())
    assert snapshot["verified"] and snapshot["score"] == 0
    assert snapshot["profile"] == full_correlation_profile(snapshot["A"], snapshot["B"])
    assert snapshot["objective_components"]["total"] == 0
    text = (tmp_path / "8.txt").read_text()
    assert "L=8\nnonzero shifts=2,6\nnonzero PACS=-4\na=" in text
    verified_paths = list((tmp_path / "results/verified").glob("*.json"))
    assert len(verified_paths) == 1
    original_verified = verified_paths[0].read_bytes()
    duplicate = run_compressed_lift_job(job(), 1_000, tmp_path, 123)
    assert verified_paths[0].read_bytes() == original_verified
    assert not list((tmp_path / "results/compressed_lift").glob("*.tmp"))
    # Different models may be returned by Z3; replay this exact saved pair
    # through the existing writer to explicitly check A/B exchange.
    from solver.search_runner import append_verified_solution_if_new
    assert not append_verified_solution_if_new(8, snapshot["B"], snapshot["A"], tmp_path)
    assert duplicate["solved"]


def test_unknown_is_persisted_and_never_written_as_solution(tmp_path, monkeypatch):
    sat = solve_compressed_lift(job().compressed_a, job().compressed_b, 2, -1, 1_000, TARGET)
    unknown = replace(sat, status="UNKNOWN", a=None, b=None, profile=None,
                      verified=False, reason="test timeout")
    monkeypatch.setattr("solver.compressed_lift.solve_compressed_lift", lambda *_a, **_k: unknown)
    result = run_compressed_lift_job(job(), 1_000, tmp_path, 123)
    assert result["status"] == "LIFT_UNKNOWN" and not result["solved"]
    assert not (tmp_path / "8.txt").exists()
    assert json.loads(open(result["snapshot"]).read())["reason"] == "test timeout"


def test_wrong_target_sat_never_written(tmp_path, monkeypatch):
    wrong = solve_compressed_lift((2, 2), (0, 0), 1, 1, 1_000)
    monkeypatch.setattr("solver.compressed_lift.solve_compressed_lift", lambda *_a, **_k: wrong)
    with pytest.raises(RuntimeError, match="unverified"):
        run_compressed_lift_job(job(), 1_000, tmp_path, 123)
    assert not (tmp_path / "8.txt").exists()


def test_bridge_bypasses_score_and_hamming_guards_for_exact_compression(tmp_path):
    args = main.parse_args(["--L", "8", "--z3", "--z3-trigger-score", "0",
                            "--z3-profile-radius", "0", "--z3-timeout", "1",
                            "--no-z3-staged"])
    bridge = main._CCompletionBridge(args, 8, tmp_path)
    score = pqcp_objective(full_correlation_profile(A, B))
    assert score > 0
    try:
        bridge.consider(A, B, {"k": 2, "sign": -1, "score": score})
    finally:
        bridge.close()
    assert bridge.trigger_count >= 1
    assert bridge.z3_execution_count >= 1
    assert bridge._results[0]["status"] == "LIFT_SAT"
    assert (tmp_path / "8.txt").exists()


def test_lift_jobs_are_not_debounced_away(tmp_path, monkeypatch):
    calls = []
    done = threading.Event()
    def fake(j, *_args):
        calls.append(j)
        return {"status": "LIFT_UNKNOWN", "solved": False, "z3_executed": True,
                "orientation_variables": 4, "construction_time": 0, "solve_time": 0}
    monkeypatch.setattr("solver.compressed_completion.run_compressed_lift_job", fake)
    args = main.parse_args(["--L", "8", "--z3"])
    def callback(_result):
        if len(calls) == 3: done.set()
    bridge = main._CCompletionBridge(args, 8, tmp_path, callback)
    try:
        for index in range(3):
            bridge._queue.put({"kind": "compressed_lift", "job": index})
        assert done.wait(5)
    finally:
        bridge.close()
    assert calls == [0, 1, 2]


def test_rollback_flag_and_z3_disabled():
    args = main.parse_args(["--L", "44", "--z3", "--no-z3-compressed-lift"])
    assert not args.z3_compressed_lift
    disabled = main._CCompletionBridge(SimpleNamespace(repair=False, z3=False), 44, None)
    assert disabled._lift_pool is None and disabled._thread is None


def test_rollback_still_calls_original_radius_completion(tmp_path, monkeypatch):
    calls = []
    def fallback(*args):
        calls.append(args)
        return {"status": "Z3_UNKNOWN", "solved": False, "z3_executed": True}
    monkeypatch.setattr(main, "_run_optional_z3", fallback)
    args = main.parse_args(["--L", "8", "--z3", "--no-z3-compressed-lift",
                            "--z3-profile-radius", "10", "--z3-trigger-score", "100"])
    bridge = main._CCompletionBridge(args, 8, tmp_path)
    try:
        bridge.consider(A, B, {"k": 2, "sign": -1,
                               "score": pqcp_objective(full_correlation_profile(A, B))})
    finally:
        bridge.close()
    assert len(calls) == 1 and bridge.z3_execution_count == 1
    assert not (tmp_path / "results/compressed_lift").exists()


@pytest.mark.parametrize("kwargs", ({"capacity": 0}, {"history_capacity": True}))
def test_invalid_pool_capacity(kwargs):
    with pytest.raises(ValueError): CompressedLiftPool(8, **kwargs)
