"""Short deterministic checks for durable, conservative Z3 scheduling."""

from dataclasses import asdict, replace
import json
import threading
from types import SimpleNamespace

import pytest

import main
from solver import staged_completion as staged
from solver.compressed_completion import CompressedLiftJob
from solver.compressed_lift import solve_compressed_lift
from solver.correlation import full_correlation_profile
from solver.structured_energy import compress_binary_sequence
from solver.target_profiles import TargetContentProfile, pair_content
from solver.verifier import verify_pqcp


A = tuple(map(int, "00000111"))
B = tuple(map(int, "00010011"))
TARGET = TargetContentProfile(8, 2, -1, *pair_content(A, B))


def job(factor=4):
    return CompressedLiftJob(compress_binary_sequence(A, factor),
                             compress_binary_sequence(B, factor), TARGET, A, B, factor)


def other_job():
    """Different ordered compressed pair, both sides with the same content."""
    original = job(2)
    return replace(original, compressed_a=original.compressed_b,
                   compressed_b=original.compressed_a, source_a=B, source_b=A)


def store_job(tmp_path, factor=4, deduplicate_equivalents=True, **policy):
    store = staged.CompletionWorkStore(tmp_path, 8, 123, staged.SlicePolicy(**policy),
                                       deduplicate_equivalents=deduplicate_equivalents)
    key = store.register("lift", asdict(job(factor)))
    return store, key


def unknown_lift():
    candidate = job(2)
    result = solve_compressed_lift(candidate.compressed_a, candidate.compressed_b,
                                   2, -1, 2000, TARGET)
    assert result.verified
    return replace(result, status="UNKNOWN", a=None, b=None, profile=None,
                   verified=False, reason="controlled timeout")


def test_multilevel_each_turn_has_at_most_one_check_and_verified_output(tmp_path):
    store, key = store_job(tmp_path)
    reports = []
    for _ in range(40):
        report = store.advance_lift(key)
        reports.append(report)
        assert report["z3_calls"] <= 1
        if report["phase"] != "READY":
            break
    assert reports[0]["work_kind"] == "enum"
    assert reports[0]["phase"] == "READY"  # middle SAT is NOT a PQCP
    assert reports[-1]["status"] == "LIFT_SAT"
    snapshot = json.loads(open(reports[-1]["snapshot"], encoding="utf-8").read())
    assert verify_pqcp(snapshot["A"], snapshot["B"]).is_valid
    assert snapshot["profile"] == full_correlation_profile(snapshot["A"], snapshot["B"])
    assert (tmp_path / "8.txt").exists()
    assert store.recover() == []  # completed work is not solved again
    assert store.register("lift", asdict(job())) is None


def test_unknown_doubles_parks_and_recovers_without_losing_candidate(tmp_path, monkeypatch):
    unknown = unknown_lift()
    budgets = []

    def fake(*args, **kwargs):
        budgets.append(args[4])
        return unknown

    monkeypatch.setattr(staged, "solve_compressed_lift", fake)
    store, key = store_job(tmp_path, factor=2, initial_ms=10, maximum_ms=40)
    for phase in ("READY", "READY", "WAITING"):
        result = store.advance_lift(key)
        assert result["status"] == "LIFT_UNKNOWN" and result["phase"] == phase
    assert budgets == [10, 20, 40]
    assert not (tmp_path / "8.txt").exists()
    recovered = staged.CompletionWorkStore(tmp_path, 8, 123, store.policy)
    assert recovered.recover() == [key]
    record = recovered.read(key)
    assert tuple(record["spec"]["source_a"]) == A
    assert record["state"]["binary"]["attempts"] == 3
    recovered.advance_lift(key)
    assert budgets == [10, 20, 40, 40]
    assert recovered.read(key)["phase"] == "WAITING"


def test_model_cap_can_resume_after_raising_limit(tmp_path, monkeypatch):
    unknown = unknown_lift()
    monkeypatch.setattr(staged, "solve_compressed_lift", lambda *_a, **_k: unknown)
    store, key = store_job(tmp_path, max_middle_models=1, maximum_ms=1000)
    store.advance_lift(key)  # enum -> first child
    first = store.read(key)["state"]["children"][0]
    store.advance_lift(key)  # timeout -> parked child
    final = store.advance_lift(key)  # cap -> parked enum, NOT UNSAT
    assert final["phase"] == "WAITING" and final["status"] == "LIFT_UNKNOWN"
    assert store.read(key)["state"]["enum"]["status"] == "LIMIT"
    larger = staged.CompletionWorkStore(tmp_path, 8, 123,
                                        staged.SlicePolicy(max_middle_models=2))
    assert larger.recover() == [key]
    record = larger.read(key)
    assert record["state"]["order"][0] == "enum"
    larger.advance_lift(key)
    children = larger.read(key)["state"]["children"]
    assert len(children) == 2
    assert (children[0]["a"], children[0]["b"]) == (first["a"], first["b"])
    assert (children[1]["a"], children[1]["b"]) != (first["a"], first["b"])


def test_exhausted_enum_with_unknown_children_is_not_unsat(tmp_path, monkeypatch):
    unknown = unknown_lift()
    monkeypatch.setattr(staged, "solve_compressed_lift", lambda *_a, **_k: unknown)
    store, key = store_job(tmp_path, max_middle_models=100, maximum_ms=1000)
    for _ in range(210):
        result = store.advance_lift(key)
        if result["phase"] != "READY":
            break
    state = store.read(key)["state"]
    assert state["enum"]["status"] == "UNSAT"
    assert state["children"] and all(c["status"] == "WAITING" for c in state["children"])
    assert result["phase"] == "WAITING" and result["status"] == "LIFT_UNKNOWN"


def test_new_middle_gets_short_attempt_before_retrying_old_child(tmp_path, monkeypatch):
    unknown = unknown_lift()
    monkeypatch.setattr(staged, "solve_compressed_lift", lambda *_a, **_k: unknown)
    store, key = store_job(tmp_path, initial_ms=1000, maximum_ms=4000)
    kinds = [store.advance_lift(key)["work_kind"] for _ in range(5)]
    assert kinds == ["enum", "child_0", "enum", "child_1", "child_0"]
    assert store.read(key)["state"]["children"][0]["budget_ms"] == 4000


def test_corrupt_checkpoint_fails_closed(tmp_path):
    store, key = store_job(tmp_path)
    path = store.path(key)
    payload = json.loads(path.read_text())
    payload["record"]["phase"] = "UNSAT"  # checksum intentionally left stale
    path.write_text(json.dumps(payload))
    before = path.read_bytes()
    with pytest.raises(ValueError, match="checksum"):
        store.recover()
    with pytest.raises(ValueError, match="checksum"):
        store.register("lift", asdict(job()))
    assert path.read_bytes() == before


def test_failed_atomic_replace_preserves_previous_work(tmp_path, monkeypatch):
    store, key = store_job(tmp_path)
    before = store.path(key).read_bytes()

    def fail(*_args):
        raise OSError("controlled disk error")

    monkeypatch.setattr("solver.checkpoint.os.replace", fail)
    record = store.read(key)
    record["turns"] = 123
    with pytest.raises(OSError, match="disk error"):
        store.save(key, record)
    assert store.path(key).read_bytes() == before


def test_unverified_sat_does_not_mark_job_complete_or_write_solution(tmp_path, monkeypatch):
    bad = replace(unknown_lift(), status="SAT")
    monkeypatch.setattr(staged, "solve_compressed_lift", lambda *_a, **_k: bad)
    store, key = store_job(tmp_path, factor=2)
    with pytest.raises(RuntimeError, match="no candidate"):
        store.advance_lift(key)
    assert store.read(key)["phase"] == "READY"
    assert not (tmp_path / "8.txt").exists()


def test_background_restart_recovers_exact_middle_without_new_seed(tmp_path):
    store, key = store_job(tmp_path)
    store.advance_lift(key)  # simulate interruption after first middle
    assert len(store.read(key)["state"]["children"]) == 1
    done = threading.Event()
    args = main.parse_args(["--L", "8", "--seed", "123", "--z3"])
    bridge = main._CCompletionBridge(args, 8, tmp_path,
                                    lambda report: done.set() if report.get("solved") else None)
    try:
        assert done.wait(10)
    finally:
        bridge.close()
    assert bridge._results[0]["work_kind"] == "child_0"
    assert store.read(key)["phase"] == "SAT"
    assert (tmp_path / "8.txt").exists()


def test_shutdown_leaves_queued_and_active_unknown_jobs_recoverable(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    unknown = unknown_lift()

    def slow(*_args, **_kwargs):
        entered.set()
        assert release.wait(10)
        return unknown

    monkeypatch.setattr(staged, "solve_compressed_lift", slow)
    # This scheduling test intentionally retains two A/B-equivalent jobs;
    # canonical admission is tested separately in test_compressed_equivalence.
    store, first = store_job(tmp_path, factor=2, deduplicate_equivalents=False)
    second = store.register("lift", asdict(other_job()))
    args = main.parse_args(["--L", "8", "--z3", "--no-z3-compressed-dedup"])
    bridge = main._CCompletionBridge(args, 8, tmp_path)
    try:
        assert entered.wait(10)
        bridge._closing.set()  # same stop request made by close() on Ctrl+C
        release.set()
    finally:
        release.set()
        bridge.close()
    assert len(bridge._results) == 1
    assert set(store.recover()) == {first, second}


def test_round_robin_yields_to_other_saved_job(tmp_path, monkeypatch):
    unknown = unknown_lift()
    monkeypatch.setattr(staged, "solve_compressed_lift", lambda *_a, **_k: unknown)
    store, first = store_job(tmp_path, factor=2, initial_ms=10, maximum_ms=20,
                             deduplicate_equivalents=False)
    second = store.register("lift", asdict(other_job()))
    done = threading.Event()
    seen = []

    def observe(result):
        seen.append((result["checkpoint"], result["slice_ms"]))
        if len(seen) == 4:
            done.set()

    args = main.parse_args(["--L", "8", "--z3", "--z3-slice-seconds", ".01",
                            "--z3-timeout", ".02", "--no-z3-compressed-dedup"])
    bridge = main._CCompletionBridge(args, 8, tmp_path, observe)
    try:
        assert done.wait(10)
    finally:
        bridge.close()
    assert seen[0][0] != seen[1][0]
    assert seen[0][0] == seen[2][0] and seen[1][0] == seen[3][0]
    assert [entry[1] for entry in seen] == [10, 10, 20, 20]
    assert {store.read(key)["phase"] for key in (first, second)} == {"WAITING"}


def test_radius_unknown_uses_same_persistent_backoff(tmp_path, monkeypatch):
    budgets, done = [], threading.Event()

    def fake(config, *_args):
        budgets.append(config.z3_timeout)
        return {"status": "Z3_UNKNOWN", "solved": False, "z3_executed": True}

    monkeypatch.setattr(main, "_run_optional_z3", fake)
    args = main.parse_args(["--L", "8", "--z3", "--no-z3-compressed-lift",
                            "--z3-slice-seconds", ".01", "--z3-timeout", ".04"])
    bridge = main._CCompletionBridge(args, 8, tmp_path,
                                    lambda r: done.set() if r["phase"] == "WAITING" else None)
    try:
        with bridge._observation_lock:
            bridge._enqueue_staged("radius", {"a": A, "b": B, "score": 10,
                                              "encoded": (), "parameters": {}})
        assert done.wait(10)
    finally:
        bridge.close()
    assert budgets == [.01, .02, .04]
    assert bridge.z3_execution_count == 3
    assert len(bridge._staged_store.recover()) == 1


@pytest.mark.parametrize("L", (44, 46, 68))
def test_default_enabled_and_rollback_preserved(tmp_path, L):
    args = main.parse_args(["--L", str(L)])
    assert args.z3 and args.z3_staged and args.z3_slice_seconds == 1
    bridge = main._CCompletionBridge(args, L, tmp_path)
    try:
        assert bridge._staged_store.policy.maximum_ms == 60000
        assert bool(bridge._multilevel_pool) == (L % 4 == 0)
    finally:
        bridge.close()
    args.z3_staged = False
    old = main._CCompletionBridge(args, L, tmp_path)
    try:
        assert old._staged_store is None
    finally:
        old.close()


@pytest.mark.parametrize("kwargs", ({"initial_ms": 0}, {"maximum_ms": -1},
                                    {"max_middle_models": 0}, {"initial_ms": True}))
def test_invalid_policy(kwargs):
    with pytest.raises(ValueError):
        staged.SlicePolicy(**kwargs)


def test_zero_ceiling_grows_budget_without_unlimited_single_check(tmp_path, monkeypatch):
    unknown = unknown_lift()
    monkeypatch.setattr(staged, "solve_compressed_lift", lambda *_a, **_k: unknown)
    store, key = store_job(tmp_path, factor=2, initial_ms=10, maximum_ms=0)
    assert [store.advance_lift(key)["slice_ms"] for _ in range(3)] == [10, 20, 40]
    assert store.read(key)["phase"] == "READY"


def test_different_source_for_same_fixed_problem_does_not_duplicate_work(tmp_path):
    store, key = store_job(tmp_path, factor=2)
    # Swapping the two halves changes the source but not its compression.
    source = A[4:] + A[:4]
    assert compress_binary_sequence(source, 2) == job(2).compressed_a
    assert store.register("lift", asdict(replace(job(2), source_a=source))) is None
    assert store.recover() == [key]
    assert tuple(store.read(key)["spec"]["source_a"]) == A


@pytest.mark.parametrize("flag", ("--no-z3-compressed-lift", "--no-z3-multilevel-lift"))
def test_disabled_lift_branch_does_not_resume_but_keeps_checkpoint(tmp_path, flag):
    store, key = store_job(tmp_path)
    args = main.parse_args(["--L", "8", "--z3", flag])
    bridge = main._CCompletionBridge(args, 8, tmp_path)
    bridge.close()
    assert bridge.trigger_count == 0
    assert store.read(key)["phase"] == "READY"


def test_unsat_means_enumeration_and_every_child_exhausted(tmp_path, monkeypatch):
    unsat = replace(unknown_lift(), status="UNSAT", reason="controlled child proof")
    monkeypatch.setattr(staged, "solve_compressed_lift", lambda *_a, **_k: unsat)
    store, key = store_job(tmp_path, max_middle_models=100)
    for _ in range(210):
        result = store.advance_lift(key)
        if result["phase"] != "READY":
            break
    state = store.read(key)["state"]
    assert state["children"] and all(c["status"] == "UNSAT" for c in state["children"])
    assert state["enum"]["status"] == "UNSAT"
    assert result["phase"] == "UNSAT"
    assert store.recover() == []


def test_empty_radius_portfolio_is_unknown_not_unsat(tmp_path, monkeypatch):
    done = threading.Event()
    monkeypatch.setattr(main, "_run_optional_z3", lambda *_a: SimpleNamespace(
        solved=False, status_counts={"SAT": 0, "UNSAT": 0, "UNKNOWN": 0},
        implementation_error=False))
    args = main.parse_args(["--L", "8", "--z3", "--no-z3-compressed-lift",
                            "--z3-timeout", "1"])
    bridge = main._CCompletionBridge(args, 8, tmp_path, lambda _r: done.set())
    try:
        with bridge._observation_lock:
            bridge._enqueue_staged("radius", {"a": A, "b": B, "score": 10,
                                              "encoded": (), "parameters": {"z3_guided": False}})
        assert done.wait(10)
    finally:
        bridge.close()
    assert bridge._results[0]["status"] == "Z3_UNKNOWN"
    assert bridge._results[0]["phase"] == "WAITING"
    assert bridge.z3_execution_count == 0
