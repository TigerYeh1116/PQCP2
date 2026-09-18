"""Small exhaustive classifications and conservative multilevel budgeting."""

from dataclasses import replace
from itertools import product
import json

import pytest

import main
from solver.compressed_completion import CompressedLiftJob, CompressedLiftPool, run_compressed_lift_job
from solver.compressed_lift import solve_compressed_lift
from solver.correlation import periodic_autocorrelation, full_correlation_profile
from solver.multilevel_lift import solve_multilevel_lift
from solver.structured_energy import compress_binary_sequence, fold_profile, integer_periodic_autocorrelation_profile, target_profile
from solver.target_profiles import TargetContentProfile, pair_content, target_content_profiles
from solver.verifier import verify_pqcp


@pytest.mark.parametrize("L", (4, 8))
def test_all_factor4_matches_against_full_binary_brute_force(L):
    words = tuple(product((0, 1), repeat=L))
    profiles = {w: tuple(periodic_autocorrelation(w, u) for u in range(L)) for w in words}
    compressed = {w: compress_binary_sequence(w, 4) for w in words}
    feasible = set()
    for a in words:
        for b in words:
            profile = tuple(x+y for x, y in zip(profiles[a], profiles[b]))
            nz = [u for u in range(1, L) if profile[u]]
            if len(nz) == 2 and all(abs(profile[u]) == 4 for u in nz):
                feasible.add((compressed[a], compressed[b], min(nz), profile[min(nz)] // 4))
    patterns = set(compressed.values())
    checked = 0
    for ca in patterns:
        for cb in patterns:
            actual = tuple(x+y for x, y in zip(integer_periodic_autocorrelation_profile(ca),
                                               integer_periodic_autocorrelation_profile(cb)))
            for k in range(1, L//2):
                for eta in (-1, 1):
                    if actual != fold_profile(target_profile(L, k, eta), 4):
                        continue
                    result = solve_multilevel_lift(ca, cb, k, eta, 2_000,
                                                   max_middle_models=None)
                    assert result.status in ("SAT", "UNSAT")
                    assert (result.status == "SAT") == ((ca, cb, k, eta) in feasible)
                    if result.status == "SAT":
                        assert result.verified and verify_pqcp(result.a, result.b).is_valid
                        assert result.profile == tuple(full_correlation_profile(result.a, result.b))
                        assert result.profile == target_profile(L, k, eta)
                        assert compress_binary_sequence(result.a, 4) == ca
                        assert compress_binary_sequence(result.b, 4) == cb
                        assert compress_binary_sequence(result.a, 2) == result.middle_a
                        assert compress_binary_sequence(result.b, 2) == result.middle_b
                    checked += 1
    assert checked > 0


def test_source_need_not_be_solution_and_preserves_exact_parity_content(tmp_path):
    a, b = tuple(map(int, "00000111")), tuple(map(int, "00010011"))
    assert not verify_pqcp(a, b).is_valid
    target = TargetContentProfile(8, 2, -1, *pair_content(a, b))
    job = CompressedLiftJob(compress_binary_sequence(a, 4), compress_binary_sequence(b, 4),
                            target, a, b, factor=4)
    result = run_compressed_lift_job(job, 2_000, tmp_path, 77)
    assert result["status"] == "LIFT_SAT" and result["new_solution_written"]
    snapshot = json.loads(open(result["snapshot"]).read())
    assert snapshot["method"] == "multilevel_z3_lift" and snapshot["factor"] == 4
    assert snapshot["profile"] == full_correlation_profile(snapshot["A"], snapshot["B"])
    assert snapshot["objective_components"]["total"] == 0
    assert pair_content(snapshot["A"], snapshot["B"]) == pair_content(a, b)
    assert result["z3_calls"] == snapshot["middle_checks"] + snapshot["binary_checks"]
    assert (tmp_path / "8.txt").exists()


@pytest.mark.parametrize("eta,cb", ((-1,(0,0,0)),(1,(4,0,0))))
def test_L12_exact_coarse_match_without_binary_lift_against_brute_force(eta,cb):
    # Odd compressed dimension >1 exercises i/i+n placement and exhausting
    # multiple middle models. These are locally infeasible fixed patterns,
    # not a claim that L12 has no PQCPs.
    ca = (4,0,0)
    words = tuple(product((0,1),repeat=12))
    aa = [w for w in words if compress_binary_sequence(w,4) == ca]
    bb = [w for w in words if compress_binary_sequence(w,4) == cb]
    pa = {w: tuple(periodic_autocorrelation(w,u) for u in range(12)) for w in aa}
    pb = {w: tuple(periodic_autocorrelation(w,u) for u in range(12)) for w in bb}
    desired = target_profile(12,3,eta)
    feasible = any(tuple(x+y for x,y in zip(pa[a],pb[b])) == desired for a in aa for b in bb)
    result = solve_multilevel_lift(ca,cb,3,eta,2_000,max_middle_models=None)
    assert result.status in ("SAT","UNSAT")
    assert (result.status == "SAT") == feasible
    assert not feasible and result.middle_models > 1
    assert result.middle_checks == result.middle_models + 1
    assert result.binary_checks == result.middle_models


def test_unmatched_factor4_is_local_unsat_without_solver(monkeypatch):
    monkeypatch.setattr("z3.SolverFor", lambda *_a: pytest.fail("unmatched pair built solver"))
    result = solve_multilevel_lift((4,)*17, (0,)*17, 1, 1, 100)
    assert result.status == "UNSAT" and not result.z3_executed and result.L == 68


def test_unknown_child_is_not_unsat_even_after_exhausting_middle(monkeypatch):
    child = solve_compressed_lift((2,2), (0,0), 1, 1, 1_000)
    unknown = replace(child, status="UNKNOWN", verified=False, a=None, b=None, profile=None)
    monkeypatch.setattr("solver.multilevel_lift.solve_compressed_lift", lambda *_a, **_kw: unknown)
    result = solve_multilevel_lift((4,), (0,), 1, 1, 1_000, max_middle_models=None)
    assert result.status == "UNKNOWN" and result.unknown_children == 1
    assert result.middle_models == 1 and "unresolved" in result.reason


def test_model_cap_is_unknown_not_unsat(monkeypatch):
    child = solve_compressed_lift((2,2), (0,0), 1, 1, 1_000)
    unsat = replace(child, status="UNSAT", verified=False, a=None, b=None, profile=None)
    monkeypatch.setattr("solver.multilevel_lift.solve_compressed_lift", lambda *_a, **_kw: unsat)
    result = solve_multilevel_lift((0,), (0,), 1, -1, 1_000, max_middle_models=1)
    assert result.status == "UNKNOWN" and result.middle_models == 1
    assert "model budget" in result.reason


def test_middle_unknown_is_not_unsat(monkeypatch):
    import z3
    class Timeout:
        def add(self, *_a): pass
        def set(self, **_kw): pass
        def check(self): return z3.unknown
        def reason_unknown(self): return "test timeout"
    monkeypatch.setattr(z3, "SolverFor", lambda *_a: Timeout())
    result = solve_multilevel_lift((4,), (0,), 1, 1, 1_000)
    assert result.status == "UNKNOWN" and result.reason == "test timeout"
    assert result.middle_checks == 1 and result.binary_checks == 0


def test_shared_budget_exhaustion_before_check_is_unknown(monkeypatch):
    clock = [0.0]
    def tick():
        clock[0] += 0.010
        return clock[0]
    monkeypatch.setattr("solver.multilevel_lift.perf_counter", tick)
    result = solve_multilevel_lift((4,), (0,), 1, 1, timeout_ms=1)
    assert result.status == "UNKNOWN" and not result.z3_executed


def test_all_L4_content_profiles_against_actual_binary_completions():
    words = tuple(product((0,1), repeat=4))
    compressed = {w: compress_binary_sequence(w,4) for w in words}
    feasible = {(compressed[a], compressed[b], pair_content(a,b), k, eta)
                for a in words for b in words for k in (1,) for eta in (-1,1)
                if tuple(full_correlation_profile(a,b)) == target_profile(4,k,eta)}
    patterns = set(compressed.values())
    for content in target_content_profiles(4):
        counts = (content.a_even_ones,content.a_odd_ones,
                  content.b_even_ones,content.b_odd_ones)
        for ca in patterns:
            for cb in patterns:
                result = solve_multilevel_lift(ca,cb,content.k,content.eta,1_000,content,
                                               max_middle_models=None)
                assert result.status in ("SAT","UNSAT")
                assert (result.status == "SAT") == ((ca,cb,counts,content.k,content.eta) in feasible)


def test_background_bridge_runs_multilevel_and_saves_candidate(tmp_path):
    a,b = tuple(map(int,"00000111")),tuple(map(int,"00010011"))
    args = main.parse_args(["--L","8","--z3","--z3-trigger-score","0","--z3-timeout","2",
                            "--no-z3-staged"])
    bridge = main._CCompletionBridge(args,8,tmp_path)
    try:
        bridge.consider(a,b,{"k":2,"sign":-1,"score":100})
    finally:
        bridge.close()
    assert bridge._results[0]["factor"] == 4
    assert bridge._results[0]["status"] == "LIFT_SAT"
    assert bridge.z3_execution_count == bridge._results[0]["z3_calls"]
    assert (tmp_path/"8.txt").exists()


def test_multilevel_can_be_disabled_without_disabling_factor2(tmp_path):
    args = main.parse_args(["--L","68","--z3","--no-z3-multilevel-lift"])
    bridge = main._CCompletionBridge(args,68,tmp_path)
    try:
        assert bridge._multilevel_pool is None and bridge._lift_pool is not None
    finally:
        bridge.close()


def test_factor4_pool_and_rollback_defaults():
    a, b = tuple(map(int, "00000111")), tuple(map(int, "00010011"))
    pool = CompressedLiftPool(8, factor=4)
    jobs = pool.offer(a,b,2,-1)
    assert jobs and all(job.factor == 4 for job in jobs)
    for job in jobs: pool.submit(job)
    assert not pool.offer(a,b,2,-1)
    assert main.parse_args(["--L","68","--z3"]).z3_multilevel_lift
    assert not main.parse_args(["--L","68","--z3","--no-z3-multilevel-lift"]).z3_multilevel_lift
    with pytest.raises(ValueError): CompressedLiftPool(46, factor=4)


@pytest.mark.parametrize("L,enabled", ((68,True),(44,True),(46,False)))
def test_bridge_activates_multilevel_only_at_divisible_lengths(tmp_path,L,enabled):
    args = main.parse_args(["--L",str(L),"--z3"])
    bridge = main._CCompletionBridge(args,L,tmp_path)
    try:
        assert (bridge._multilevel_pool is not None) == enabled
    finally:
        bridge.close()


@pytest.mark.parametrize("kw", ({"max_middle_models":0},{"timeout_ms":-1},
                                 {"max_middle_models":True}))
def test_invalid_budgets_are_rejected(kw):
    with pytest.raises(ValueError): solve_multilevel_lift((4,), (0,), 1, 1, **kw)
