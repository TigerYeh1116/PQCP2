"""Odd-half exact preprocessing, independently checked without solution archives."""

from dataclasses import asdict
from itertools import product
import json
from pathlib import Path
import random
import threading

import pytest
import z3

import main
from solver.compressed_completion import CompressedLiftJob, run_compressed_lift_job
from solver.compressed_lift import solve_compressed_lift
from solver.correlation import full_correlation_profile
from solver.orientation_gf2 import odd_orientation_system
from solver.staged_completion import CompletionWorkStore, SlicePolicy
from solver.structured_energy import (
    compress_binary_sequence, fold_profile,
    integer_periodic_autocorrelation_profile as pacf, target_profile,
)
from solver.target_profiles import TargetContentProfile, pair_content
from solver.verifier import verify_pqcp


def negacyclic(a, b):
    d = len(a)
    return tuple(sum(word[i] * word[(i + u) % d] * (1 if i + u < d else -1)
                     for word in (a, b) for i in range(d)) for u in range(d))


@pytest.mark.parametrize("d", [3, 5, 7])
def test_free_parameterization_equals_all_mod4_solutions(d):
    rng = random.Random(d)
    for _ in range(12):
        ca, cb = (tuple(rng.choice((-2, -2, 0, 2)) for _ in range(d)) for _ in range(2))
        variables = [(s, i) for s, c in enumerate((ca, cb)) for i, v in enumerate(c) if v == 0]
        hs = [[rng.choice((-1, 1)) if v == 0 else 0 for v in c] for c in (ca, cb)]
        target = negacyclic(*hs)
        system = odd_orientation_system(ca, cb, target)
        assert system.consistent
        expected = set()
        for bits in product((0, 1), repeat=len(variables)):
            rows = [[0] * d for _ in range(2)]
            for (s, i), value in zip(variables, bits):
                rows[s][i] = 1 - 2 * value
            observed = negacyclic(*rows)
            if all((x - y) % 4 == 0 for x, y in zip(observed, target)):
                expected.add(bits)
        actual = set()
        expressions = system.expressions()
        for assignment in range(1 << len(system.free_columns)):
            actual.add(tuple((bin(mask & assignment).count("1") % 2) ^ constant
                             for mask, constant in expressions))
        assert actual == expected
        assert len(actual) == 1 << (len(variables) - system.rank)


@pytest.mark.parametrize("L", [6, 10, 14, 46, 86, 94])
def test_true_full_profiles_never_rejected_including_L86(L):
    rng = random.Random(L)
    d = L // 2
    for _ in range(30):
        a, b = (tuple(rng.randrange(2) for _ in range(L)) for _ in range(2))
        ca, cb = (compress_binary_sequence(w, 2) for w in (a, b))
        full = full_correlation_profile(a, b)
        residual = tuple((full[u] - full[u + d]) // 4 for u in range(d))
        hs = [tuple(w[i + d] - w[i] for i in range(d)) for w in (a, b)]
        assert negacyclic(*hs) == residual
        system = odd_orientation_system(ca, cb, residual)
        assert system.consistent
        assignment = sum((a, b)[s][i] << j for j, (s, i) in enumerate(system.variables))
        for _, mask, rhs, _ in system.equations:
            assert bin(mask & assignment).count("1") % 2 == rhs


def matched_cases(L, limit):
    """Small compressed patterns from complete ternary lists, NOT known PQCPs."""
    words = list(product((-2, 0, 2), repeat=L // 2))
    random.Random(L).shuffle(words)
    buckets = {}
    for w in words:
        buckets.setdefault(pacf(w), []).append(w)
    count = 0
    for k, eta in product(range(1, L // 2), (-1, 1)):
        wanted = fold_profile(target_profile(L, k, eta), 2)
        for ca in words:
            complement = tuple(t - x for t, x in zip(wanted, pacf(ca)))
            for cb in buckets.get(complement, ()):
                yield ca, cb, k, eta
                count += 1
                if count >= limit:
                    return


def literal_lifts(c):
    zeros = [i for i, v in enumerate(c) if v == 0]
    for choices in product((0, 1), repeat=len(zeros)):
        bits = [int(v == -2) for v in c] * 2
        for i, bit in zip(zeros, choices):
            bits[i], bits[i + len(c)] = bit, 1 - bit
        yield tuple(bits)


@pytest.mark.parametrize("L", [10, 14])
def test_old_new_z3_and_literal_full_oracle_agree(L):
    for ca, cb, k, eta in matched_cases(L, 20):
        expected = any(tuple(full_correlation_profile(a, b)) == target_profile(L, k, eta)
                       for a, b in product(literal_lifts(ca), literal_lifts(cb)))
        for enabled in (False, True):
            result = solve_compressed_lift(ca, cb, k, eta, 2000,
                                           orientation_preprocessing=enabled)
            assert result.status == ("SAT" if expected else "UNSAT")
            if result.verified:
                assert verify_pqcp(result.a, result.b).is_valid
                assert result.profile == target_profile(L, k, eta)
            if enabled:
                assert result.gf2_applied
                assert result.gf2_free_bits == L // 2 - result.gf2_rank


def small_job():
    ca, cb = (2, 2, 0), (0, 0, 2)
    a, b = next(literal_lifts(ca)), next(literal_lifts(cb))
    p = TargetContentProfile(6, 1, 1, *pair_content(a, b))
    return CompressedLiftJob(ca, cb, p, a, b)


def test_preferred_bits_do_not_become_constraints():
    j = small_job()
    result = solve_compressed_lift(j.compressed_a, j.compressed_b, 1, 1, 2000,
                                  preferred_pair=((0,) * 6, (0,) * 6))
    assert result.status == "SAT" and result.verified
    assert result.preferred_values <= result.gf2_free_bits


def test_real_staged_pipeline_and_recovery(tmp_path):
    j = small_job()
    store = CompletionWorkStore(tmp_path, 6, 123, SlicePolicy())
    key = store.register("lift", asdict(j))
    # Reopen saved work with no initialized search or FKM seed.
    recovered = CompletionWorkStore(tmp_path, 6, 123, SlicePolicy())
    assert key in recovered.recover()
    report = recovered.advance_lift(key)
    assert report["gf2_applied"]
    assert report["phase"] in ("SAT", "UNSAT")
    snapshot = json.loads(Path(report["snapshot"]).read_text())
    assert snapshot["gf2_rank"] == report["gf2_rank"]
    assert snapshot["gf2_free_bits"] == report["gf2_free_bits"]
    assert report["z3_calls"] == int(report["z3_executed"])


@pytest.mark.parametrize("enabled", [False, True])
def test_one_shot_and_staged_switch(tmp_path, enabled):
    j = small_job()
    one = run_compressed_lift_job(j, 1000, tmp_path / "one", 123,
                                  orientation_preprocessing=enabled)
    assert one["gf2_applied"] is enabled
    store = CompletionWorkStore(tmp_path / "staged", 6, 123,
                                SlicePolicy(orientation_preprocessing=enabled))
    key = store.register("lift", asdict(j))
    report = store.advance_lift(key)
    assert report["gf2_applied"] is enabled
    assert report["status"] == one["status"]


def test_unknown_is_saved_not_unsat(monkeypatch, tmp_path):
    class Timeout:
        def set(self, **kwargs): pass
        def add(self, *args): pass
        def check(self): return z3.unknown
        def reason_unknown(self): return "controlled timeout"
    monkeypatch.setattr(z3, "SolverFor", lambda *args: Timeout())
    j = small_job()
    store = CompletionWorkStore(tmp_path, 6, 123, SlicePolicy(initial_ms=1, maximum_ms=1))
    key = store.register("lift", asdict(j))
    report = store.advance_lift(key)
    assert report["phase"] == "WAITING"
    assert report["status"] == "LIFT_UNKNOWN" and report["z3_executed"]
    assert key in CompletionWorkStore(tmp_path, 6, 123, SlicePolicy()).recover()


def test_cli_default_and_rollback():
    assert main.parse_args(["--L", "86"]).z3_orientation_preprocessing
    assert not main.parse_args(["--L", "86", "--no-z3-orientation-preprocessing"]).z3_orientation_preprocessing


@pytest.mark.parametrize("enabled", [False, True])
def test_main_one_shot_routes_odd_half_option(tmp_path, enabled):
    options = ["--L", "6", "--seed", "123", "--no-z3-staged"]
    if not enabled:
        options.append("--no-z3-orientation-preprocessing")
    done = threading.Event()
    reports = []
    def callback(report):
        reports.append(report)
        done.set()
    bridge = main._CCompletionBridge(main.parse_args(options), 6, tmp_path, callback)
    try:
        bridge._queue.put(dict(kind="compressed_lift", job=small_job()))
        assert done.wait(5)
    finally:
        bridge.close()
    assert reports[0]["gf2_applied"] is enabled


def test_L86_production_startup_enables_preprocessing(tmp_path, capsys):
    args = main.parse_args(["--L", "86", "--seed", "123"])
    bridge = main._CCompletionBridge(args, 86, tmp_path)
    try:
        assert bridge._staged_store.policy.orientation_preprocessing
        assert bridge._lift_pool.factor == 2
        assert bridge._multilevel_pool is None
    finally:
        bridge.close()
    assert "奇數半長度方向 GF(2) 前處理已啟用" in capsys.readouterr().out


def test_even_half_encoding_unchanged():
    a = solve_compressed_lift((2, 2), (0, 0), 1, 1, 1000)
    b = solve_compressed_lift((2, 2), (0, 0), 1, 1, 1000, orientation_preprocessing=False)
    assert a.status == b.status == "SAT"
    assert not a.gf2_applied and a.correlation_constraints == b.correlation_constraints


def test_bad_preprocessing_argument():
    with pytest.raises(ValueError):
        solve_compressed_lift((2, 2, 0), (0, 0, 2), 1, 1, orientation_preprocessing=1)


def test_linear_contradiction_rejects_without_z3(monkeypatch):
    ca, cb = (0, 0, 2, 2, -2, 2, 0), (-2, 0, -2, 0, 2, 0, 0)
    assert solve_compressed_lift(ca, cb, 1, -1, 1000,
                                 orientation_preprocessing=False).status == "UNSAT"
    def forbidden(*args):
        raise AssertionError("linear contradiction must not construct Z3")
    monkeypatch.setattr(z3, "SolverFor", forbidden)
    result = solve_compressed_lift(ca, cb, 1, -1, 1000)
    assert result.status == "UNSAT" and not result.z3_executed
    assert result.gf2_applied and "GF(2)" in result.reason
    assert result.correlation_constraints == 0


def test_invalid_odd_system_inputs():
    for ca, cb, r in [((0, 0), (0, 0), (0, 0)),
                      ((0, 0, True), (0, 0, 0), (0, 0, 0)),
                      ((0, 0, 0), (0, 0, 0), (0, 0))]:
        with pytest.raises(ValueError):
            odd_orientation_system(ca, cb, r)
