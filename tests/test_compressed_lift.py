"""Exact lift encoding cross-checked against full binary enumeration."""

from dataclasses import replace
from itertools import product
from unittest.mock import patch

import pytest
import z3

from solver.compressed_lift import solve_compressed_lift
from solver.correlation import full_correlation_profile, periodic_autocorrelation
from solver.structured_energy import (
    compress_binary_sequence, fold_profile,
    integer_periodic_autocorrelation_profile, target_profile,
)
from solver.target_profiles import TargetContentProfile, pair_content, target_content_profiles
from solver.verifier import verify_pqcp


@pytest.mark.parametrize("L", (4, 6, 8))
def test_all_matched_compressed_patterns_against_full_brute_force(L):
    words = tuple(product((0, 1), repeat=L))
    pacfs = {w: tuple(periodic_autocorrelation(w, u) for u in range(L)) for w in words}
    compressed = {w: compress_binary_sequence(w, 2) for w in words}
    feasible = set()
    for a in words:
        for b in words:
            profile = tuple(x + y for x, y in zip(pacfs[a], pacfs[b]))
            nz = [u for u in range(1, L) if profile[u]]
            if len(nz) == 2 and all(abs(profile[u]) == 4 for u in nz):
                feasible.add((compressed[a], compressed[b], min(nz), profile[min(nz)] // 4))
    patterns = set(compressed.values())
    compressed_pacfs = {c: integer_periodic_autocorrelation_profile(c) for c in patterns}
    checked = 0
    for ca in patterns:
        for cb in patterns:
            actual = tuple(x + y for x, y in zip(compressed_pacfs[ca], compressed_pacfs[cb]))
            for k in range(1, L // 2):
                for eta in (-1, 1):
                    if actual != fold_profile(target_profile(L, k, eta), 2):
                        continue
                    result = solve_compressed_lift(ca, cb, k, eta, timeout_ms=2_000)
                    assert result.status in ("SAT", "UNSAT")
                    assert (result.status == "SAT") == ((ca, cb, k, eta) in feasible)
                    assert result.orientation_variables == L // 2
                    assert result.correlation_constraints == L // 2 - 1
                    if result.status == "SAT":
                        assert result.verified and verify_pqcp(result.a, result.b).is_valid
                        assert compress_binary_sequence(result.a, 2) == ca
                        assert compress_binary_sequence(result.b, 2) == cb
                        assert result.profile == tuple(full_correlation_profile(result.a, result.b))
                        assert result.profile == target_profile(L, k, eta)
                    checked += 1
    assert checked > 0


def test_all_content_profiles_at_L6_against_brute_force():
    # L/2 odd: orientations affect alternating sums. Compare classifications
    # with the actual four parity counts, not merely ordinary weights.
    L = 6
    words = tuple(product((0, 1), repeat=L))
    patterns = {compress_binary_sequence(w, 2) for w in words}
    lifts = {c: [w for w in words if compress_binary_sequence(w, 2) == c] for c in patterns}
    ca, cb = (2, 2, 0), (0, 0, 2)
    for content in target_content_profiles(L):
        expected = any(
            pair_content(a, b) == (content.a_even_ones, content.a_odd_ones,
                                  content.b_even_ones, content.b_odd_ones)
            and tuple(full_correlation_profile(a, b)) == target_profile(L, content.k, content.eta)
            for a in lifts[ca] for b in lifts[cb]
        )
        result = solve_compressed_lift(ca, cb, content.k, content.eta, 1_000, content)
        assert result.status in ("SAT", "UNSAT")
        assert (result.status == "SAT") == expected
        if result.status == "SAT":
            assert pair_content(result.a, result.b) == (
                content.a_even_ones, content.a_odd_ones,
                content.b_even_ones, content.b_odd_ones)


def test_non_solution_source_can_lift_to_exact_solution_without_radius():
    a, b = tuple(map(int, "00000111")), tuple(map(int, "00010011"))
    assert not verify_pqcp(a, b).is_valid
    content = TargetContentProfile(8, 2, -1, *pair_content(a, b))
    result = solve_compressed_lift(compress_binary_sequence(a, 2),
                                   compress_binary_sequence(b, 2), 2, -1, 1_000, content)
    assert result.status == "SAT" and result.verified
    assert result.content_constraints == 4


def test_antipodal_placement_not_adjacent_expansion():
    result = solve_compressed_lift((2, 2), (0, 0), 1, 1, 1_000)
    assert result.status == "SAT"
    assert result.a == (0, 0, 0, 0)
    assert result.b[0] != result.b[2] and result.b[1] != result.b[3]


def test_unmatched_compression_is_local_unsat_without_z3():
    with patch("z3.SolverFor", side_effect=AssertionError("must not build a solver")):
        result = solve_compressed_lift((2,) * 34, (0,) * 34, 1, 1, 1_000)
    assert result.status == "UNSAT" and not result.z3_executed
    assert result.L == 68 and result.orientation_variables == 34
    assert "compressed PACF" in result.reason


def test_unknown_remains_unknown(monkeypatch):
    class TimeoutSolver:
        def set(self, **kwargs): pass
        def add(self, *_args): pass
        def check(self): return z3.unknown
        def reason_unknown(self): return "test timeout"
    monkeypatch.setattr(z3, "SolverFor", lambda *_args: TimeoutSolver())
    result = solve_compressed_lift((2, 2), (0, 0), 1, 1, 1)
    assert result.status == "UNKNOWN" and result.reason == "test timeout"
    assert result.z3_executed and not result.verified and result.a is None


def test_sat_verification_failure_raises(monkeypatch):
    monkeypatch.setattr("solver.compressed_lift.verify_pqcp",
                        lambda a, b: replace(verify_pqcp(a, b), is_valid=False))
    with pytest.raises(RuntimeError, match="independent"):
        solve_compressed_lift((2, 2), (0, 0), 1, 1, 1_000)


@pytest.mark.parametrize("ca,cb,k,eta,timeout", (
    ((True, 0), (0, 0), 1, 1, 100),
    ((1, 0), (0, 0), 1, 1, 100),
    ((2,), (0,), 1, 1, 100),
    ((2, 2), (0, 0, 0), 1, 1, 100),
    ((2, 2), (0, 0), 2, 1, 100),
    ((2, 2), (0, 0), True, 1, 100),
    ((2, 2), (0, 0), 1, True, 100),
    ((2, 2), (0, 0), 1, 0, 100),
    ((2, 2), (0, 0), 1, 1, -1),
    ((2, 2), (0, 0), 1, 1, 1.5),
))
def test_invalid_inputs(ca, cb, k, eta, timeout):
    with pytest.raises(ValueError):
        solve_compressed_lift(ca, cb, k, eta, timeout)


def test_mismatched_content_target_raises():
    with pytest.raises(ValueError, match="content profile"):
        solve_compressed_lift((2, 2), (0, 0), 1, 1, 100,
                              TargetContentProfile(4, 1, -1, 0, 0, 1, 1))
