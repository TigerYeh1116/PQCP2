"""Small exact-completion tests, including independent verifier checks."""

from itertools import product

import pytest

import solver.z3_solver as z3_solver
from solver.target_profiles import target_content_profiles, profile_matches_pair_content
from solver.z3_solver import save_verified_solution, solve_with_z3


KNOWN_A = (0, 0, 0, 0)
KNOWN_B = (0, 0, 1, 1)


def test_radius_zero_known_solution_is_sat_and_independently_verified():
    result = solve_with_z3(4, KNOWN_A, KNOWN_B, radius=0, timeout_ms=1_000)
    assert result.status == "SAT"
    assert result.verified
    assert result.a == KNOWN_A
    assert result.b == KNOWN_B


def test_radius_zero_non_solution_is_unsat():
    result = solve_with_z3(4, (0, 0, 0, 0), (0, 0, 0, 0), radius=0, timeout_ms=1_000)
    assert result.status == "UNSAT"


def test_positive_radius_finds_a_nearby_known_solution():
    # B differs from KNOWN_B at exactly one bit, so radius one can complete it.
    result = solve_with_z3(4, KNOWN_A, (0, 0, 1, 0), radius=1, timeout_ms=1_000)
    assert result.status == "SAT"
    assert result.verified


def test_known_globally_unsatisfiable_small_length_is_unsat():
    result = solve_with_z3(2, (0, 0), (0, 0), radius=3, timeout_ms=1_000)
    assert result.status == "UNSAT"


@pytest.mark.parametrize("length,radius", ((3, 1), (4, 2), (5, 1), (6, 2)))
def test_bool_flip_encoding_matches_brute_force_neighborhood_existence(length, radius):
    center_a = tuple(index % 2 for index in range(length))
    center_b = tuple((index // 2) % 2 for index in range(length))
    expected = any(
        sum(left != right for left, right in zip(center_a + center_b, a + b)) <= radius
        and z3_solver.verify_pqcp(a, b).is_valid
        for a in product((0, 1), repeat=length)
        for b in product((0, 1), repeat=length)
    )
    result = solve_with_z3(length, center_a, center_b, radius=radius, timeout_ms=5_000)
    assert (result.status == "SAT") is expected
    assert result.status in ("SAT", "UNSAT")
    if result.status == "SAT":
        assert result.verified


def test_same_constraints_have_same_sat_classification():
    first = solve_with_z3(6, (0, 0, 0, 0, 0, 1), (0, 0, 0, 0, 1, 1), radius=0, timeout_ms=1_000)
    second = solve_with_z3(6, (0, 0, 0, 0, 0, 1), (0, 0, 0, 0, 1, 1), radius=0, timeout_ms=1_000)
    assert first.status == second.status == "SAT"
    assert first.verified and second.verified


def test_reachable_weight_pairs_remove_only_pairs_outside_hamming_ball():
    center_a = (1,) * 18 + (0,) * 26
    center_b = (1,) * 20 + (0,) * 24
    assert z3_solver._reachable_admissible_weight_pairs(44, center_a, center_b, 3) == ((18, 20),)
    assert (18, 24) in z3_solver._reachable_admissible_weight_pairs(44, center_a, center_b, 4)


def test_target_content_encoding_preserves_sat_classification_on_small_cases():
    for length in (4, 6, 8):
        center_a = tuple(index % 2 for index in range(length))
        center_b = tuple((index // 2) % 2 for index in range(length))
        for radius in (0, 1, 2):
            old = solve_with_z3(
                length, center_a, center_b, radius=radius, timeout_ms=5_000,
                use_target_content_profiles=False,
            )
            new = solve_with_z3(
                length, center_a, center_b, radius=radius, timeout_ms=5_000,
                use_target_content_profiles=True,
            )
            assert new.status == old.status


def test_explicit_target_profile_handoff_is_exact_for_known_solution():
    compatible = tuple(
        profile for profile in target_content_profiles(4)
        if profile_matches_pair_content(profile, KNOWN_A, KNOWN_B)
    )
    assert compatible
    result = solve_with_z3(
        4, KNOWN_A, KNOWN_B, radius=0, timeout_ms=1_000,
        allowed_target_content_profiles=compatible,
    )
    assert result.status == "SAT" and result.verified

    incompatible = next(
        profile for profile in target_content_profiles(4)
        if not profile_matches_pair_content(profile, KNOWN_A, KNOWN_B)
    )
    result = solve_with_z3(
        4, KNOWN_A, KNOWN_B, radius=0, timeout_ms=1_000,
        allowed_target_content_profiles=(incompatible,),
    )
    assert result.status == "UNSAT"


def test_unknown_is_reported_as_unknown_not_unsat(monkeypatch):
    class TimeoutSolver:
        def set(self, **_kwargs):
            pass

        def add(self, *_constraints):
            pass

        def check(self):
            return z3_solver.z3.unknown

        def reason_unknown(self):
            return "timeout"

    monkeypatch.setattr(z3_solver, "_make_solver", TimeoutSolver)
    result = solve_with_z3(4, KNOWN_A, KNOWN_B, radius=0, timeout_ms=1)
    assert result.status == "UNKNOWN"
    assert result.reason == "timeout"


def test_verified_solution_storage_never_overwrites(tmp_path):
    result = solve_with_z3(4, KNOWN_A, KNOWN_B, radius=0, timeout_ms=1_000)
    saved = save_verified_solution(result, tmp_path)
    contents = saved.read_text(encoding="utf-8")
    assert "L=4" in contents
    assert "verification status=verified" in contents
    with pytest.raises(FileExistsError):
        save_verified_solution(result, tmp_path)


@pytest.mark.parametrize("radius", [-1, 9, True])
def test_invalid_radius_is_rejected(radius):
    with pytest.raises(ValueError):
        solve_with_z3(4, KNOWN_A, KNOWN_B, radius=radius)
