"""Correctness tests for experimental gradient-ranked exact moves."""

import pytest

torch = pytest.importorskip("torch")

from solver.annealing import initialize_from_fkm_content_profile
from solver.correlation import full_correlation_profile
from solver.target_profiles import canonical_target_content_profiles, pair_content
from solver.torch_guidance import (
    rank_same_parity_swaps, torch_gradient_kick, torch_gradient_repair,
)


def test_ranked_moves_are_legal_and_preserve_parity_content():
    profile = canonical_target_content_profiles(8)[0]
    a, b = initialize_from_fkm_content_profile(profile, seed=5)
    moves = rank_same_parity_swaps(a, b, profile, limit=20)
    assert moves
    for move in moves:
        values = a if move.sequence == "a" else b
        assert values[move.zero_position] == 0
        assert values[move.one_position] == 1
        assert move.zero_position % 2 == move.one_position % 2


def test_gradient_repair_is_exact_deterministic_and_never_worsens_objective():
    profile = canonical_target_content_profiles(8)[0]
    a, b = initialize_from_fkm_content_profile(profile, seed=9)
    first = torch_gradient_repair(a, b, profile, max_steps=5, candidate_limit=10)
    second = torch_gradient_repair(a, b, profile, max_steps=5, candidate_limit=10)
    assert first == second
    assert first.best_score <= first.initial_score
    assert pair_content(first.a, first.b) == pair_content(a, b)
    assert len(full_correlation_profile(first.a, first.b)) == 8


def test_gradient_kick_is_deterministic_and_content_preserving():
    profile = canonical_target_content_profiles(8)[0]
    a, b = initialize_from_fkm_content_profile(profile, seed=12)
    first = torch_gradient_kick(a, b, profile, steps=3, candidate_limit=8)
    second = torch_gradient_kick(a, b, profile, steps=3, candidate_limit=8)
    assert first == second
    assert pair_content(*first) == pair_content(a, b)
