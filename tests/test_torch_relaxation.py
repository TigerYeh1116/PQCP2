"""Lightweight correctness tests for the optional PyTorch experiment."""

import pytest

torch = pytest.importorskip("torch")

from solver.correlation import full_correlation_profile
from solver.objective import pqcp_objective
from solver.torch_relaxation import (
    RelaxationParameters,
    project_logits_fixed_weight,
    project_logits_target_content,
    relax_candidate,
    relax_candidate_for_profile,
    relax_candidate_batch_for_profile,
    torch_pair_correlation,
)
from solver.target_profiles import canonical_target_content_profiles, pair_content


def test_torch_profile_matches_exact_binary_sign_profile():
    a = (0, 1, 1, 0, 1, 0)
    b = (1, 0, 1, 1, 0, 0)
    sign_a = torch.tensor([1.0 if bit == 0 else -1.0 for bit in a], dtype=torch.float64)
    sign_b = torch.tensor([1.0 if bit == 0 else -1.0 for bit in b], dtype=torch.float64)
    assert tuple(int(value) for value in torch_pair_correlation(sign_a, sign_b).tolist()) == tuple(
        full_correlation_profile(a, b)
    )


@pytest.mark.parametrize("weight", (0, 1, 3, 5))
def test_projection_has_exact_requested_weight(weight):
    logits = torch.tensor([0.5, -2.0, 0.1, 4.0, -0.3], dtype=torch.float64)
    bits = project_logits_fixed_weight(logits, weight)
    assert len(bits) == 5
    assert sum(bits) == weight
    assert set(bits) <= {0, 1}


def test_relaxation_is_deterministic_and_returns_exact_consistent_candidate():
    a = (0, 1, 0, 1, 1, 0, 0, 1)
    b = (1, 0, 0, 1, 0, 1, 1, 0)
    parameters = RelaxationParameters(steps=8, observation_interval=2, seed=77)
    first = relax_candidate(a, b, parameters)
    second = relax_candidate(a, b, parameters)
    assert first == second
    assert sum(first.best_a) == sum(a)
    assert sum(first.best_b) == sum(b)
    assert first.best_profile == tuple(full_correlation_profile(first.best_a, first.best_b))
    assert first.best_score == pqcp_objective(first.best_profile)
    assert first.best_score <= first.initial_score


def test_profile_projection_preserves_all_four_parity_counts():
    profile = canonical_target_content_profiles(8)[0]
    logits_a = torch.tensor([0.7, -0.2, 1.1, -2.0, 0.1, 0.4, -0.8, 0.0])
    logits_b = -logits_a
    a, b = project_logits_target_content(logits_a, logits_b, profile)
    assert pair_content(a, b) == (
        profile.a_even_ones, profile.a_odd_ones,
        profile.b_even_ones, profile.b_odd_ones,
    )


def test_profile_relaxation_is_deterministic_exact_and_content_preserving():
    profile = canonical_target_content_profiles(8)[0]
    logits = torch.arange(8, dtype=torch.float64)
    a, b = project_logits_target_content(logits, -logits, profile)
    parameters = RelaxationParameters(steps=8, observation_interval=2, seed=88)
    first = relax_candidate_for_profile(a, b, profile, parameters)
    second = relax_candidate_for_profile(a, b, profile, parameters)
    assert first == second
    assert pair_content(first.best_a, first.best_b) == pair_content(a, b)
    assert first.best_profile == tuple(full_correlation_profile(first.best_a, first.best_b))
    assert first.best_score == pqcp_objective(first.best_profile)
    assert first.best_score <= first.initial_score


def test_batched_profile_relaxation_is_exact_deterministic_and_no_worse_than_inputs():
    profile = canonical_target_content_profiles(8)[0]
    first_logits = torch.arange(8, dtype=torch.float64)
    second_logits = torch.tensor([3.0, -1.0, 2.0, 0.0, -2.0, 4.0, 1.0, -3.0])
    candidates = (
        project_logits_target_content(first_logits, -first_logits, profile),
        project_logits_target_content(second_logits, -second_logits, profile),
    )
    parameters = RelaxationParameters(steps=6, observation_interval=2, seed=99)
    first = relax_candidate_batch_for_profile(candidates, profile, parameters)
    second = relax_candidate_batch_for_profile(candidates, profile, parameters)
    assert first == second
    assert pair_content(first.best_a, first.best_b) == pair_content(*candidates[0])
    assert first.best_profile == tuple(full_correlation_profile(first.best_a, first.best_b))
    assert first.best_score <= min(
        pqcp_objective(full_correlation_profile(*candidate)) for candidate in candidates
    )
