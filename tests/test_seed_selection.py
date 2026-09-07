"""Tests for deterministic fixed-content FKM seed selection."""

from dataclasses import asdict

import pytest

from solver.annealing import initialize_from_fkm_content_profile
from solver.correlation import full_correlation_profile
from solver.objective import pqcp_objective
from solver.seed_selection import (
    SEED_POLICIES,
    seed_energy_key,
    select_fkm_content_seed,
)
from solver.structured_energy import structured_energy_breakdown
from solver.target_profiles import (
    TargetContentProfile,
    canonical_target_content_profiles,
    pair_content,
)


def _profile(length=8):
    return canonical_target_content_profiles(length)[0]


@pytest.mark.parametrize(
    "length,seed,pool_size", ((8, 19, 16), (14, -5, 16), (44, 123, 32)),
)
def test_legacy_is_exact_rollback_of_existing_initializer(length, seed, pool_size):
    profile = _profile(length)
    expected = initialize_from_fkm_content_profile(
        profile, seed=seed, pool_size=pool_size
    )
    selected = select_fkm_content_seed(
        profile, seed=seed, pool_size=pool_size, policy="legacy"
    )
    assert selected.pair == expected
    assert selected.diagnostics.policy == "legacy"
    assert selected.diagnostics.candidates_examined == 1
    assert selected.diagnostics.selected_phase_a == 0
    assert selected.diagnostics.selected_phase_b == 0


@pytest.mark.parametrize("policy", SEED_POLICIES)
@pytest.mark.parametrize("length", (8, 14, 44))
def test_every_policy_preserves_exact_target_content(policy, length):
    profile = _profile(length)
    selected = select_fkm_content_seed(
        profile,
        seed=1234 + length,
        pool_size=16,
        policy=policy,
        candidate_count=8,
        elite_count=3,
    )
    expected = (
        profile.a_even_ones,
        profile.a_odd_ones,
        profile.b_even_ones,
        profile.b_odd_ones,
    )
    assert len(selected.a) == len(selected.b) == length
    assert pair_content(selected.a, selected.b) == expected
    assert selected.diagnostics.target_content == expected
    assert sum(selected.a) == sum(expected[:2])
    assert sum(selected.b) == sum(expected[2:])


@pytest.mark.parametrize("policy", SEED_POLICIES)
def test_same_seed_is_fully_deterministic_including_diagnostics(policy):
    profile = _profile(12)
    kwargs = dict(
        profile=profile,
        seed=-987654321,
        pool_size=20,
        policy=policy,
        candidate_count=12,
        elite_count=4,
    )
    first = select_fkm_content_seed(**kwargs)
    second = select_fkm_content_seed(**kwargs)
    assert first == second
    # Diagnostics are intentionally JSON/asdict-friendly except for tuples,
    # which json serializes as arrays without custom objects.
    assert asdict(first.diagnostics) == asdict(second.diagnostics)


def test_phase_policies_use_domain_separated_repeatable_choices():
    profile = _profile(44)
    random_phase = select_fkm_content_seed(
        profile, 314159, pool_size=32, policy="phase_random"
    )
    ranked = select_fkm_content_seed(
        profile, 314159, pool_size=32, policy="phase_top_q",
        candidate_count=10, elite_count=3,
    )
    assert random_phase.diagnostics.candidates_examined == 1
    assert ranked.diagnostics.candidates_examined == 10
    assert ranked.diagnostics.elite_count == 3
    assert 0 <= ranked.diagnostics.selected_rank < 3
    # Candidate enumeration is independent of the top-q choice stream.
    reranked = select_fkm_content_seed(
        profile, 314159, pool_size=32, policy="phase_top_q",
        candidate_count=10, elite_count=1,
    )
    assert ranked.diagnostics.ranked_phases == reranked.diagnostics.ranked_phases
    assert ranked.diagnostics.ranked_energy_keys == reranked.diagnostics.ranked_energy_keys


def test_every_policy_uses_the_same_legacy_fkm_necklace_choices():
    """Policy ablations may change relative phase, never the four base words."""
    profile = _profile(44)
    selections = tuple(
        select_fkm_content_seed(
            profile, 271828, pool_size=32, policy=policy,
            candidate_count=12, elite_count=4,
        )
        for policy in ("legacy", "phase_random", "phase_top_q")
    )
    assert len({item.diagnostics.pool_indices for item in selections}) == 1


def test_compressed_top_q_reports_separate_candidate_source():
    profile = _profile(44)
    selected = select_fkm_content_seed(
        profile, 271828, pool_size=32, policy="compressed_top_q",
        candidate_count=12, elite_count=4,
    )
    diagnostics = selected.diagnostics
    assert diagnostics.pool_indices == (-1, -1, -1, -1)
    assert diagnostics.selected_phase_a == diagnostics.selected_phase_b == -1
    assert diagnostics.candidates_examined == 12
    assert diagnostics.ranked_energy_keys == tuple(sorted(
        diagnostics.ranked_energy_keys
    ))


def test_phase_top_q_caps_unique_phase_enumeration():
    profile = _profile(8)  # parity subsequences have length four: 16 phase pairs
    selected = select_fkm_content_seed(
        profile, seed=5, pool_size=8, policy="phase_top_q",
        candidate_count=100, elite_count=20,
    )
    diagnostics = selected.diagnostics
    assert diagnostics.candidates_examined == 16
    assert diagnostics.elite_count == 16
    assert len(set(diagnostics.ranked_phases)) == 16


@pytest.mark.parametrize("length,expected_factors", ((8, (2, 4)), (14, (2,))))
def test_energy_ranking_uses_full_then_required_compression_tiebreaks(
    length, expected_factors,
):
    profile = _profile(length)
    selected = select_fkm_content_seed(
        profile, seed=2026, pool_size=16, policy="phase_top_q",
        candidate_count=12, elite_count=4,
    )
    diagnostics = selected.diagnostics
    assert diagnostics.compression_factors == expected_factors
    assert diagnostics.ranked_energy_keys == tuple(sorted(
        diagnostics.ranked_energy_keys
    ))
    assert all(
        len(key) == 1 + len(expected_factors)
        for key in diagnostics.ranked_energy_keys
    )
    observed = full_correlation_profile(selected.a, selected.b)
    breakdown = structured_energy_breakdown(
        observed, profile.k, profile.eta, expected_factors
    )
    expected_key = (breakdown.full,) + tuple(
        breakdown.component(factor) for factor in expected_factors
    )
    assert diagnostics.selected_energy_key == expected_key
    assert seed_energy_key(selected.a, selected.b, profile) == expected_key


def test_diagnostics_are_recomputed_from_selected_complete_pair():
    profile = _profile(44)
    selected = select_fkm_content_seed(
        profile, seed=77, pool_size=32, policy="phase_top_q",
        candidate_count=16, elite_count=4,
    )
    observed = full_correlation_profile(selected.a, selected.b)
    assert selected.diagnostics.selected_objective == pqcp_objective(observed)
    assert selected.diagnostics.selected_energy_key == seed_energy_key(
        selected.a, selected.b, profile
    )
    assert 1 <= selected.diagnostics.unique_profiles <= 16


@pytest.mark.parametrize(
    "kwargs,exception",
    (
        ({"seed": True}, ValueError),
        ({"pool_size": 0}, ValueError),
        ({"policy": "unknown"}, ValueError),
        ({"candidate_count": 0}, ValueError),
        ({"elite_count": 0}, ValueError),
        ({"policy": "phase_top_q", "candidate_count": 2, "elite_count": 3}, ValueError),
    ),
)
def test_invalid_selection_controls_are_rejected(kwargs, exception):
    arguments = dict(
        profile=_profile(8), seed=1, pool_size=8, policy="legacy",
        candidate_count=4, elite_count=2,
    )
    arguments.update(kwargs)
    with pytest.raises(exception):
        select_fkm_content_seed(**arguments)


def test_invalid_target_profile_is_rejected_before_generation():
    invalid = TargetContentProfile(8, 4, 1, 1, 1, 1, 1)
    with pytest.raises(ValueError, match="invalid Project target"):
        select_fkm_content_seed(invalid, seed=1)
