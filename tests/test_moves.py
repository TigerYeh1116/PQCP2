"""Unit tests for exact fixed-content search moves."""

import random

from solver.compression import CorrelationState
from solver.correlation import full_correlation_profile
from solver.moves import (
    WeightPreservingSwap,
    apply_weight_preserving_swap,
    choose_weight_preserving_swap,
    rollback_weight_preserving_swap,
    sample_weight_preserving_swaps,
    trial_fixed_target_swap_energy,
    trial_fixed_target_swap_energy_key,
    trial_multiscale_fixed_target_swap_energy,
    trial_weight_preserving_swap,
)
from solver.structured_energy import multiscale_error_energy, structured_energy_breakdown
from solver.target_profiles import pair_content


def test_apply_and_rollback_preserve_weights_and_exact_profile():
    state = CorrelationState((0, 1, 0, 1, 0, 1), (1, 0, 0, 1, 1, 0))
    before = (state.a, state.b, state.profile, state.score)
    weights = (sum(state.a), sum(state.b))
    move = choose_weight_preserving_swap(state.a, state.b, random.Random(12))
    assert move is not None
    apply_weight_preserving_swap(state, move)
    assert (sum(state.a), sum(state.b)) == weights
    assert state.profile == tuple(full_correlation_profile(state.a, state.b))
    rollback_weight_preserving_swap(state, move)
    assert (state.a, state.b, state.profile, state.score) == before


def test_uniform_pair_has_no_weight_preserving_move():
    assert choose_weight_preserving_swap((0, 0), (1, 1), random.Random(1)) is None


def test_batched_single_sample_matches_historical_single_choice():
    a, b = (0, 1, 0, 1), (1, 0, 0, 1)
    for seed in range(20):
        expected = choose_weight_preserving_swap(a, b, random.Random(seed))
        actual = sample_weight_preserving_swaps(a, b, random.Random(seed), 1)
        assert actual == (expected,)


def test_batched_samples_are_legal_and_validate_count():
    a, b = (0, 1, 0, 1), (1, 0, 0, 1)
    moves = sample_weight_preserving_swaps(a, b, random.Random(9), 25)
    assert len(moves) == 25
    for move in moves:
        values = a if move.sequence == "a" else b
        assert values[move.zero_position] == 0
        assert values[move.one_position] == 1


def test_same_parity_swaps_preserve_ordinary_and_alternating_content():
    a = (0, 1, 1, 0, 0, 1, 1, 0)
    b = (1, 0, 0, 1, 1, 0, 0, 1)
    state = CorrelationState(a, b)
    expected = pair_content(a, b)
    for move in sample_weight_preserving_swaps(a, b, random.Random(91), 100, same_parity=True):
        assert move.zero_position % 2 == move.one_position % 2
        apply_weight_preserving_swap(state, move)
        assert pair_content(state.a, state.b) == expected
        rollback_weight_preserving_swap(state, move)


def test_trial_swap_is_exact_and_nonmutating_for_random_sequences():
    """Cross-check the analytical O(L) trial against actual incremental updates."""
    for length in (2, 3, 4, 5, 8, 16, 44):
        for seed in range(8):
            rng = random.Random(1000 * length + seed)
            a = tuple(rng.randrange(2) for _ in range(length))
            b = tuple(rng.randrange(2) for _ in range(length))
            state = CorrelationState(a, b)
            before = (state.a, state.b, state.profile, state.score, state.target_pair_energy)
            for name, values in (("a", a), ("b", b)):
                zeros = [index for index, value in enumerate(values) if value == 0]
                ones = [index for index, value in enumerate(values) if value == 1]
                for zero in zeros:
                    for one in ones:
                        move = WeightPreservingSwap(name, zero, one)
                        evaluation = trial_weight_preserving_swap(state, move)
                        assert (state.a, state.b, state.profile, state.score, state.target_pair_energy) == before
                        apply_weight_preserving_swap(state, move)
                        assert evaluation.score == state.score
                        assert evaluation.target_pair_energy == state.target_pair_energy
                        assert evaluation.profile == state.profile
                        assert state.profile == tuple(full_correlation_profile(state.a, state.b))
                        rollback_weight_preserving_swap(state, move)
                        assert (state.a, state.b, state.profile, state.score, state.target_pair_energy) == before


def test_fixed_target_trial_energy_matches_applied_profile_exactly():
    """The optimized production rank must equal the original full scan."""
    for length in (4, 8, 16, 44):
        for seed in range(6):
            rng = random.Random(7000 * length + seed)
            a = tuple(rng.randrange(2) for _ in range(length))
            b = tuple(rng.randrange(2) for _ in range(length))
            state = CorrelationState(a, b)
            moves = sample_weight_preserving_swaps(a, b, rng, 20)
            for move in moves:
                target_shift = rng.randrange(1, length // 2)
                target_value = rng.choice((-4, 4))
                before = (state.a, state.b, state.profile, state.score)
                actual = trial_fixed_target_swap_energy(
                    state, move, target_shift, target_value,
                )
                assert (state.a, state.b, state.profile, state.score) == before
                apply_weight_preserving_swap(state, move)
                expected = sum(
                    (state.profile[shift] - (
                        target_value if shift == target_shift else 0
                    )) ** 2
                    for shift in range(1, length // 2 + 1)
                ) // 16
                assert actual == expected
                rollback_weight_preserving_swap(state, move)
                assert (state.a, state.b, state.profile, state.score) == before


def test_multiscale_trial_energy_matches_direct_sequence_compression_exactly():
    """The fused hot path must equal full residual folding after every swap."""
    for length in (8, 44, 46, 68):
        weights = {2: 2}
        if length % 4 == 0:
            weights[4] = 4
        for seed in range(5):
            rng = random.Random(90_000 * length + seed)
            a = tuple(rng.randrange(2) for _ in range(length))
            b = tuple(rng.randrange(2) for _ in range(length))
            state = CorrelationState(a, b)
            for move in sample_weight_preserving_swaps(a, b, rng, 16):
                target_shift = rng.randrange(1, length // 2)
                eta = rng.choice((-1, 1))
                before = (state.a, state.b, state.profile, state.score)
                actual = trial_multiscale_fixed_target_swap_energy(
                    state, move, target_shift, 4 * eta, weights,
                )
                assert (state.a, state.b, state.profile, state.score) == before
                apply_weight_preserving_swap(state, move)
                expected = multiscale_error_energy(
                    state.profile, target_shift, eta, weights,
                ) // 16
                assert actual == expected
                rollback_weight_preserving_swap(state, move)
                assert (state.a, state.b, state.profile, state.score) == before


def test_fixed_target_energy_key_matches_full_and_compressed_recomputation():
    for length in (44, 46):
        factors = (2, 4) if length % 4 == 0 else (2,)
        rng = random.Random(123_000 + length)
        a = tuple(rng.randrange(2) for _ in range(length))
        b = tuple(rng.randrange(2) for _ in range(length))
        state = CorrelationState(a, b)
        for move in sample_weight_preserving_swaps(a, b, rng, 20):
            shift = rng.randrange(1, length // 2)
            eta = rng.choice((-1, 1))
            key = trial_fixed_target_swap_energy_key(
                state, move, shift, 4 * eta, factors,
            )
            apply_weight_preserving_swap(state, move)
            expected = structured_energy_breakdown(state.profile, shift, eta, factors)
            assert key == (expected.full,) + tuple(
                expected.component(factor) for factor in factors
            )
            rollback_weight_preserving_swap(state, move)
