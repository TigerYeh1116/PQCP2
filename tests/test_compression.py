"""Randomized exactness tests for incremental correlation updates."""

import random

import pytest

from solver.compression import CorrelationState
from solver.correlation import full_correlation_profile
from solver.objective import pqcp_objective, target_pair_squared_energy


LENGTHS = (1, 2, 3, 4, 5, 8, 16, 44)


def _random_bits(rng, length):
    """Generate deterministic-test random binary bits."""
    return [rng.randrange(2) for _ in range(length)]


@pytest.mark.parametrize("length", LENGTHS)
def test_incremental_updates_match_baseline_after_many_random_flips(length):
    rng = random.Random(20_260_000 + length)
    for _ in range(6):
        a = _random_bits(rng, length)
        b = _random_bits(rng, length)
        state = CorrelationState(a, b)
        assert state.profile == tuple(full_correlation_profile(a, b))
        assert state.score == pqcp_objective(state.profile)
        assert state.target_pair_energy == target_pair_squared_energy(state.profile)

        for _ in range(24):
            position = rng.randrange(length)
            if rng.randrange(2) == 0:
                a[position] ^= 1
                state.flip_a(position)
            else:
                b[position] ^= 1
                state.flip_b(position)
            assert state.a == tuple(a)
            assert state.b == tuple(b)
            assert state.profile == tuple(full_correlation_profile(a, b))
            assert state.score == pqcp_objective(state.profile)
            assert state.target_pair_energy == target_pair_squared_energy(state.profile)


def test_repeated_flips_at_same_position_restore_state_exactly():
    a = [0, 1, 1, 0, 1, 0, 0, 1]
    b = [1, 0, 0, 1, 0, 1, 1, 0]
    state = CorrelationState(a, b)
    original = state.profile

    for _ in range(4):
        a[3] ^= 1
        state.flip_a(3)
        assert state.profile == tuple(full_correlation_profile(a, b))

    assert state.a == tuple([0, 1, 1, 0, 1, 0, 0, 1])
    assert state.profile == original


def test_binary_tuple_cache_invalidates_only_the_flipped_sequence():
    state = CorrelationState((0, 1, 0, 1), (1, 1, 0, 0))
    original_a = state.a
    original_b = state.b

    state.flip_a(2)
    assert state.a == (0, 1, 1, 1)
    assert state.a is state.a
    assert state.b is original_b

    cached_a = state.a
    state.flip_b(0)
    assert state.a is cached_a
    assert state.b == (0, 1, 0, 0)
    assert state.b is state.b
    assert original_a == (0, 1, 0, 1)


def test_constructor_and_flip_positions_validate_inputs():
    with pytest.raises(ValueError):
        CorrelationState((0, 1), (0, 1, 0))
    with pytest.raises(ValueError):
        CorrelationState((0, 2), (0, 1))

    state = CorrelationState((0, 1), (1, 0))
    for position in (-1, 2, True):
        with pytest.raises(ValueError):
            state.flip_a(position)
