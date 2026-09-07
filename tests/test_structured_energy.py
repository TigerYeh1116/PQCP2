"""Tests for fixed-target and multiscale exact residual energies."""

import random

import pytest

from solver.correlation import full_correlation_profile
from solver.structured_energy import (
    compress_binary_sequence,
    compressed_error_energy,
    compressed_pair_profile,
    fold_profile,
    full_error_energy,
    multiscale_error_energy,
    residual_profile,
    structured_energy_breakdown,
    target_profile,
)


def test_exact_target_has_zero_at_every_energy_level():
    profile = target_profile(44, 4, -1)
    assert residual_profile(profile, 4, -1) == (0,) * 44
    assert full_error_energy(profile, 4, -1) == 0
    assert compressed_error_energy(profile, 4, -1, 2) == 0
    assert compressed_error_energy(profile, 4, -1, 4) == 0
    assert multiscale_error_energy(profile, 4, -1, {2: 2, 4: 4}) == 0


def test_wrong_target_position_or_sign_has_positive_full_energy():
    profile = target_profile(44, 4, -1)
    assert full_error_energy(profile, 2, -1) > 0
    assert full_error_energy(profile, 4, 1) > 0


def test_multiscale_formula_uses_explicit_weights():
    profile = list(target_profile(8, 1, 1))
    profile[2] = profile[6] = 4
    full = full_error_energy(profile, 1, 1)
    e2 = compressed_error_energy(profile, 1, 1, 2)
    e4 = compressed_error_energy(profile, 1, 1, 4)
    assert multiscale_error_energy(profile, 1, 1, {2: 2, 4: 4}) == full + 2 * e2 + 4 * e4


def test_invalid_compression_factor_is_rejected():
    with pytest.raises(ValueError):
        compressed_error_energy(target_profile(10, 1, 1), 1, 1, 4)


@pytest.mark.parametrize("length,factors", ((8, (2, 4)), (44, (2, 4)), (46, (2,)), (68, (2, 4))))
def test_direct_sequence_compression_equals_exact_pacf_folding(length, factors):
    """Cross-check the literature compression identity from actual sequences."""
    for seed in range(12):
        rng = random.Random(10_000 * length + seed)
        a = tuple(rng.randrange(2) for _ in range(length))
        b = tuple(rng.randrange(2) for _ in range(length))
        profile = full_correlation_profile(a, b)
        for factor in factors:
            assert compressed_pair_profile(a, b, factor) == fold_profile(profile, factor)


def test_factor_two_compressed_symbols_use_opposite_half_positions():
    bits = (0, 0, 1, 1, 0, 1, 1, 0)
    # sign pairs are (1+1), (1-1), (-1-1), (-1+1)
    assert compress_binary_sequence(bits, 2) == (2, 0, -2, 0)


def test_target_collision_is_preserved_by_residual_first_folding():
    # L=8, factor 2 gives d=4.  k=2 and L-k=6 collide at compressed shift 2.
    target = target_profile(8, 2, 1)
    assert fold_profile(target, 2) == (16, 0, 8, 0)
    # factor 4 gives d=2 and both target sidelobes fold to the origin.
    assert fold_profile(target, 4) == (24, 0)


def test_compressed_zero_does_not_imply_full_target():
    profile = list(target_profile(8, 1, 1))
    # Symmetric residuals cancel in each factor-two class.
    profile[1] += 4
    profile[7] += 4
    profile[3] -= 4
    profile[5] -= 4
    assert full_error_energy(profile, 1, 1) > 0
    assert compressed_error_energy(profile, 1, 1, 2) == 0


def test_structured_breakdown_is_scaled_exactly_and_weighted_explicitly():
    profile = list(target_profile(44, 4, -1))
    profile[2] = profile[42] = 8
    breakdown = structured_energy_breakdown(profile, 4, -1, (4, 2, 2))
    assert breakdown.full == full_error_energy(profile, 4, -1) // 16
    assert breakdown.component(2) == compressed_error_energy(profile, 4, -1, 2) // 16
    assert breakdown.component(4) == compressed_error_energy(profile, 4, -1, 4) // 16
    assert breakdown.weighted_total({2: 2, 4: 4}) == (
        multiscale_error_energy(profile, 4, -1, {2: 2, 4: 4}) // 16
    )
