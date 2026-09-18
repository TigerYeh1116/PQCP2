"""Exhaustive checks for the exact self-symmetric-shift seed condition."""

from itertools import product
import random

import pytest

from solver.correlation import full_correlation_profile
from solver.frequency_constraints import quarter_frequency_feasible
from solver.half_shift import (
    construct_half_shift_zero_partner,
    half_shift_mismatch_count,
)
from solver.target_profiles import canonical_target_content_profiles, pair_content


@pytest.mark.parametrize("length", (4, 6, 8))
def test_half_shift_identity_for_every_binary_word(length):
    half = length // 2
    for bits in product((0, 1), repeat=length):
        signs = tuple(1 - 2 * bit for bit in bits)
        rho = sum(signs[index] * signs[(index + half) % length]
                  for index in range(length))
        assert rho == length - 4 * half_shift_mismatch_count(bits)


@pytest.mark.parametrize("length", (4, 6, 8))
def test_partner_constructor_matches_complete_feasibility_enumeration(length):
    """None/non-None must agree with every possible exact-content B word."""
    sequences = tuple(product((0, 1), repeat=length))
    half = length // 2
    by_content = {
        (even, odd): tuple(
            bits for bits in sequences
            if sum(bits[0::2]) == even and sum(bits[1::2]) == odd
        )
        for even in range(half + 1)
        for odd in range(half + 1)
    }
    for a_index, a in enumerate(sequences):
        required = half - half_shift_mismatch_count(a)
        for (even, odd), candidates in by_content.items():
            expected = any(
                half_shift_mismatch_count(b) == required for b in candidates
            )
            actual = construct_half_shift_zero_partner(
                a, even, odd,
                random.Random(a_index * 10_000 + even * 100 + odd),
            )
            assert (actual is not None) == expected
            if actual is not None:
                assert (sum(actual[0::2]), sum(actual[1::2])) == (even, odd)
                assert full_correlation_profile(a, actual)[half] == 0


def test_l68_production_profiles_initialize_with_all_safe_conditions(tmp_path):
    """Every retained L=68 content orbit can produce an exact half-zero seed."""
    torch = pytest.importorskip("torch")
    from solver.torch_search import TorchSearch, TorchSearchConfig

    retained = tuple(
        profile for profile in canonical_target_content_profiles(68)
        if quarter_frequency_feasible(profile)
    )
    search = TorchSearch(TorchSearchConfig(
        68,
        device="cpu",
        batch_size=len(retained),
        fkm_pool_size=8,
        frequency_pruning=True,
        symmetry_pruning=True,
    ), tmp_path)
    bits = search._initial_pairs(retained, generation=17).to(dtype=torch.int64)
    assert len(retained) == len(search.available) == 8
    for index, profile in enumerate(retained):
        a = tuple(bits[index, 0].tolist())
        b = tuple(bits[index, 1].tolist())
        assert pair_content(a, b) == (
            profile.a_even_ones, profile.a_odd_ones,
            profile.b_even_ones, profile.b_odd_ones,
        )
        assert full_correlation_profile(a, b)[34] == 0
