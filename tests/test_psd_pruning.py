"""Safety and integration tests for certified general-frequency PSD pruning."""

import cmath
from itertools import product
import random

import pytest

from solver.compressed_pairing import (
    compressed_pair_matches_target,
    pair_by_compressed_signature,
)
from solver.psd_pruning import (
    certified_binary_psd_pruning,
    certified_psd_pruning_from_profiles,
)
from solver.structured_energy import (
    integer_periodic_autocorrelation_profile,
    target_profile,
)
from solver.target_profiles import (
    TargetContentProfile,
    canonical_target_content_profiles,
    pair_content,
)
from solver.verifier import verify_pqcp


KNOWN_L44_A = tuple(map(int, "10001001110010011010011110101000011000010000"))
KNOWN_L44_B = tuple(map(int, "11000000111101111000001011001010101101000100"))


def _target_for_verified_pair(a, b):
    check = verify_pqcp(a, b)
    assert check.is_valid
    shift = min(check.nonzero_shifts)
    return TargetContentProfile(
        len(a), shift, check.profile[shift] // 4, *pair_content(a, b)
    )


def _individual_pacf(bits):
    signs = tuple(1 if bit == 0 else -1 for bit in bits)
    return integer_periodic_autocorrelation_profile(signs)


def _dft(profile, frequency):
    length = len(profile)
    return sum(
        value * cmath.exp(-2j * cmath.pi * frequency * shift / length)
        for shift, value in enumerate(profile)
    ).real


def test_known_project_solution_survives_every_general_frequency():
    target = _target_for_verified_pair(KNOWN_L44_A, KNOWN_L44_B)
    left = certified_binary_psd_pruning(KNOWN_L44_A, target)
    right = certified_binary_psd_pruning(KNOWN_L44_B, target)
    assert not left.should_prune
    assert not right.should_prune
    assert not left.violating_frequencies
    assert not right.violating_frequencies

    matches = pair_by_compressed_signature(
        (KNOWN_L44_A,), (KNOWN_L44_B,), target,
        general_psd_pruning=True,
    )
    assert [(match.a, match.b) for match in matches] == [
        (KNOWN_L44_A, KNOWN_L44_B)
    ]


@pytest.mark.parametrize("length", (4, 6, 8))
def test_exhaustive_small_verified_pairs_are_never_pruned(length):
    words = tuple(product((0, 1), repeat=length))
    found = 0
    for a in words:
        for b in words:
            check = verify_pqcp(a, b)
            if not check.is_valid:
                continue
            found += 1
            target = _target_for_verified_pair(a, b)
            assert not certified_binary_psd_pruning(a, target).should_prune
            assert not certified_binary_psd_pruning(b, target).should_prune
    # Some very small lengths may have no Project target; the cross-check is
    # still exhaustive.  L=4 is kept as a positive guard against vacuity.
    if length == 4:
        assert found > 0


def test_irrational_frequency_violation_is_certified_not_rounded():
    target = TargetContentProfile(8, 2, -1, 1, 2, 1, 2)
    candidate = (0, 0, 0, 0, 0, 1, 1, 1)
    result = certified_binary_psd_pruning(candidate, target)
    assert result.should_prune
    assert 1 in result.violating_frequencies  # cos(pi/4), not a rational character


def test_every_certified_random_rejection_has_a_strict_high_precision_gap():
    rng = random.Random(20260912)
    for length in (8, 12, 14, 16):
        target = canonical_target_content_profiles(length)[0]
        desired = target_profile(length, target.k, target.eta)
        for _ in range(25):
            bits = tuple(rng.randrange(2) for _ in range(length))
            individual = _individual_pacf(bits)
            result = certified_binary_psd_pruning(bits, target)
            for frequency in result.violating_frequencies:
                gap = _dft(desired, frequency) - _dft(individual, frequency)
                assert gap < -1e-8


def test_compressed_hash_join_can_enable_safe_general_psd_filter():
    target = canonical_target_content_profiles(8)[0]
    words = tuple(product((0, 1), repeat=8))
    a_pool = tuple(word for word in words if pair_content(word, (0,) * 8)[:2] == (
        target.a_even_ones, target.a_odd_ones
    ))
    b_pool = tuple(word for word in words if pair_content((0,) * 8, word)[2:] == (
        target.b_even_ones, target.b_odd_ones
    ))
    actual = pair_by_compressed_signature(
        a_pool, b_pool, target, general_psd_pruning=True
    )
    expected = {
        (a, b) for a in a_pool for b in b_pool
        if not certified_binary_psd_pruning(a, target).should_prune
        and not certified_binary_psd_pruning(b, target).should_prune
        and compressed_pair_matches_target(a, b, target)
    }
    assert {(match.a, match.b) for match in actual} == expected


@pytest.mark.parametrize(
    "individual,target,kwargs",
    (
        ((1, 0), (1,), {}),
        ((1, 2, 0), (1, 0, 0), {}),
        ((1,), (1,), {"frequencies": (1,)}),
        ((1,), (1,), {"precisions": (10,)}),
    ),
)
def test_invalid_profile_or_control_is_rejected(individual, target, kwargs):
    with pytest.raises(ValueError):
        certified_psd_pruning_from_profiles(individual, target, **kwargs)
