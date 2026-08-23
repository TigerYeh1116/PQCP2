"""Tests for small, hand-computed periodic correlations."""

import pytest

from solver.correlation import (
    full_correlation_profile,
    pair_autocorrelation_sum,
    periodic_autocorrelation,
)


def test_periodic_autocorrelation_small_hand_case():
    # For 0101, equal bit pairs contribute +1 and unequal pairs contribute -1.
    sequence = (0, 1, 0, 1)
    assert [periodic_autocorrelation(sequence, shift) for shift in range(4)] == [4, -4, 4, -4]


def test_pair_profile_small_hand_case():
    a = (0, 1, 0, 1)
    b = (0, 0, 1, 1)
    assert pair_autocorrelation_sum(a, b, 1) == -4
    assert full_correlation_profile(a, b) == [8, -4, 0, -4]


@pytest.mark.parametrize("shift", [-1, 4, True])
def test_periodic_autocorrelation_rejects_noncanonical_shift(shift):
    with pytest.raises(ValueError):
        periodic_autocorrelation((0, 1, 0, 1), shift)


def test_correlation_rejects_nonbinary_or_unequal_inputs():
    with pytest.raises(ValueError):
        periodic_autocorrelation((0, 2), 0)
    with pytest.raises(ValueError):
        pair_autocorrelation_sum((0, 1), (0, 1, 0), 0)
