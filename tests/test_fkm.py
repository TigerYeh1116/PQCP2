"""Brute-force checked tests for FKM binary cyclic representatives."""

from itertools import product

import pytest

from solver.fkm import generate_fkm_sequences, is_cyclic_representative


def _canonical_rotation(word):
    """Return the lexicographically least rotation of a small binary word."""
    return min(word[offset:] + word[:offset] for offset in range(len(word)))


def _brute_force_representatives(length, weight=None):
    """Classify all small binary words into cyclic orbits for test comparison."""
    words = product((0, 1), repeat=length)
    if weight is not None:
        words = (word for word in words if sum(word) == weight)
    return {_canonical_rotation(word) for word in words}


@pytest.mark.parametrize("length", [1, 2, 3, 4, 5, 6])
def test_fkm_matches_brute_force_cyclic_orbits(length):
    generated = set(generate_fkm_sequences(length))
    expected = _brute_force_representatives(length)
    assert generated == expected
    assert all(len(word) == length and set(word) <= {0, 1} for word in generated)
    assert all(is_cyclic_representative(word) for word in generated)


@pytest.mark.parametrize("length, weight", [(4, 0), (4, 2), (5, 2), (6, 3)])
def test_weighted_fkm_matches_brute_force_cyclic_orbits(length, weight):
    generated = list(generate_fkm_sequences(length, weight=weight))
    expected = _brute_force_representatives(length, weight=weight)
    assert set(generated) == expected
    assert len(generated) == len(set(generated))
    assert all(sum(word) == weight for word in generated)


def test_limit_yields_only_requested_prefix():
    all_candidates = list(generate_fkm_sequences(6))
    assert list(generate_fkm_sequences(6, limit=3)) == all_candidates[:3]
    assert list(generate_fkm_sequences(6, limit=0)) == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"L": 0},
        {"L": True},
        {"L": 4, "weight": -1},
        {"L": 4, "weight": 5},
        {"L": 4, "limit": -1},
    ],
)
def test_invalid_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        generate_fkm_sequences(**kwargs)


def test_representative_helper_rejects_invalid_words():
    with pytest.raises(ValueError):
        is_cyclic_representative(())
    with pytest.raises(ValueError):
        is_cyclic_representative((0, 2))
