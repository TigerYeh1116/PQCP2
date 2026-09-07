"""Exact tests for the safe total-autocorrelation Hamming-weight condition."""

import pytest

from solver.weight_constraints import admissible_weight_pairs, canonical_weight_pairs, is_admissible_weight_pair


def test_project_weight_classes_match_the_exact_square_identity():
    assert canonical_weight_pairs(44) == ((18, 20),)
    assert canonical_weight_pairs(46) == ((18, 23), (19, 20))
    assert canonical_weight_pairs(68) == ((28, 34), (30, 30))
    assert canonical_weight_pairs(86) == ((37, 40), (38, 39))
    assert canonical_weight_pairs(94) == ((40, 47), (41, 44))


@pytest.mark.parametrize("length", (58, 90))
def test_lengths_without_integer_weight_pairs_are_proven_impossible_by_this_condition(length):
    assert admissible_weight_pairs(length) == ()


def test_ordered_pairs_include_existing_l44_weight_pair_and_its_swap():
    assert is_admissible_weight_pair(44, 18, 20)
    assert is_admissible_weight_pair(44, 20, 18)


def test_weight_condition_requires_positive_even_length():
    with pytest.raises(ValueError):
        canonical_weight_pairs(45)
