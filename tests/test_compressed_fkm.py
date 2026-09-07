"""Brute-force and structural tests for genuine factor-two FKM lifts."""

from itertools import product

import pytest

from solver.compressed_fkm import (
    compressed_fkm_lift_candidates,
    factor2_compressed_contents,
    generate_compressed_fkm_sequences,
    lift_factor2_compressed,
)
from solver.structured_energy import compress_binary_sequence
from solver.target_profiles import canonical_target_content_profiles, pair_content


def _canonical_rotation(word):
    return min(word[offset:] + word[:offset] for offset in range(len(word)))


@pytest.mark.parametrize(
    "content",
    ((1, 0, 0), (0, 2, 1), (1, 2, 1), (2, 1, 2)),
)
def test_ternary_fixed_content_fkm_matches_brute_force_orbits(content):
    alphabet = (-2, 0, 2)
    length = sum(content)
    expected = {
        _canonical_rotation(word)
        for word in product(alphabet, repeat=length)
        if tuple(word.count(value) for value in alphabet) == content
    }
    actual = tuple(generate_compressed_fkm_sequences(content))
    assert set(actual) == expected
    assert len(actual) == len(set(actual))


def test_ternary_fkm_limit_is_lazy_prefix():
    complete = tuple(generate_compressed_fkm_sequences((2, 2, 2)))
    assert tuple(generate_compressed_fkm_sequences((2, 2, 2), limit=3)) == complete[:3]
    assert tuple(generate_compressed_fkm_sequences((2, 2, 2), limit=0)) == ()


def test_factor_two_lift_uses_opposite_halves_not_adjacent_positions():
    compressed = (-2, 0, 2)
    lifted = lift_factor2_compressed(compressed, (0,))
    assert lifted == (1, 0, 0, 1, 1, 0)
    assert compress_binary_sequence(lifted, 2) == compressed


@pytest.mark.parametrize("length", (44, 46))
def test_compressed_fkm_candidates_are_deterministic_and_preserve_content(length):
    target = canonical_target_content_profiles(length)[0]
    kwargs = dict(
        length=length,
        even_ones=target.a_even_ones,
        odd_ones=target.a_odd_ones,
        seed=20260906,
        limit=12,
        max_necklaces=512,
    )
    first = compressed_fkm_lift_candidates(**kwargs)
    second = compressed_fkm_lift_candidates(**kwargs)
    assert first == second
    assert len(first) == 12
    assert len(first) == len(set(first))
    for candidate in first:
        assert len(candidate) == length
        assert (sum(candidate[0::2]), sum(candidate[1::2])) == (
            target.a_even_ones, target.a_odd_ones,
        )
        assert set(compress_binary_sequence(candidate, 2)) <= {-2, 0, 2}


@pytest.mark.parametrize("length", (8, 14, 44, 46))
def test_content_enumeration_is_exact_for_every_returned_triple(length):
    target = canonical_target_content_profiles(length)[0]
    contents = factor2_compressed_contents(
        length, target.a_even_ones, target.a_odd_ones
    )
    assert contents
    distance = length // 2
    weight = target.a_even_ones + target.a_odd_ones
    assert all(sum(content) == distance for content in contents)
    assert all(2 * negative + zero == weight for negative, zero, _ in contents)


@pytest.mark.parametrize(
    "call",
    (
        lambda: tuple(generate_compressed_fkm_sequences((1, -1, 1))),
        lambda: tuple(generate_compressed_fkm_sequences((0, 0, 0))),
        lambda: lift_factor2_compressed((2, 0), ()),
        lambda: compressed_fkm_lift_candidates(5, 1, 1, seed=1),
        lambda: compressed_fkm_lift_candidates(8, 9, 1, seed=1),
    ),
)
def test_invalid_inputs_are_rejected(call):
    with pytest.raises(ValueError):
        call()
