"""Literal PACF oracles, independent of polynomial/finite-field arithmetic."""
from collections import defaultdict

import pytest

from solver.cyclic_support import iter_family, parity_support_family, support_ring, rotation_representatives
from solver.cyclic_support_native import nearest_supports


def literal_signature(mask, d):
    bits = [(mask >> i) & 1 for i in range(d)]
    return tuple(sum(bits[i]*bits[(i+u) % d] for i in range(d)) % 2
                 for u in range(d//2+1))


@pytest.mark.parametrize('d', [3, 5, 11])
def test_all_norm_families_equal_every_literal_support(d):
    index = defaultdict(set)
    signatures = [literal_signature(mask, d) for mask in range(1 << d)]
    for mask, signature in enumerate(signatures):
        index[signature].add(mask)
    for u, signature in enumerate(signatures):
        for k in range(1, d//2+1):
            desired = tuple(bit ^ int(shift in (0, k))
                            for shift, bit in enumerate(signature))
            family = parity_support_family(u, d, k)
            masks = list(iter_family(family))
            assert len(masks) == len(set(masks)) == family.size
            assert set(masks) == index[desired]


def test_l43_factorization_and_large_family_samples():
    ring = support_ring(43)
    assert ring.degree == 14
    assert ring.factors == (3, 0x4ff9, 0x5495, 0x688b)
    for u in (0, (1 << 43)-1, 0x51819324135):
        for k in (1, 2, 21):
            family = parity_support_family(u, 43, k)
            assert family.size <= 129**3
            # Widely separated phases in all three factors.
            for phase in (0, 1, 42, 85, 128):
                v = family.base
                for group in family.choices:
                    v ^= group[phase % len(group)]
                a, b = literal_signature(u, 43), literal_signature(v, 43)
                assert tuple(x ^ y for x, y in zip(a, b)) == tuple(
                    int(shift in (0, k)) for shift in range(22))


@pytest.mark.parametrize('d', [3, 5, 11])
def test_native_nearest_weight_filter_equals_literal_enumeration(d):
    for u in range(1 << d) if d < 11 else range(0, 1 << d, 127):
        for k in range(1, d//2+1):
            family = parity_support_family(u, d, k)
            for weight in range(d+1):
                valid = [v for v in range(1 << d) if bin(v).count('1') == weight
                         and tuple(x ^ y for x, y in zip(literal_signature(u, d),
                                   literal_signature(v, d))) == tuple(
                                       int(shift in (0, k)) for shift in range(d//2+1))]
                reference = u ^ ((1 << d)-1)
                distance = min((bin(v ^ reference).count('1') for v in valid), default=None)
                expected = {v for v in valid if bin(v ^ reference).count('1') == distance}
                for capacity in (1, 129):
                    result = nearest_supports(family, reference, weight, capacity=capacity)
                    assert result['distance'] == distance
                    assert result['weight_matches'] == len(valid)
                    assert result['nearest_ties'] == len(expected)
                    assert result['complete_nearest'] == (len(expected) <= capacity)
                    assert set(result['masks']) <= expected
                    assert len(result['masks']) == min(capacity, len(expected))


def test_l43_native_full_enumeration_retains_exact_reference():
    family = parity_support_family(0x51819324135, 43, 2)
    reference = next(iter_family(family))
    result = nearest_supports(family, reference, bin(reference).count('1'), capacity=1)
    assert result['family_size'] == 129**3
    assert result['distance'] == 0
    assert result['masks'] == (reference,)
    assert result['nearest_ties'] == 1
    assert result['complete_nearest']


@pytest.mark.parametrize('d', [3, 5, 11])
@pytest.mark.parametrize('vector_tail', [False, True])
def test_distance_floor_selects_next_layer_without_changing_weight_counts(d, vector_tail):
    family = parity_support_family((1 << (d//2))-1, d, 1)
    if vector_tail:
        from dataclasses import replace
        # Put the nontrivial group last, exercising ARM's four-word batches
        # and scalar tail even on these fully enumerable small fields.
        family = replace(family, choices=((0,), (0,), family.choices[0]))
    reference = (1 << d)-1
    literal = list(iter_family(family))
    for weight in range(d+1):
        all_valid = [v for v in literal if bin(v).count('1') == weight]
        for floor in range(d+1):
            valid = [v for v in all_valid if bin(v ^ reference).count('1') >= floor]
            minimum = min((bin(v ^ reference).count('1') for v in valid), default=None)
            expected = {v for v in valid if bin(v ^ reference).count('1') == minimum}
            out = nearest_supports(family, reference, weight, capacity=129, minimum_distance=floor)
            assert out['weight_matches'] == len(all_valid)
            assert out['distance'] == minimum
            assert out['nearest_ties'] == len(expected)
            assert set(out['masks']) == expected


@pytest.mark.parametrize('d', [3, 5, 11])
def test_rotation_quotient_contains_exactly_one_member_of_every_literal_orbit(d):
    full = (1 << d)-1
    for u in range(1 << d):
        for k in range(1, d//2+1):
            family = parity_support_family(u, d, k)
            reps = list(iter_family(rotation_representatives(family)))
            expanded = set()
            for v in reps:
                orbit = {v} | {((v << s) | (v >> (d-s))) & full for s in range(1, d)}
                assert not expanded & orbit
                expanded |= orbit
            assert expanded == set(iter_family(family))


def test_l43_rotation_quotient_has_43_times_fewer_phases():
    family = parity_support_family(0x51819324135, 43, 2)
    reduced = rotation_representatives(family)
    assert family.size == 129**3
    assert reduced.size == 3*129**2


@pytest.mark.parametrize('d', [True, 7, 9, 47, 86])
def test_unsupported_rings_are_rejected(d):
    with pytest.raises(ValueError):
        support_ring(d)


@pytest.mark.parametrize('u,k', [(-1, 1), (32, 1), (True, 1), (1, 0), (1, 3), (1, True)])
def test_invalid_supports_or_shifts_are_rejected(u, k):
    with pytest.raises(ValueError):
        parity_support_family(u, 5, k)
