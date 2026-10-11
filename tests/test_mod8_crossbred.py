from random import Random

import pytest

from solver.mod8_crossbred import (
    affine_tail_indices, boolean_equations, degree_three_tail_consequences,
    replay_certificate, specialize_tail_system,
)
from solver.mps_mod8_prefix import evaluate


def coefficients(q, rng):
    linear = tuple(rng.randrange(256) for _ in range(q))
    quadratic = [0]*(q*q)
    for i in range(q):
        for j in range(i+1, q):
            quadratic[i*q+j] = quadratic[j*q+i] = rng.randrange(256)
    return rng.randrange(256), linear, tuple(quadratic)


@pytest.mark.parametrize('q', range(9))
def test_every_original_zero_retained_and_all_tail_candidates_correct(q):
    rng = Random(8600+q)
    for case in range(3):
        coef = coefficients(q, rng)
        for tail in range(min(q, 4)+1):
            axes = tuple(range(q)) if case != 1 else tuple(range(0, q, 2))
            derived = degree_three_tail_consequences(coef, tail, multiplier_axes=axes)
            originals = boolean_equations(coef)
            for row in derived['consequences']:
                assert replay_certificate(originals, q, int(row['proof'], 16)) == frozenset(row['terms'])
            retained = []
            for prefix in range(1 << (q-tail)):
                system = specialize_tail_system(derived, prefix)
                candidates = affine_tail_indices(*system)
                direct = tuple(t for t in range(1 << tail)
                               if system[0] == xor_selected(system[1], t))
                assert candidates == direct
                for t in candidates:
                    index = prefix | (t << (q-tail))
                    if evaluate(coef, index) == 0:
                        retained.append(index)
            assert sorted(retained) == [i for i in range(1 << q) if evaluate(coef, i) == 0]


def xor_selected(columns, bits):
    result = 0
    for j, value in enumerate(columns):
        if bits >> j & 1:
            result ^= value
    return result


def test_zero_system_keeps_entire_nullspace_and_constant_contradiction_closes_it():
    for q in (0, 1, 6):
        zero = (0, (0,)*q, (0,)*(q*q))
        derived = degree_three_tail_consequences(zero, q)
        assert affine_tail_indices(*specialize_tail_system(derived, 0)) == tuple(range(1 << q))
        impossible = degree_three_tail_consequences((1, zero[1], zero[2]), q)
        assert affine_tail_indices(*specialize_tail_system(impossible, 0)) == ()


def test_dependency_is_not_mistaken_for_unique_tail():
    assert affine_tail_indices(3, (3, 3, 0)) == (1, 2, 5, 6)
    assert affine_tail_indices(1, (3, 3, 0)) == ()


@pytest.mark.parametrize('tail,axes', [(-1, None), (4, None), (1, (0, 0)), (1, (3,)), (1, (True,))])
def test_invalid_domains_are_rejected(tail, axes):
    with pytest.raises(ValueError):
        degree_three_tail_consequences((0, (0,)*3, (0,)*9), tail, multiplier_axes=axes)
