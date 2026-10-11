"""CPU algebra audit; real GPU full-zero-set audit lives in experiments/."""
from random import Random

import pytest

from solver.mod8_model_reference import all_mod8_indices, pack_projection_native
from solver.mps_mod8_prefix import complete_mod8_models, evaluate, packed_coefficients


def coefficients(q, rng):
    quadratic = [0]*(q*q)
    for i in range(q):
        for j in range(i+1, q):
            quadratic[i*q+j] = quadratic[j*q+i] = rng.getrandbits(32)
    return rng.getrandbits(32), tuple(rng.getrandbits(32) for _ in range(q)), tuple(quadratic)


@pytest.mark.parametrize('q', range(25))
def test_native_projection_matches_independent_python_quotient_all_dimensions(q):
    rng = Random(86000+q)
    for case in range(2):
        coef = coefficients(q, rng)
        for tail in (1, 3, 6, 8):
            assert pack_projection_native(coef, tail) == packed_coefficients(coef, tail)


@pytest.mark.parametrize('q', range(11))
def test_full_gray_reference_matches_original_polynomial(q):
    rng = Random(860000+q)
    for case in range(3):
        coef = coefficients(q, rng)
        # Include a known polynomial zero, not just likely empty systems.
        witness = rng.randrange(1 << q)
        coef = (coef[0] ^ evaluate(coef, witness), coef[1], coef[2])
        expected = tuple(j for j in range(1 << q) if evaluate(coef, j) == 0)
        reference = all_mod8_indices(coef)
        assert reference == dict(complete=True, candidate_count=len(expected), indices=expected)


def test_overflow_never_becomes_a_complete_empty_set_or_an_unsat_decision():
    reference = all_mod8_indices((0, (0,)*6, (0,)*36), capacity=1)
    assert reference == dict(complete=False, candidate_count=64, indices=None)
    result = complete_mod8_models({}, reference)
    assert result['status'] == 'UNKNOWN'
    assert result['words'] is None


@pytest.mark.parametrize('coef', [
    (0, (0, 0), (0,)), (0, (0, 0), (0, 1, 0, 0)),
    (0, (0, 0), (1, 0, 0, 0)), (-1, (), ()), (2**32, (), ()),
])
def test_invalid_matrices_rejected_before_native_memory_access(coef):
    with pytest.raises(ValueError):
        all_mod8_indices(coef)
    with pytest.raises(ValueError):
        pack_projection_native(coef, 6)


@pytest.mark.parametrize('tail', (-1, 0, 9, True))
def test_invalid_tail_rejected(tail):
    with pytest.raises(ValueError):
        pack_projection_native((0, (), ()), tail)
