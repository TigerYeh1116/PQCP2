from random import Random

import pytest

from solver.mod8_crossbred import degree_three_tail_consequences
from solver.mod8_crossbred_native import native_tail_consequences, lookup_tail_consequences, specialize_lookup
from solver.mps_mod8_prefix import evaluate


def coefficients(q, rng):
    quadratic = [0]*(q*q)
    for i in range(q):
        for j in range(i+1, q):
            quadratic[i*q+j] = quadratic[j*q+i] = rng.getrandbits(32)
    return rng.getrandbits(32), tuple(rng.getrandbits(32) for _ in range(q)), tuple(quadratic)


@pytest.mark.parametrize('q', (0, 1, 2, 3, 5, 8, 12, 16, 21, 24))
def test_native_entire_derived_basis_and_every_proof_match_python(q):
    coef = coefficients(q, Random(862000+q))
    for tail in (0, min(q, 3), min(q, 8)):
        for axes in ((), tuple(range(q)), tuple(range(0, q, 2))):
            expected = degree_three_tail_consequences(coef, tail, multiplier_axes=axes)
            actual = native_tail_consequences(coef, tail, multiplier_axes=axes)
            assert actual == expected


def test_native_zero_and_contradiction_keep_same_complete_nullspace_semantics():
    for q in (0, 1, 6, 24):
        tail = min(q, 8)
        for constant in (0, 1):
            coef = constant, (0,)*q, (0,)*(q*q)
            assert native_tail_consequences(coef, tail) == degree_three_tail_consequences(coef, tail)


def test_fast_mode_preserves_every_polynomial_and_certificate():
    coef = coefficients(8, Random(8608))
    audited = native_tail_consequences(coef, 6)
    fast = native_tail_consequences(coef, 6, audit_certificates=False)
    assert audited['all_certificates_independently_replayed']
    assert not fast['all_certificates_independently_replayed']
    assert {k:v for k,v in fast.items() if k != 'all_certificates_independently_replayed'} == {
        k:v for k,v in audited.items() if k != 'all_certificates_independently_replayed'}


@pytest.mark.parametrize('tail,axes', [(-1, None), (4, None), (1, (0, 0)), (1, (3,)), (1, (True,))])
def test_invalid_domains_rejected_before_native_call(tail, axes):
    with pytest.raises(ValueError):
        native_tail_consequences((0, (0,)*3, (0,)*9), tail, multiplier_axes=axes)


@pytest.mark.parametrize('q', (0, 1, 3, 5, 8, 12))
def test_lookup_tables_match_every_selected_polynomial_and_proof(q):
    rng=Random(863000+q)
    full=coefficients(q,rng)
    coef=full[0]&255,tuple(v&255 for v in full[1]),tuple(v&255 for v in full[2])
    witness=rng.randrange(1<<q)
    coef=coef[0]^evaluate(coef,witness),coef[1],coef[2]
    for tail in sorted({0,min(q,2),min(q,6)}):
        lookup=lookup_tail_consequences(coef,tail,audit_certificates=True)
        prefix=q-tail
        for bits in range(1<<prefix):
            constant,columns=specialize_lookup(lookup,bits)
            for suffix in range(1<<tail):
                # Evaluating coefficients after table specialization must
                # equal direct Boolean evaluation of EVERY selected row.
                packed=constant
                for j,column in enumerate(columns):
                    if suffix>>j&1:packed^=column
                index=bits|(suffix<<prefix);expected=0
                for equation,terms in enumerate(lookup['terms']):
                    value=sum(index&term==term for term in terms)%2
                    expected|=value<<equation
                assert packed==expected
