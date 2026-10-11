from random import Random

import pytest

from solver.mod8_model_reference import (
    _pack_bit_coefficients_from_native, pack_bit_tail_native, pack_projection_native,
)
from solver.mps_mod8_prefix import evaluate


def coefficients(q,rng):
    quadratic=[0]*(q*q)
    for i in range(q):
        for j in range(i+1,q):quadratic[i*q+j]=quadratic[j*q+i]=rng.getrandbits(25)
    return rng.getrandbits(25),tuple(rng.getrandbits(25) for _ in range(q)),tuple(quadratic)


@pytest.mark.parametrize('q',(0,1,3,5,8,16,21,24))
def test_all_tail_truth_values_reproduce_original_polynomial(q):
    rng=Random(865000+q);coef=coefficients(q,rng)
    for tail in sorted({0,min(q,3),min(q,6),min(q,8)}):
        raw,table,chunks=_pack_bit_coefficients_from_native(coef,tail,25)
        expected_raw,_,_=pack_projection_native(coef,max(1,tail))
        checked,checked_chunks=pack_bit_tail_native(expected_raw,q,tail,25)
        assert list(raw)==expected_raw and list(table)==list(checked) and chunks==checked_chunks
        prefix=q-tail
        for bits in sorted({0,(1<<prefix)-1,rng.randrange(1<<prefix)}):
            rhs=evaluate(coef,bits)
            columns=tuple(evaluate(coef,bits|(1<<(prefix+j)))^rhs for j in range(tail))
            for member in range(1<<tail):
                syndrome=rhs
                for j,column in enumerate(columns):
                    if member>>j&1:syndrome^=column
                for equation in range(25):
                    syndrome^=((int(table[equation*chunks+member//32])>>(member%32))&1)<<equation
                assert syndrome==evaluate(coef,bits|(member<<prefix))
                for j in range(tail):
                    assert (int(table[(25+j)*chunks+member//32])>>(member%32))&1 == (member>>j)&1


@pytest.mark.parametrize('q,tail,equations',( (25,1,23),(3,4,23),(3,-1,23),(3,1,26),(3,1,0) ))
def test_invalid_dimensions_rejected(q,tail,equations):
    with pytest.raises(ValueError):pack_bit_tail_native([0]*601,q,tail,equations)


def test_native_packet_width_guard_rejects_truncated_constraints():
    with pytest.raises(RuntimeError):_pack_bit_coefficients_from_native((1<<23,(0,), (0,)),1,23)
    with pytest.raises(ValueError):pack_bit_tail_native([1<<23]+[0]*600,1,1,23)
