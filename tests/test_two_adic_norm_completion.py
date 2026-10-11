from itertools import product
from random import Random

import pytest

from solver.cyclic_support import support_ring,poly_divmod,poly_mul
from solver.structured_energy import integer_periodic_autocorrelation_profile as pacf
from solver.ternary_system_native import prepare_native_system,run_native_system
from solver.two_adic_norm_completion import (lifted_idempotents,native_project,cyclic_product,
    two_adic_norm_obstruction,verify_two_adic_certificate,lifted_unary_signs)


def planes(row):return tuple(sum(((v%8)>>j&1)<<i for i,v in enumerate(row)) for j in range(3))


def test_lifted_crt_projectors_and_native_bit_planes_match_literal_integer_convolution():
    rng=Random(8)
    for d in (3,5,11,43):
        for e in lifted_idempotents(d):
            for sample in range(8):
                row=tuple(rng.randrange(-d,d+1) for i in range(d))
                assert native_project(planes(row),e,d)==planes(cyclic_product(row,e,8))


def test_every_small_true_signed_norm_survives_mod4_obstruction():
    for d in (3,5):
        for row in product((-1,0,1),repeat=d):assert two_adic_norm_obstruction(tuple(pacf(row))) is None
    # This is a valid SUM of two unknown norms, but not a single norm.
    u=(1,0,0);v=(0,1,1);summed=tuple(a+b for a,b in zip(pacf(u),pacf(v)))
    proof=two_adic_norm_obstruction(summed)
    assert summed==(3,1,1) and proof and verify_two_adic_certificate(proof)
    plan=prepare_native_system((1,6),3,summed,(0,0),conditioning=False)
    with pytest.raises(ValueError):lifted_unary_signs(plan,summed,0,100,minimum_free=0)


def test_lifted_small_single_sign_fibers_match_every_literal_signed_assignment():
    triggered=0
    for d in (3,5):
        for support in range(1<<d):
            weight=bin(support).count('1');expected=set()
            for row in product((-1,0,1),repeat=d):
                if sum((v!=0)<<i for i,v in enumerate(row))==support:expected.add((tuple(pacf(row)),row.count(-1)))
            for negative in range(weight+1):
                sigma=weight-2*negative
                for half in product(range(-weight,weight+1),repeat=d//2):
                    profile=(weight,)+half+tuple(reversed(half))
                    if sum(profile)!=sigma*sigma:continue
                    plan=prepare_native_system((support,0),d,profile,(negative,0),conditioning=False)
                    proof=two_adic_norm_obstruction(profile)
                    original=run_native_system(plan,2**31)
                    if proof:
                        assert original['status']=='UNSAT_FIXED_SUPPORT';continue
                    out=lifted_unary_signs(plan,profile,negative,2**31,minimum_free=0)
                    if out is None:out=original
                    else:triggered+=1
                    assert (out['status']=='SAT_COMPRESSED')==((profile,negative) in expected)
                    if out['words']:assert tuple(pacf(out['words'][0]))==profile
    assert triggered


def test_high_nullity_l43_zero_components_keep_actual_signed_positive_control():
    ring=support_ring(43);e=ring.idempotents[1]
    columns=tuple(poly_divmod(poly_mul(1<<i,e),(1<<43)|1)[1] for i in range(14))
    mask=0;support=None
    for index in range(1<<14):
        if index:mask^=columns[(index&-index).bit_length()-1]
        if bin(mask).count('1')==30:support=mask;break
    assert support is not None
    positions=tuple(i for i in range(43) if support>>i&1);negative=set(positions[:13])
    row=tuple(-1 if i in negative else 1 if support>>i&1 else 0 for i in range(43));profile=tuple(pacf(row))
    plan=prepare_native_system((support,0),43,profile,(13,0),conditioning=False)
    assert plan['consistent'] and plan['meta'][2]>=20 and two_adic_norm_obstruction(profile) is None
    out=lifted_unary_signs(plan,profile,13,2**31)
    assert out and out['status']=='SAT_COMPRESSED' and out['zero_norm_components']==2
    assert tuple(pacf(out['words'][0]))==profile and out['words'][0].count(-1)==13
    conditioned=prepare_native_system((support,0),43,profile,(13,0))
    other=lifted_unary_signs(conditioned,profile,13,2**31)
    assert other and other['status']=='SAT_COMPRESSED' and tuple(pacf(other['words'][0]))==profile
    # All primitive residues vanish: a constant row has a 42-dimensional
    # affine space when forced-cardinality propagation is deliberately off.
    support=(1<<43)-1;profile=(43,)*43
    plan=prepare_native_system((support,0),43,profile,(0,0),conditioning=False,cardinality_pruning=False)
    assert plan['meta'][2]==42
    out=lifted_unary_signs(plan,profile,0,100)
    assert out['status']=='SAT_COMPRESSED' and out['assignments']==1 and out['zero_norm_components']==3
