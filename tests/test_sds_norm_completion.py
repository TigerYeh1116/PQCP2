from itertools import product

import pytest

from solver.sds_norm_completion import (complete_sds_support,cyclic_binary_norm,sds_ingredient_key,
                                        native_odd_cross_vectors)


def _rotate(mask,u,d):
    return ((mask<<u)|(mask>>(d-u)))&((1<<d)-1) if u else mask


@pytest.mark.parametrize('d',(3,5))
def test_all_small_sds_completions_against_full_literal_binary_space(d):
    saw_nonempty=saw_empty=False
    for fixed in range(1<<d):
        for weight in range(d+1):
            for lam in range(2*d+1):
                expected={v for v in range(1<<d) if bin(v).count('1')==weight
                          and all(bin(fixed&_rotate(fixed,u,d)).count('1')
                                  +bin(v&_rotate(v,u,d)).count('1')==lam for u in range(1,d))}
                out=complete_sds_support(fixed,d,weight,lam,rotation_quotient=False)
                assert out['complete'] and set(out['masks'])==expected
                saw_nonempty|=bool(expected);saw_empty|=not expected
                quotient=complete_sds_support(fixed,d,weight,lam)
                lifted={_rotate(v,u,d) for v in quotient['masks'] for u in range(d)}
                assert lifted==expected
    assert saw_nonempty and saw_empty


def test_published43_pair_retained_and_rotation_quotient_has_all_literal_preimages():
    fixed=sum(1<<i for i in (0,3,4,9,11,13,15,18,21,22,23,24,25,35,38,40))
    known=sum(1<<i for i in (1,3,6,9,10,11,16,17,18,21,22,23,25,26,27,33,36,40))
    full=complete_sds_support(fixed,43,18,13,rotation_quotient=False)
    quotient=complete_sds_support(fixed,43,18,13)
    assert known in full['masks']
    assert {_rotate(v,u,43) for v in quotient['masks'] for u in range(43)}==set(full['masks'])
    assert quotient['family_size']*43==full['family_size']
    for v in full['masks']:
        assert cyclic_binary_norm(fixed,43)^cyclic_binary_norm(v,43)==((1<<43)-1)^1


def test_incompatible_counts_are_proved_empty_not_unknown():
    out=complete_sds_support(1,43,18,13)
    assert out['complete'] and not out['masks'] and out['visited']==0
    with pytest.raises(ValueError):complete_sds_support(1<<43,43,18,13)


def test_native_fixed_ingredient_unit_translation_key_matches_literal_group():
    for d in (3,5,11,43):
        masks=range(1<<d) if d<=5 else (0,1,(1<<d)-1,sum(1<<i for i in range(0,d,3)))
        for mask in masks:
            expected=min(_rotate(sum(((mask>>((q*i)%d))&1)<<i for i in range(d)),r,d)
                         for q in range(1,d) for r in range(d))
            assert sds_ingredient_key(mask,d)==expected


def test_sds_interleave_join_retains_full_small_solution_control():
    # D-optimal3 even rows of sums 3/-1, perfect odd rows of sums 1/-1.
    # Their full row sums are 4/-2, satisfying DC 20 = 2*6+8.
    # Test the underlying exact join; the experiment's L86 guard stays intact.
    from solver.interleaving import odd_cross_vectors,sparse_odd_matches,interleave_bits
    from solver.verifier import verify_pqcp
    ea=(1,1,1);eb=(-1,-1,1);o=(1,1,-1)
    left=odd_cross_vectors(ea,o);right=odd_cross_vectors(eb,o)
    matches=list(sparse_odd_matches(left,right))
    assert matches
    for k,da,db in matches:
        assert verify_pqcp(interleave_bits(ea,o,da),interleave_bits(eb,o,db,-1)).is_valid
    from solver.interleaving import best_sparse_odd_candidate
    best=best_sparse_odd_candidate(left,right)
    assert best['full_energy']==0
    assert verify_pqcp(interleave_bits(ea,o,best['left_preimage']),
                       interleave_bits(eb,o,best['right_preimage'],-1)).is_valid
    assert best_sparse_odd_candidate({(4,2):(0,)},{(0,0):(0,)}) is None


def test_native_odd_profiles_match_every_small_literal_pair_and_all_relative_phases():
    from solver.interleaving import odd_cross_vectors
    for d in (3,5):
        for a,b in product(range(1<<d),repeat=2):
            rows=tuple(tuple(1-2*((m>>i)&1) for i in range(d)) for m in (a,b))
            assert native_odd_cross_vectors(a,b,d)==odd_cross_vectors(*rows)
    for a,b in ((1408814590489,2534516066577),(834793047329,8071051198610),(0,(1<<43)-1)):
        rows=tuple(tuple(1-2*((m>>i)&1) for i in range(43)) for m in (a,b))
        assert native_odd_cross_vectors(a,b)==odd_cross_vectors(*rows)
