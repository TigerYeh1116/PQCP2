from collections import defaultdict
from itertools import product

import pytest

from solver.sds_norm_completion import (complete_binary_intersection_profile,
    complete_sds_many_supports,intersection_profile)


def rotate(mask,u,d):return ((mask<<u)|(mask>>(d-u)))&((1<<d)-1) if u else mask


@pytest.mark.parametrize('d',(3,5,11))
def test_every_realizable_small_profile_keeps_all_literal_and_rotation_preimages(d):
    index=defaultdict(set)
    for mask in range(1<<d):index[intersection_profile(mask,d)].add(mask)
    for wanted,expected in index.items():
        literal=complete_binary_intersection_profile(wanted,rotation_quotient=False)
        assert literal['complete'] and set(literal['masks'])==expected
        quotient=complete_binary_intersection_profile(wanted)
        assert {rotate(mask,u,d) for mask in quotient['masks'] for u in range(d)}==expected


def test_nonrealizable_profile_passes_range_and_sum_but_is_completely_decided_empty():
    # Three points with two distance-1 edges force a distance-2 edge, so this
    # profile's distance-3 edge is impossible; it passes elementary bounds.
    wanted=(3,2,0,1,0,0,0,0,1,0,2)
    result=complete_binary_intersection_profile(wanted,rotation_quotient=False,certified_psd=False)
    assert result['complete'] and result['visited']>0 and not result['masks']
    assert all(intersection_profile(m,11)!=wanted for m in range(1<<11))


def test_all_fixed_triples_at_l3_match_entire_original_four_row_binary_space():
    d=3
    for fixed in product(range(1<<d),repeat=3):
        profiles=tuple(intersection_profile(m,d) for m in fixed)
        for w in range(d+1):
            for lam in range(4*d+1):
                expected={v for v in range(1<<d) if bin(v).count('1')==w
                          and all(sum(p[u] for p in profiles)+intersection_profile(v,d)[u]==lam
                                  for u in range(1,d))}
                result=complete_sds_many_supports(fixed,d,w,lam,rotation_quotient=False)
                assert result['complete'] and set(result['masks'])==expected


def test_l43_published_pcs4_known_fourth_rows_survive_complete_norm_completion():
    # Literal source quadruples independently have four-row PACF sum zero.
    # All V completions are retained; original published V is only a control.
    from solver.interleaving import periodic_sign_profile
    for quad in ((1687788056856,3946068457116,3493597975730,1089223681038),
                 (125838764928,225335763648,5883663058516,8119296105360),
                 (1306810654920,2957941461204,3252979340554,1220076600438)):
        assert tuple(bin(m).count('1') for m in quad)==(16,18,21,21)
        rows=tuple(tuple(1-2*((m>>i)&1) for i in range(43)) for m in quad)
        assert tuple(map(sum,zip(*(periodic_sign_profile(row) for row in rows))))==(172,)+(0,)*42
        full=complete_sds_many_supports(quad[:3],43,21,33,rotation_quotient=False)
        quotient=complete_sds_many_supports(quad[:3],43,21,33)
        assert quad[3] in full['masks']
        assert {rotate(m,u,43) for m in quotient['masks'] for u in range(43)}==set(full['masks'])


def test_completed_pcs4_can_produce_a_strict_full_small_pqcp():
    from solver.interleaving import interleave_bits,odd_cross_vectors,sparse_odd_matches
    from solver.verifier import verify_pqcp
    fixed=(0,3,4)
    result=complete_sds_many_supports(fixed,3,1,1,rotation_quotient=False)
    def row(mask):return tuple(1-2*((mask>>i)&1) for i in range(3))
    hits=[]
    for v in result['masks']:
        left=odd_cross_vectors(row(fixed[0]),row(fixed[2]))
        right=odd_cross_vectors(row(fixed[1]),row(v))
        for k,da,db in sparse_odd_matches(left,right):
            words=(interleave_bits(row(fixed[0]),row(fixed[2]),da),interleave_bits(row(fixed[1]),row(v),db,-1))
            assert verify_pqcp(*words).is_valid
            hits.append(words)
    assert hits


def test_invalid_profile_does_not_become_a_silent_exclusion():
    with pytest.raises(ValueError):complete_binary_intersection_profile((2,0,1,0,0))
