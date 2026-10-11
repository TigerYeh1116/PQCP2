from itertools import product
from math import gcd
from random import Random

import pytest

from solver.coarse_anchor_orbits import coarse_anchor_key, coarse_anchor_key_with_action,coarse_word_orbit_key
from solver.coarse_row_completion import coarse_target,coarse_l1_anchors
from solver.full_row_completion import reconstruct_crt_binary
from solver.structured_energy import integer_periodic_autocorrelation_profile as pacf


def direct_key(row):
    d=len(row);norm=pacf(row);values=[]
    for unit in range(1,d):
        if gcd(unit,d)!=1:continue
        transformed=tuple(norm[unit*j%d] for j in range(d))
        for offset in range(d):
            support=sum((row[unit*(j+offset)%d]!=0)<<j for j in range(d))
            values.append((transformed,support))
    return min(values)


@pytest.mark.parametrize('d',(3,5))
def test_every_literal_small_anchor_matches_independent_exact_key(d):
    for row in product((-1,0,1),repeat=d):
        key,(unit,offset)=coarse_anchor_key_with_action(row)
        transformed=tuple(row[unit*(j+offset)%d] for j in range(d))
        assert key==direct_key(row)
        assert key==(tuple(pacf(transformed)),sum((v!=0)<<j for j,v in enumerate(transformed)))
        assert coarse_anchor_key(tuple(-v for v in row))==key


@pytest.mark.parametrize('d',(3,5,11,43))
def test_binary_odd_unit_action_preserves_target_parity_and_coarse_domains(d):
    rng=Random(864000+d)
    for case in range(8):
        c=tuple(rng.choice((-1,0,1)) for _ in range(d))
        g=tuple(rng.choice((-1,1)) if not v else 0 for v in c)
        key,(unit,offset)=coarse_anchor_key_with_action(c)
        cp=tuple(c[unit*(j+offset)%d] for j in range(d))
        gp=tuple(g[unit*(j+offset)%d] for j in range(d))
        original=reconstruct_crt_binary(c,g);changed=reconstruct_crt_binary(cp,gp)
        lift=unit if unit%2 else unit+d
        assert gcd(lift,2*d)==1
        assert changed==tuple(original[(lift*i+2*unit*offset)%(2*d)] for i in range(2*d))
        target_map=set()
        for k in range(1,d):
            mapped=k*pow(lift,-1,2*d)%(2*d);mapped=min(mapped,2*d-mapped)
            assert k%2==mapped%2
            target_map.add(mapped)
            for eta in (-1,1):
                old=coarse_target(c,k,eta);new=coarse_target(cp,mapped,eta)
                assert (old is None)==(new is None)
                if old is None:continue
                assert old['c_b_negative']==new['c_b_negative']
                assert old['g_negative_pairs']==new['g_negative_pairs']
                assert new['wanted_c_b']==tuple(old['wanted_c_b'][unit*j%d] for j in range(d))
                assert new['wanted_g_pair']==tuple(old['wanted_g_pair'][unit*j%d] for j in range(d))
        assert target_map==set(range(1,d))
        assert coarse_anchor_key(cp)==key


def test_homometric_sign_words_same_support_give_identical_completion_specs():
    groups={}
    for row in product((-1,0,1),repeat=5):
        norm=tuple(pacf(row));mask=sum((v!=0)<<j for j,v in enumerate(row))
        previous=groups.setdefault((norm,mask),row)
        for k in range(1,5):
            for eta in (-1,1):assert coarse_target(previous,k,eta)==coarse_target(row,k,eta)


@pytest.mark.parametrize('d',(3,5))
def test_literal_word_isometry_key_and_full_ball_are_exact(d):
    for row in product((-1,0,1),repeat=d):
        alternatives=[]
        for unit in range(1,d):
            for offset in range(d):
                word=tuple(row[unit*(j+offset)%d] for j in range(d))
                mask=sum((v!=0)<<j for j,v in enumerate(word));minus=sum((v==-1)<<j for j,v in enumerate(word))
                alternatives.append((mask,min(minus,mask^minus)))
        assert coarse_word_orbit_key(row)==min(alternatives)
    original=tuple((-1,0,1)[j%3] for j in range(d))
    first={coarse_word_orbit_key(row) for row,_ in coarse_l1_anchors(original,2)}
    for unit in range(1,d):
        for offset in range(d):
            changed=tuple(-original[unit*(j+offset)%d] for j in range(d))
            assert {coarse_word_orbit_key(row) for row,_ in coarse_l1_anchors(changed,2)}==first
