from itertools import product
from random import Random

import pytest

from solver.near_norm_trades import content_trade_groups,exact_profile_pairs,packed,profile,signs,zero_odd_offset_pairs


@pytest.mark.parametrize('d',(3,5,7,9))
def test_all_content_trade_words_and_integer_codes_against_complete_original_space(d):
    rng=Random(860300+d)
    for _ in range(4):
        row=tuple(rng.choice((-1,1)) for _ in range(d))
        for bound in sorted({1,3,min(5,d)}):
            for target_sum in range(-d,d+1,2):
                result=content_trade_groups(row,target_sum,bound)
                expected={sum((v==-1)<<i for i,v in enumerate(word))
                          for word in product((-1,1),repeat=d)
                          if sum(word)==target_sum and sum(a!=b for a,b in zip(word,row))<=bound}
                actual=set()
                for code,items in result['groups'].items():
                    for item in items:
                        word=signs(item['mask'],d);actual.add(item['mask'])
                        assert code==packed(profile(word))
                        assert tuple(i for i,(a,b) in enumerate(zip(row,word)) if a!=b)==item['flips']
                assert actual==expected and len(actual)==result['literal_variants']


@pytest.mark.parametrize('d',(3,5,7))
def test_exact_packed_join_preserves_every_physical_pair_preimage(d):
    a=tuple((-1)**i for i in range(d));b=tuple((-1)**(i//2) for i in range(d))
    left=content_trade_groups(a,1,3);right=content_trade_groups(b,1,3)
    l=[item for models in left['groups'].values() for item in models]
    r=[item for models in right['groups'].values() for item in models]
    targets={tuple(x+y for x,y in zip(profile(signs(aa['mask'],d)),profile(signs(bb['mask'],d))))
             for aa in l for bb in r}
    for target in targets:
        expected={(aa['mask'],bb['mask']) for aa in l for bb in r
                  if tuple(x+y for x,y in zip(profile(signs(aa['mask'],d)),profile(signs(bb['mask'],d))))==target}
        actual={(aa['mask'],bb['mask']) for aa,bb in exact_profile_pairs(left,right,target)}
        assert actual==expected


def test_signed_packing_is_not_a_digit_modulus():
    assert packed((-1,1))==511
    assert packed((511,0))==511  # Outside the physical bound: a real collision.
    left=content_trade_groups((1,1,-1),1,1)
    with pytest.raises(ValueError):list(exact_profile_pairs(left,left,(511,)))


def test_complete_even_peak_construction_against_original_length6_binary_oracle():
    from solver.correlation import full_correlation_profile
    from solver.interleaving import interleave_bits
    from solver.structured_energy import target_profile
    from solver.verifier import verify_pqcp
    rows=tuple(product((-1,1),repeat=3));target=target_profile(6,2,1);positive=0
    for e0,o0,e1,o1 in product(rows,repeat=4):
        if sum(profile(row)[0] for row in (e0,o0,e1,o1))!=4:continue
        actual=set(zero_odd_offset_pairs(e0,o0,e1,o1));expected=set()
        for da,db in product(range(3),repeat=2):
            words=(interleave_bits(e0,o0,da),interleave_bits(e1,o1,db,-1))
            if tuple(full_correlation_profile(*words))==target:
                expected.add((da,db));assert verify_pqcp(*words).is_valid
        assert actual==expected
        positive+=len(actual)
    assert positive>0
