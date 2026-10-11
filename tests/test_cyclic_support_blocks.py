from itertools import product

import pytest

from solver.cyclic_support_blocks import join_support_blocks,support_block_options
from solver.ternary_support_search import support_signature


def literal_options(reference,d,positions,weights):
    selected=sum(1<<i for i in positions)
    return {reference&~selected | sum(v<<i for i,v in zip(positions,bits))
        for bits in product((0,1),repeat=len(positions))
        if bin(reference&~selected | sum(v<<i for i,v in zip(positions,bits))).count('1') in weights}


@pytest.mark.parametrize('d',(3,5,11))
def test_all_gray_options_equal_literal_subsets_and_integer_signatures(d):
    positions=tuple(range(min(d,5)))
    for reference in range(1<<d) if d<6 else range(0,1<<d,127):
        for weights in (tuple(range(d+1)),tuple(range(0,d+1,2)),(d,),()):
            out=support_block_options(reference,d,positions,weights)
            assert set(out.masks)==literal_options(reference,d,positions,weights)
            assert len(out.masks)==len(set(out.masks))
            assert out.covered_configurations==1<<len(positions)
            for mask,signature,weight in zip(out.masks,out.signatures,out.weights):
                assert signature==support_signature(mask,d) and weight==bin(mask).count('1')


@pytest.mark.parametrize('d',(3,5,11))
def test_support_join_matches_complete_literal_cartesian_pairs(d):
    masks=((1<<(d//2))-1,(1<<d)-1)
    positions=(tuple(range(min(d,4))),tuple(range(d-min(d,4),d)))
    weight_pairs=tuple((w,d-w) for w in range(d+1))
    rows=[literal_options(mask,d,p,range(d+1)) for mask,p in zip(masks,positions)]
    for k in range(1,d//2+1):
        expected={(a,b) for a,b in product(*rows) if bin(a).count('1')+bin(b).count('1')==d
            and support_signature(a,d) ^ support_signature(b,d)==1<<(k-1)}
        out=join_support_blocks(masks,d,k,positions,weight_pairs,audit_all=True)
        assert out['complete'] and set(map(tuple,out['masks']))==expected


def test_every_retained_l43_gray_option_matches_independent_integer_signature():
    mask=0x51819324135
    positions=tuple(range(0,32,2))
    out=support_block_options(mask,43,positions,(14,16,18,20,22,24,26))
    assert out.covered_configurations==65536
    selected=sum(1<<p for p in positions)
    for word,signature,weight in zip(out.masks,out.signatures,out.weights):
        assert word&~selected==mask&~selected
        assert signature==support_signature(word,43) and weight==bin(word).count('1')
