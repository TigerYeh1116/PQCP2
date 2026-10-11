import random

import pytest

from solver.cyclic_support import iter_family,parity_support_family
from solver.cyclic_support_sampling import sample_weighted_support,weighted_support_at


@pytest.mark.parametrize('d',(3,5,11))
def test_every_weighted_index_retains_all_literal_norm_preimages(d):
    for u in range(1<<d) if d<11 else range(0,1<<d,127):
        for k in range(1,d//2+1):
            family=parity_support_family(u,d,k)
            literal=list(iter_family(family))
            for weight in range(d+1):
                expected=[v for v in literal if bin(v).count('1')==weight]
                for index,mask in enumerate(expected):
                    actual,count=weighted_support_at(family,weight,index)
                    assert actual==mask and count==len(expected)
                out=sample_weighted_support(family,weight,random.Random(123))
                assert out['weight_matches']==len(expected)
                assert out['mask'] in expected if expected else out['mask'] is None


def test_sampled_l43_roots_have_exact_weights_and_reproducible_indices():
    family=parity_support_family(0x51819324135,43,2)
    for weight in (15,19,27):
        a=sample_weighted_support(family,weight,random.Random(43))
        b=sample_weighted_support(family,weight,random.Random(43))
        assert a==b and a['mask'] is not None


@pytest.mark.parametrize('d',(3,5,11))
def test_phase_blocks_reproduce_every_original_literal_weighted_rank(d):
    from solver.cyclic_support_sampling import _phase_count
    from solver.cyclic_support_native import nearest_supports
    from solver.cyclic_support import NormSupportFamily
    for u in range(1<<d) if d<11 else range(0,1<<d,127):
        for k in range(1,d//2+1):
            family=parity_support_family(u,d,k);literal=list(iter_family(family))
            for weight in range(d+1):
                expected=[v for v in literal if bin(v).count('1')==weight]
                quotient,first,blocks,per_block=_phase_count(family,weight)
                assert blocks*per_block==len(expected)
                if first is not None:
                    width=len(quotient.choices[first])
                    for block in range(blocks):
                        groups=list(family.choices);groups[first]=groups[first][block*width:(block+1)*width]
                        part=NormSupportFamily(d,family.support_a,k,family.base,tuple(groups))
                        assert nearest_supports(part,0,weight,capacity=1)['weight_matches']==per_block
                class Pick:
                    def __init__(self,index):self.index=index
                    def randrange(self,count):assert count==len(expected);return self.index
                for index,mask in enumerate(expected):
                    actual=sample_weighted_support(family,weight,Pick(index),phase_count_pruning=True)
                    assert actual['mask']==mask and actual['selected_index']==index


def test_phase_pruning_preserves_l43_rng_and_actual_mask():
    for u in (0x51819324135,0x32494D32295,0x0F1362B4987):
        family=parity_support_family(u,43,2)
        for weight in (15,19,27):
            for seed in (43,97):
                old=sample_weighted_support(family,weight,random.Random(seed))
                new=sample_weighted_support(family,weight,random.Random(seed),phase_count_pruning=True)
                assert (new['mask'],new['weight_matches'],new.get('selected_index'))==(old['mask'],old['weight_matches'],old.get('selected_index'))
                assert new['counted_family_size']*new['phase_blocks']==family.size
