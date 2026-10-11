from collections import defaultdict
from itertools import product

import pytest

from solver.correlation import full_correlation_profile
from solver.projection_group_batch import complete_projection_group_batch
from solver.projection_group_targets import group_projection_targets
from solver.structured_energy import target_profile
from solver.target_profiles import canonical_target_content_profiles,pair_content
from solver.torch_compressed_search import paired_contents


def test_all_l6_batch_domains_against_every_original_binary_pair_and_bounded_unknown():
    words=list(product((0,1),repeat=6));index=defaultdict(set)
    for a in words:
        for b in words:
            masks=tuple(sum(int(w[i]==w[i+3])<<i for i in range(3)) for w in (a,b))
            index[(pair_content(a,b),tuple(full_correlation_profile(a,b)),masks)].add((a,b))
    choices=tuple((t,c) for t in canonical_target_content_profiles(6) for c in paired_contents(t))
    jobs=[];expected=[]
    for group in group_projection_targets(choices):
        refs=tuple(tuple([-1]*n+[0]*z+[1]*p) for n,z,p in group['contents'])
        for masks in product(range(8),repeat=2):
            if tuple(bin(m).count('1') for m in masks)!=tuple(3-c[1] for c in group['contents']):continue
            solutions=set()
            for target in group['targets']:
                solutions.update(index[((target.a_even_ones,target.a_odd_ones,target.b_even_ones,target.b_odd_ones),
                                        target_profile(6,target.k,target.eta),masks)])
            jobs.append((*refs,group['targets'],masks));expected.append(solutions)
    complete=complete_projection_group_batch(jobs,use_mps=False)
    assert any(solutions for solutions in expected)
    for out,solutions in zip(complete,expected):
        assert out['status']==('SAT_PQCP' if solutions else 'UNSAT_BINARY_SUPPORT_GROUP')
        if solutions:assert out['verified'] and out['words'] in solutions
        else:assert not out['verified']
    bounded=complete_projection_group_batch(jobs,use_mps=False,max_assignments=1)
    for out,solutions in zip(bounded,expected):
        if out['status']=='SAT_PQCP':assert out['words'] in solutions
        elif out['status']=='UNSAT_BINARY_SUPPORT_GROUP':assert not solutions
        else:assert out['status']=='UNKNOWN' and not out['verified']


def test_empty_batch_rejected():
    with pytest.raises(ValueError):complete_projection_group_batch((),use_mps=False)
