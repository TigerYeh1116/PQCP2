from collections import defaultdict
from itertools import product

import pytest

from solver.correlation import full_correlation_profile
from solver.projection_group_targets import group_projection_targets
from solver.structured_energy import target_profile
from solver.target_profiles import canonical_target_content_profiles,pair_content
from solver.torch_compressed_search import paired_contents
from solver.weight_group_completion import complete_weight_group
from solver.weight_group_targets import group_weight_targets


@pytest.mark.parametrize('shared',(False,True))
@pytest.mark.parametrize('paired_c',(False,True))
def test_every_small_group_against_all_original_binary_pairs_and_partial_unknown(shared,paired_c):
    length=6;d=3;bits=list(product((0,1),repeat=length));index=defaultdict(set)
    for a in bits:
        for b in bits:
            masks=tuple(sum(int(w[i]==w[i+d])<<i for i in range(d)) for w in (a,b))
            index[(pair_content(a,b),tuple(full_correlation_profile(a,b)),masks)].add((a,b))
    choices=tuple((t,c) for t in canonical_target_content_profiles(length) for c in paired_contents(t))
    groups=group_weight_targets(group_projection_targets(choices));seen_positive=False
    for group in groups:
        targets=tuple(t for e in group['c_groups'] for t in e['targets'])
        for masks in product(range(1<<d),repeat=2):
            if tuple(bin(m).count('1') for m in masks)!=group['weights']:continue
            expected=set()
            for t in targets:
                expected.update(index[((t.a_even_ones,t.a_odd_ones,t.b_even_ones,t.b_odd_ones),
                    target_profile(length,t.k,t.eta),masks)])
            out=complete_weight_group(group,masks,shared=shared,paired_c=paired_c)
            assert out['status']==('SAT_PQCP' if expected else 'UNSAT_BINARY_WEIGHT_GROUP')
            if expected:
                assert out['verified'] and out['words'] in expected;seen_positive=True
            else:assert out['words'] is None and not out['verified']
            bounded=complete_weight_group(group,masks,shared=shared,max_assignments=1,paired_c=paired_c)
            if bounded['status']=='SAT_PQCP':assert bounded['words'] in expected
            elif bounded['status']=='UNSAT_BINARY_WEIGHT_GROUP':assert not expected
            else:assert bounded['status']=='UNKNOWN' and not bounded['verified']
    assert seen_positive
