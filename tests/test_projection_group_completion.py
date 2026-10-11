from collections import defaultdict
import pytest
from itertools import product

from solver.correlation import full_correlation_profile
from solver.projection_group_completion import complete_projection_group
from solver.projection_group_targets import group_projection_targets
from solver.structured_energy import target_profile,fold_profile
from solver.target_profiles import canonical_target_content_profiles,pair_content
from solver.torch_compressed_search import paired_contents,search_paired_contents


@pytest.mark.parametrize('difference_first',(False,True))
@pytest.mark.parametrize('lazy_c_gate',(False,True))
def test_every_l6_group_against_all_original_binary_pairs_and_budget_unknown(difference_first,lazy_c_gate):
    bits=list(product((0,1),repeat=6));index=defaultdict(set)
    for a in bits:
        for b in bits:
            masks=tuple(sum(int(w[i]==w[i+3])<<i for i in range(3)) for w in (a,b))
            index[(pair_content(a,b),tuple(full_correlation_profile(a,b)),masks)].add((a,b))
    choices=tuple((t,c) for t in canonical_target_content_profiles(6) for c in paired_contents(t))
    groups=group_projection_targets(choices);seen_sat=False
    for group in groups:
        refs=tuple(tuple([-1]*n+[0]*z+[1]*p) for n,z,p in group['contents'])
        for masks in product(range(8),repeat=2):
            if tuple(bin(m).count('1') for m in masks)!=tuple(3-c[1] for c in group['contents']):continue
            expected=set()
            for t in group['targets']:
                expected.update(index[((t.a_even_ones,t.a_odd_ones,t.b_even_ones,t.b_odd_ones),target_profile(6,t.k,t.eta),masks)])
            out=complete_projection_group(*refs,group['targets'],masks,difference_first=difference_first,lazy_c_gate=lazy_c_gate)
            rollback=complete_projection_group(*refs,group['targets'],masks,lazy_c_gate=False)
            assert (out['status'],out['words'],out['actual_target'])==(rollback['status'],rollback['words'],rollback['actual_target'])
            assert out['status']==('SAT_PQCP' if expected else 'UNSAT_BINARY_SUPPORT_GROUP')
            if expected:assert out['words'] in expected and out['verified'];seen_sat=True
            else:assert not out['verified']
            bounded=complete_projection_group(*refs,group['targets'],masks,max_assignments=1,difference_first=difference_first,lazy_c_gate=lazy_c_gate)
            old_bounded=complete_projection_group(*refs,group['targets'],masks,max_assignments=1,lazy_c_gate=False)
            assert (bounded['status'],bounded['words'],bounded['actual_target'])==(old_bounded['status'],old_bounded['words'],old_bounded['actual_target'])
            if bounded['status']=='UNSAT_BINARY_SUPPORT_GROUP':assert not expected
            elif bounded['status']=='SAT_PQCP':assert bounded['words'] in expected
            else:assert bounded['status']=='UNKNOWN' and not bounded['verified']
    assert seen_sat


def test_l86_normalization_preserves_all_53_cases_in_27_c_groups():
    choices=tuple((t,c) for t in canonical_target_content_profiles(86) for c in search_paired_contents(t))
    groups=group_projection_targets(choices)
    assert len(choices)==53 and len(groups)==27
    assert sorted(len(g['targets']) for g in groups)==[1]+[2]*26
    for group in groups:
        folded=None
        for old,new,unit in zip(group['original_targets'],group['targets'],group['target_units']):
            assert unit%2 and unit*old.k%86==new.k
            assert (new.a_even_ones,new.a_odd_ones,new.b_even_ones,new.b_odd_ones)==(old.a_even_ones,old.a_odd_ones,old.b_even_ones,old.b_odd_ones)
            current=fold_profile(target_profile(86,new.k,new.eta),2)
            if folded is not None:assert current==folded
            folded=current
    assert sum(len(g['targets']) for g in groups)==len(choices)
