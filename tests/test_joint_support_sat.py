from collections import defaultdict
from itertools import product

import pytest

from solver.correlation import full_correlation_profile
from solver.joint_support_sat import build_joint_support_model,check_joint_support_model
from solver.structured_energy import target_profile
from solver.target_profiles import canonical_target_content_profiles,pair_content,TargetContentProfile
from solver.ternary_support_search import support_mask
from solver.torch_compressed_search import paired_contents


def test_all_l6_literal_supports_and_contents_against_original_binary_pairs():
    bits=list(product((0,1),repeat=6));seen_sat=False
    for target in canonical_target_content_profiles(6):
        expected=defaultdict(set)
        for a in bits:
            for b in bits:
                if (pair_content(a,b)!=(target.a_even_ones,target.a_odd_ones,target.b_even_ones,target.b_odd_ones)
                        or tuple(full_correlation_profile(a,b))!=target_profile(6,target.k,target.eta)):
                    continue
                c=tuple(tuple(1-w[i]-w[i+3] for i in range(3)) for w in (a,b))
                expected[tuple(support_mask(w) for w in c)].add((a,b))
        for contents in paired_contents(target):
            references=tuple(tuple([-1]*n+[0]*z+[1]*p) for n,z,p in contents)
            domain=[]
            for masks in product(range(8),repeat=2):
                if tuple(bin(m).count('1') for m in masks)!=tuple(3-c[1] for c in contents):continue
                out=check_joint_support_model(build_joint_support_model(*references,target,fixed_supports=masks))
                wanted=expected[masks];domain.extend(wanted)
                assert out['status']==('SAT_PQCP' if wanted else 'UNSAT_CONTENT_DOMAIN')
                if wanted:assert out['words'] in wanted and out['verified'];seen_sat=True
                else:assert not out['verified']
                for mod8 in (False,True):
                    projected=check_joint_support_model(build_joint_support_model(*references,target,
                        fixed_supports=masks,affine_projection=True,mod8_projection=mod8))
                    assert projected['status']==out['status']
                    if wanted:assert projected['words'] in wanted
            for gauge in (False,True):
                out=check_joint_support_model(build_joint_support_model(*references,target,rotation_gauge=gauge))
                assert out['status']==('SAT_PQCP' if domain else 'UNSAT_CONTENT_DOMAIN')
                if domain:assert out['words'] in domain
            if domain:
                a,b=domain[0];c_a=tuple(1-a[i]-a[i+3] for i in range(3))
                out=check_joint_support_model(build_joint_support_model(*references,target,fixed_c_a=c_a))
                assert out['status']=='SAT_PQCP' and out['coarse_words'][0]==c_a
    assert seen_sat


def test_l14_support_parity_obstruction_is_complete_unsat():
    from solver.ternary_support_search import support_signature
    for target in canonical_target_content_profiles(14):
        for contents in paired_contents(target):
            references=tuple(tuple([-1]*n+[0]*z+[1]*p) for n,z,p in contents)
            weights=tuple(7-c[1] for c in contents)
            for masks in product(range(128),repeat=2):
                if tuple(bin(m).count('1') for m in masks)!=weights:continue
                if support_signature(masks[0],7)^support_signature(masks[1],7)==1<<(min(target.k,7-target.k)-1):continue
                out=check_joint_support_model(build_joint_support_model(*references,target,fixed_supports=masks))
                assert out['status']=='UNSAT_CONTENT_DOMAIN' and not out['verified']
                return
    raise AssertionError('no independent parity-negative test case')


def test_resource_exhaustion_is_not_an_exclusion():
    target=TargetContentProfile(86,1,-1,16,22,18,21)
    a=tuple([-1]*7+[0]*24+[1]*12);b=tuple([-1]*10+[0]*19+[1]*14)
    out=check_joint_support_model(build_joint_support_model(a,b,target),resource_limit=1)
    assert out['status']=='UNKNOWN' and not out['verified'] and out['words'] is None


def test_reject_inconsistent_fixed_side_instead_of_changing_search_domain():
    target=canonical_target_content_profiles(6)[0]
    contents=paired_contents(target)[0]
    a,b=tuple(tuple([-1]*n+[0]*z+[1]*p) for n,z,p in contents)
    with pytest.raises(ValueError):build_joint_support_model(a,b,target,fixed_supports=(0,0))
    with pytest.raises(ValueError):build_joint_support_model(a,b,target,fixed_c_a=(9,0,0))


def test_optional_fallback_completes_native_unknown_on_the_same_supports():
    from solver.pqcp_support_solver import complete_pqcp_supports
    for target in canonical_target_content_profiles(14):
        for contents in paired_contents(target):
            refs=tuple(tuple([-1]*n+[0]*z+[1]*p) for n,z,p in contents)
            for masks in product(range(128),repeat=2):
                if tuple(bin(m).count('1') for m in masks)!=tuple(7-c[1] for c in contents):continue
                original=complete_pqcp_supports(*refs,target,masks,max_assignments=1)
                if original['status']!='UNKNOWN':continue
                completed=complete_pqcp_supports(*refs,target,masks,max_assignments=1,sat_unknown_completion=True)
                full=complete_pqcp_supports(*refs,target,masks)
                assert completed['status']==full['status']
                assert completed['attempts']['joint_sat']['status']!='UNKNOWN'
                if full['verified']:assert completed['verified']
                projected=check_joint_support_model(build_joint_support_model(*refs,target,fixed_supports=masks,
                    affine_projection=True,mod8_projection=True))
                assert projected['status']==('SAT_PQCP' if full['verified'] else 'UNSAT_CONTENT_DOMAIN')
                partial=complete_pqcp_supports(*refs,target,masks,max_assignments=1,
                    sat_unknown_completion=True,sat_resource_limit=1)
                assert partial['status'] in ('UNKNOWN',full['status'])
                if partial['status']=='UNKNOWN':assert not partial['verified']
                return
    raise AssertionError('no real bounded-native UNKNOWN fixture')


def test_known_half_witnesses_preserve_completion_and_are_checked():
    target=canonical_target_content_profiles(6)[0]
    contents=paired_contents(target)[0]
    refs=tuple(tuple([-1]*n+[0]*z+[1]*p) for n,z,p in contents)
    original=check_joint_support_model(build_joint_support_model(*refs,target))
    assert original['verified']
    c,g=original['coarse_words'],original['difference_words'];masks=tuple(support_mask(w) for w in c)
    for kwargs in (dict(known_coarse_words=c),dict(known_difference_words=g),dict(known_coarse_words=c,known_difference_words=g)):
        replay=check_joint_support_model(build_joint_support_model(*refs,target,fixed_supports=masks,**kwargs))
        assert replay['verified']
        if 'known_coarse_words' in kwargs:assert replay['coarse_words']==c
        if 'known_difference_words' in kwargs:assert replay['difference_words']==g
    with pytest.raises(ValueError):build_joint_support_model(*refs,target,known_coarse_words=c)
    with pytest.raises(ValueError):build_joint_support_model(*refs,target,fixed_supports=masks,known_coarse_words=((0,0,0),(0,0,0)))
