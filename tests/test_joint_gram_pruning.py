from itertools import product

import sympy

from solver.certified_profile_psd import SCALE
from solver.joint_gram_pruning import cosine_interval_grid,joint_gram_target_filter,verify_joint_gram_certificate
from solver.sds_norm_completion import intersection_profile


def cross(mask,partner,d):
    all_bits=(1<<d)-1
    def rotate(value,u):return ((value<<u)|(value>>(d-u)))&all_bits if u else value
    return tuple(bin(mask&rotate(partner,u)).count('1')+bin(mask&rotate(partner,-u%d)).count('1') for u in range(d))


def test_outward_cosine_intervals_include_every_true_supported_phase():
    for d in (3,5,11,43):
        lo,hi=cosine_interval_grid(d)
        for phase in range(d//2+1):
            exact=sympy.N(sympy.cos(2*sympy.pi*phase/d),90)
            assert sympy.Rational(lo[phase],SCALE)<=exact<=sympy.Rational(hi[phase],SCALE)
        assert lo[0]==hi[0]==SCALE


def test_every_small_literal_u_w_pair_survives_its_actual_joint_cross_target():
    for d in (3,5):
        k=1;tau=((k-d)//2)%d
        for mask,partner in product(range(1<<d),repeat=2):
            wanted=intersection_profile(mask,d);actual=cross(mask,partner,d)
            for eta in (-1,1):
                base=tuple(v-eta*int(u in (tau,-tau%d)) for u,v in enumerate(actual))
                out=joint_gram_target_filter(wanted,partner,base,((k,eta),))
                assert out['complete'] and out['allowed_targets']==((k,eta),) and not out['certificates']


def test_joint_gram_excludes_range_and_sum_compatible_cross_with_positive_norms():
    wanted=intersection_profile(3,5);base=(4,0,2,2,0)
    assert sum(base)==2*2*2 and all(0<=v<=4 for v in base) and base[0]%2==0
    out=joint_gram_target_filter(wanted,3,base,((2,1),))
    assert not out['allowed_targets'] and out['excluded_targets']==1
    certificate=out['certificates'][0]
    assert certificate['reason']=='NEGATIVE_JOINT_GRAM_UPPER' and verify_joint_gram_certificate(certificate)
    damaged=dict(certificate,upper_bound_numerator=certificate['upper_bound_numerator']-1)
    assert not verify_joint_gram_certificate(damaged)


def test_native_target_batch_certificate_matches_independent_python_reconstruction():
    wanted=intersection_profile(3493597975730,43);partner=1687788056856
    base=(0,)+tuple(50 for _ in range(42));targets=tuple((k,eta) for k in range(1,43) for eta in (-1,1))
    out=joint_gram_target_filter(wanted,partner,base,targets)
    assert out['complete'] and out['tested_targets']==84 and len(out['certificates'])>0
    assert all(verify_joint_gram_certificate(c) for c in out['certificates'])
    assert len(out['allowed_targets'])+len(out['certificates'])==84
