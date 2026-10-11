from itertools import product
from random import Random

from solver.finite_field_trace_pruning import finite_trace_target_filter,verify_finite_trace_certificate
from solver.sds_norm_completion import intersection_profile
from tests.test_joint_gram_pruning import cross


def test_all_small_literal_norm_cross_targets_survive_including_zero_partners():
    for d in (3,5):
        tau=((1-d)//2)%d
        for mask,partner in product(range(1<<d),repeat=2):
            wanted=intersection_profile(mask,d);actual=cross(mask,partner,d)
            for eta in (-1,1):
                base=tuple(v-eta*int(u in (tau,-tau%d)) for u,v in enumerate(actual))
                out=finite_trace_target_filter(wanted,partner,base,((1,eta),))
                assert out['complete'] and out['allowed_targets']==((1,eta),) and not out['certificates']


def test_all_l11_literal_inputs_and_l43_sampled_inputs_keep_true_full_integer_cross():
    rng=Random(257)
    for d in (11,43):
        masks=range(1<<d) if d==11 else tuple(rng.getrandbits(d) for i in range(100))
        for mask in masks:
            partner=rng.getrandbits(d);wanted=intersection_profile(mask,d);actual=cross(mask,partner,d)
            out=finite_trace_target_filter(wanted,partner,actual,((2,1),))
            assert out['allowed_targets']==((2,1),)


def test_impossible_field_trace_targets_have_independently_verified_proofs():
    wanted=intersection_profile(3493597975730,43);partner=1687788056856
    base=(0,)+tuple(12 for i in range(42));targets=tuple((k,eta) for k in range(1,43) for eta in (-1,1))
    out=finite_trace_target_filter(wanted,partner,base,targets)
    assert out['complete'] and out['excluded_targets']>0
    assert all(verify_finite_trace_certificate(c) for c in out['certificates'])
    assert len(out['allowed_targets'])+out['excluded_targets']==84


def test_affine_filter_is_exactly_all_individual_targets_with_independent_proofs():
    from solver.finite_field_trace_pruning import finite_trace_affine_filter
    rng=Random(43)
    for d in (3,5,11,43):
        targets=tuple((k,e) for k in range(1,d) for e in (-1,1))
        for sample in range(12):
            wanted=intersection_profile(rng.getrandbits(d),d);partner=rng.getrandbits(d)
            base=(0,)+tuple(rng.randrange(-d,d+1) for i in range(d//2))
            base+=tuple(reversed(base[1:]))
            actual=finite_trace_affine_filter(wanted,partner,base,targets)
            allowed=[];certificates=[]
            for k,e in targets:
                profile=list(wanted)
                if not k%2:profile[k//2]+=e;profile[-(k//2)]+=e
                # Affine public input includes signed bases and profiles at
                # generic bounded integer values; choose safe test origins.
                if max(profile)>d:continue
                single=finite_trace_target_filter(profile,partner,base,((k,e),))
                allowed.extend(single['allowed_targets']);certificates.extend(single['certificates'])
                assert ((k,e) in actual['allowed_targets'])==bool(single['allowed_targets'])
                assert tuple(c for c in actual['certificates'] if tuple(c['target'])==(k,e))==tuple(single['certificates'])
            assert all(verify_finite_trace_certificate(c) for c in actual['certificates'])
