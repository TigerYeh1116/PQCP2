from itertools import permutations

import pytest

from solver.target_profiles import canonical_target_content_profiles
from solver.torch_compressed_search import paired_contents
from solver.ternary_native_search import search_ternary_native


@pytest.mark.parametrize('length',[6,14,46,86])
def test_every_proposal_audited_against_full_integer_pacf(length):
    # audit_interval=1 compares each proposal, including rejected moves,
    # to an independent O(d²) recomputation inside C. Python audits the best.
    for target in canonical_target_content_profiles(length):
        for ca,cb in paired_contents(target)[::max(1,len(paired_contents(target))//3)]:
            a,b=(tuple([-1]*n+[0]*z+[1]*p) for n,z,p in (ca,cb))
            result=search_ternary_native(a,b,target,moves=128,seed=7,audit_interval=1)
            assert result['energy']<=result['initial_energy']
            assert result['audits']==result['proposals']
            assert result['proposals']<=128
            assert result['status']==('SAT_COMPRESSED' if result['energy']==0 else 'BOUNDED_MISS')
            assert all(sum(w)==sum(old) for w,old in zip(result['words'],(a,b)))


def test_zero_budget_keeps_reference_and_determinism():
    target=canonical_target_content_profiles(86)[0]
    ca,cb=paired_contents(target)[5]
    a,b=(tuple([-1]*n+[0]*z+[1]*p) for n,z,p in (ca,cb))
    zero=search_ternary_native(a,b,target,moves=0)
    assert zero['words']==(a,b) and zero['proposals']==0
    first=search_ternary_native(a,b,target,moves=1000,seed=123)
    second=search_ternary_native(a,b,target,moves=1000,seed=123)
    for key in ('words','energy','proposals','accepted','transfer_proposals','transfer_accepted'):
        assert first[key]==second[key]


def test_small_native_hits_belong_to_literal_solution_set():
    from solver.structured_energy import fold_profile,target_profile,integer_periodic_autocorrelation_profile as pacf
    target=canonical_target_content_profiles(6)[0]
    solutions=set()
    wanted=tuple(v//4 for v in fold_profile(target_profile(6,target.k,target.eta),2))
    for ca,cb in paired_contents(target):
        aa,bb=(set(permutations([-1]*n+[0]*z+[1]*p)) for n,z,p in (ca,cb))
        for a in aa:
            for b in bb:
                if tuple(x+y for x,y in zip(pacf(a),pacf(b)))==wanted:solutions.add((a,b))
    saw_hit=False
    for ca,cb in paired_contents(target):
        a,b=(tuple([-1]*n+[0]*z+[1]*p) for n,z,p in (ca,cb))
        r=search_ternary_native(a,b,target,moves=1000,audit_interval=1)
        if r['status']=='SAT_COMPRESSED':
            saw_hit=True;assert r['words'] in solutions
    assert saw_hit


@pytest.mark.parametrize('kwargs',[{'moves':-1},{'period':0},{'audit_interval':0},{'seed':True}])
def test_invalid_arguments(kwargs):
    t=canonical_target_content_profiles(6)[0]
    ca,cb=paired_contents(t)[0]
    a,b=(tuple([-1]*n+[0]*z+[1]*p) for n,z,p in (ca,cb))
    with pytest.raises(ValueError):search_ternary_native(a,b,t,**kwargs)
