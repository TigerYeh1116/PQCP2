from itertools import product

import pytest

from solver.structured_energy import integer_periodic_autocorrelation_profile as pacf, fold_profile, target_profile
from solver.target_profiles import canonical_target_content_profiles
from solver.torch_compressed_search import paired_contents
from solver.ternary_mitm import repair_mitm


def neighborhood(word):
    def swaps(w):
        for p in range(len(w)):
            for q in range(p+1,len(w)):
                v=list(w);v[p],v[q]=v[q],v[p];yield tuple(v)
    one={word,*swaps(word)}
    return one | {v for w in one for v in swaps(w)}


@pytest.mark.parametrize('length',[6,14])
def test_all_mitm_outputs_equal_literal_neighborhood(length):
    saw_multiple=False
    for target in canonical_target_content_profiles(length):
        wanted=tuple(v//4 for v in fold_profile(target_profile(length,target.k,target.eta),2))
        for ca,cb in paired_contents(target):
            a,b=(tuple([-1]*n+[0]*z+[1]*p) for n,z,p in (ca,cb))
            aa,bb=neighborhood(a),neighborhood(b)
            by_profile={}
            for v in bb:by_profile.setdefault(tuple(pacf(v)),[]).append(v)
            expected={(u,v) for u in aa for v in by_profile.get(tuple(x-y for x,y in zip(wanted,pacf(u))),())}
            result=repair_mitm(a,b,target,max_results=4096)
            assert result['complete']
            assert set(result['words'])==expected
            assert len(result['words'])==len(expected)
            if len(expected)>1:
                saw_multiple=True
                capped=repair_mitm(a,b,target,max_results=1)
                assert capped['status']=='SAT_TRUNCATED' and not capped['complete']
                assert len(capped['words'])==1 and capped['words'][0] in expected
    assert saw_multiple


def test_invalid_mitm_cap():
    t=canonical_target_content_profiles(6)[0]
    ca,cb=paired_contents(t)[0]
    a,b=(tuple([-1]*n+[0]*z+[1]*p) for n,z,p in (ca,cb))
    with pytest.raises(ValueError):repair_mitm(a,b,t,max_results=0)
