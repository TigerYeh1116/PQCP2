import pytest
import torch

from solver.target_profiles import canonical_target_content_profiles
from solver.structured_energy import integer_periodic_autocorrelation_profile as pacf
from solver.torch_compressed_search import content_transfer_trials, search_compressed, ternary_profile


@pytest.mark.parametrize('device',['cpu','mps'])
def test_transfer_tensor_delta_matches_full_profile(device):
    if device=='mps' and not torch.backends.mps.is_available():pytest.skip('MPS unavailable')
    gen=torch.Generator().manual_seed(71)
    current=torch.randint(-1,2,(128,2,7),generator=gen).float().to(device)
    draws=torch.rand(128,16,8,generator=gen).to(device)
    lower=torch.zeros(128,2,device=device)
    upper=torch.full((128,2),7,device=device)
    trials,delta,legal=content_transfer_trials(current,draws,lower,upper)
    assert legal.any()
    full=ternary_profile(trials.reshape(-1,2,7)).reshape(128,16,7)
    predicted=ternary_profile(current)[:,None]+delta
    assert torch.equal(full[legal],predicted[legal])
    assert torch.all(delta[...,0][legal]==0)
    assert torch.equal(trials.sum(-1)[legal],current.sum(-1)[:,None].expand(-1,16,-1)[legal])


@pytest.mark.parametrize('device',['cpu','mps'])
def test_transfer_search_preserves_target_constraints_and_can_change_content(device):
    if device=='mps' and not torch.backends.mps.is_available():pytest.skip('MPS unavailable')
    ts=canonical_target_content_profiles(46)
    profiles=tuple(ts[i%len(ts)] for i in range(32))
    trace=[]
    result=search_compressed(profiles,device=torch.device(device),seed=789,steps=64,
                             transfer_interval=2,trace=trace)
    assert result.transfer_accepted>0 and result.transfer_proposals==32*32*8
    counts=[]
    for pairs,observed in trace:
        counts.append([[w.count(-1) for w in pair] for pair in pairs])
        for pair,profile,t in zip(pairs,observed,profiles):
            assert sum(v*v for w in pair for v in w)==23
            assert list(map(sum,zip(*(pacf(tuple(map(int,w))) for w in pair))))==profile
            for w,e,o in ((pair[0],t.a_even_ones,t.a_odd_ones),(pair[1],t.b_even_ones,t.b_odd_ones)):
                n,z=w.count(-1),w.count(0)
                assert e+o==2*n+z and 0<=e-n<=z and 0<=o-n<=z
    assert any(c!=counts[0] for c in counts[1:])
