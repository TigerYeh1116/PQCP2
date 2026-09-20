"""Batching changes scheduling, not the raw stream, child order or proofs."""

from dataclasses import asdict,replace

import pytest

from solver import middle_mitm as mitm
from solver.compressed_completion import CompressedLiftJob
from solver.compressed_equivalence import compressed_job_equivalence_key as key
from solver.staged_completion import CompletionWorkStore,SlicePolicy
from solver.structured_energy import compress_binary_sequence
from solver.target_profiles import TargetContentProfile,pair_content


def job():
    a=tuple(map(int,'10111100001011110001100111101101101010101101'))
    b=tuple(map(int,'00101001010010011001111101100100010111001000'))
    return CompressedLiftJob(compress_binary_sequence(a,4),compress_binary_sequence(b,4),
                             TargetContentProfile(44,4,-1,*pair_content(a,b)),a,b,4)


def test_batch_and_singleton_have_identical_order_and_cursor():
    j=job();seen=set();cursor=0;single=[]
    for _ in range(4):
        r=mitm.next_middle(j.compressed_a,j.compressed_b,j.target,2000,cursor=cursor,excluded=seen)
        assert r['status']=='FOUND'
        single.extend(r['matches']);cursor=r['cursor']
        seen.add(key(r['a'],r['b'],j.target,2))
    batch=mitm.next_middle(j.compressed_a,j.compressed_b,j.target,2000,batch_size=4)
    assert list(batch['matches'])==single and batch['cursor']==cursor


def test_partial_batch_on_timeout_does_not_drop_discovered_children(monkeypatch):
    j=job();original=mitm.middle_matches
    def interrupted(*args,**kwargs):
        yield next(original(*args,**kwargs))
        raise mitm.BudgetExpired('controlled stop')
    monkeypatch.setattr(mitm,'middle_matches',interrupted)
    r=mitm.next_middle(j.compressed_a,j.compressed_b,j.target,2000,batch_size=4)
    assert r['status']=='FOUND' and len(r['matches'])==1 and not r['exhausted']
    assert r['cursor']==1 and 'stop' in r['reason']


def test_batch_checkpoint_is_atomic_and_model_cap_is_preserved(tmp_path,monkeypatch):
    j=job();policy=SlicePolicy(middle_mitm=True,affine_lift=True,rectangle_search=False,
                              mitm_batch_size=4,max_middle_models=3)
    store=CompletionWorkStore(tmp_path,44,123,policy)
    ident=store.register('lift',asdict(j));before=store.path(ident).read_bytes()
    import solver.checkpoint as checkpoint
    original=checkpoint.os.replace
    def fail(*_):raise OSError('controlled storage failure')
    monkeypatch.setattr(checkpoint.os,'replace',fail)
    with pytest.raises(OSError):store.advance_lift(ident)
    assert store.path(ident).read_bytes()==before
    monkeypatch.setattr(checkpoint.os,'replace',original)
    r=store.advance_lift(ident)
    state=store.read(ident)['state']
    assert len(state['children'])==r['mitm_batch_count']==3
    assert state['order']==[0,1,2,'enum']
    # Changing batch size on restart must not reset raw-match progress or lose
    # any already queued representative; all children remain separate work.
    restored=CompletionWorkStore(tmp_path,44,123,replace(policy,mitm_batch_size=1))
    assert restored.recover()==[ident]
    recovered=restored.read(ident)['state']
    assert recovered['children']==state['children']
    assert recovered['enum']['mitm']['cursor']==state['enum']['mitm']['cursor']


@pytest.mark.parametrize('size',(0,17,True,1.5))
def test_invalid_batch_cap(size):
    with pytest.raises(ValueError):SlicePolicy(mitm_batch_size=size)
    j=job()
    with pytest.raises(ValueError):
        mitm.next_middle(j.compressed_a,j.compressed_b,j.target,1000,batch_size=size)
