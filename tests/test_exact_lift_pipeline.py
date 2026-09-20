"""Production MITM/affine integration, rollback and durable proof ownership."""

from dataclasses import asdict, replace
from itertools import product
import json
from pathlib import Path
import threading

import pytest

import main
from solver import staged_completion as staged
from solver.affine_lift import solve_affine_lift
from solver.compressed_completion import CompressedLiftJob
from solver.compressed_equivalence import compressed_job_equivalence_key
from solver.middle_mitm import middle_matches, next_middle
from solver.structured_energy import (
    compress_binary_sequence, fold_profile,
    integer_periodic_autocorrelation_profile as pacf, target_profile,
)
from solver.target_profiles import TargetContentProfile, pair_content
from solver.verifier import verify_pqcp


def job():
    # Existing random-source compression fixture, not a valid PQCP seed.
    a,b = tuple(map(int,'101110111100')),tuple(map(int,'111001110011'))
    assert not verify_pqcp(a,b).is_valid
    return CompressedLiftJob(compress_binary_sequence(a,4),compress_binary_sequence(b,4),
                             TargetContentProfile(12,1,1,*pair_content(a,b)),a,b,4)


def setup_store(root, **options):
    kwargs = dict(middle_mitm=True,affine_lift=True,rectangle_search=False)
    kwargs.update(options)
    store = staged.CompletionWorkStore(root,12,123,staged.SlicePolicy(**kwargs))
    return store,store.register('lift',asdict(job()))


@pytest.mark.parametrize('ca,cb,p',(
    ((4,),(0,),TargetContentProfile(4,1,1,0,0,1,1)),
    ((2,0),(2,0),TargetContentProfile(8,2,-1,1,2,1,2)),
    ((4,0,0),(0,0,0),TargetContentProfile(12,3,-1,1,3,2,4)),
))
def test_production_middle_join_matches_entire_ternary_oracle(ca,cb,p):
    n = len(ca)
    pools = []
    for c,e,o in ((ca,p.a_even_ones,p.a_odd_ones),(cb,p.b_even_ones,p.b_odd_ones)):
        pools.append([w for w in product((-2,0,2),repeat=2*n)
                      if tuple(w[i]+w[i+n] for i in range(n))==c
                      and sum(w[::2])==2*n-2*e and sum(w[1::2])==2*n-2*o])
    target = fold_profile(target_profile(p.L,p.k,p.eta),2)
    expected = {(a,b) for a,b in product(*pools)
                if tuple(x+y for x,y in zip(pacf(a),pacf(b)))==target}
    actual = list(middle_matches(ca,cb,p,10000,lambda:None))
    assert len(actual)==len(set(actual)) and set(actual)==expected


def test_middle_cursor_and_equivalence_dedup_cover_all_orbits():
    j = job()
    all_models = list(middle_matches(j.compressed_a,j.compressed_b,j.target,10000,lambda:None))
    expected = {compressed_job_equivalence_key(a,b,j.target,2) for a,b in all_models}
    cursor = 0
    seen = set()
    duplicates = 0
    while True:
        r = next_middle(j.compressed_a,j.compressed_b,j.target,2000,cursor=cursor,excluded=seen)
        assert r['cursor']>=cursor
        cursor = r['cursor']
        duplicates += r['duplicates']
        if r['status']=='EXHAUSTED':
            break
        assert r['status']=='FOUND'
        key = compressed_job_equivalence_key(r['a'],r['b'],j.target,2)
        assert key not in seen
        seen.add(key)
    assert seen==expected and cursor==len(all_models)
    assert duplicates==len(all_models)-len(seen)>0


def test_middle_caps_defer_never_prove_unsat(monkeypatch):
    j=job()
    r=next_middle(j.compressed_a,j.compressed_b,j.target,2000,max_index_entries=1)
    assert r['status']=='DEFERRED' and 'cap' in r['reason']
    ticks=iter((0,2,3,4,5))
    monkeypatch.setattr('solver.middle_mitm.perf_counter',lambda:next(ticks))
    r=next_middle(j.compressed_a,j.compressed_b,j.target,1)
    assert r['status']=='DEFERRED' and 'time' in r['reason']


def test_impossible_saved_middle_cursor_fails_closed():
    j=job()
    with pytest.raises(ValueError,match='cursor beyond'):
        next_middle(j.compressed_a,j.compressed_b,j.target,1000,cursor=1000000)


def test_real_pipeline_sat_saved_verified_and_not_counted_as_z3(tmp_path):
    store,key=setup_store(tmp_path)
    middle=store.advance_lift(key)
    assert middle['middle_backend']=='mitm' and middle['phase']=='READY'
    assert middle['z3_calls']==0 and not (tmp_path/'12.txt').exists()
    report=store.advance_lift(key)
    assert report['phase']=='SAT' and report['binary_backend']=='affine'
    assert report['z3_calls']==0 and not report['z3_executed']
    saved=json.loads(Path(report['snapshot']).read_text())
    assert saved['method']=='affine_compressed_lift'
    assert verify_pqcp(saved['A'],saved['B']).is_valid
    assert saved['algorithm_parameters']['affine_rank']==2
    assert (tmp_path/'12.txt').exists() and store.recover()==[]


@pytest.mark.parametrize('options,field',(
    ({'mitm_index_entries':1},'middle_backend'),
    ({'affine_max_assignments':1},'binary_backend'),
    ({'middle_mitm':False},'middle_backend'),
    ({'affine_lift':False},'binary_backend'),
))
def test_real_z3_fallback_and_rollback(tmp_path,options,field):
    store,key=setup_store(tmp_path,**options)
    r=store.advance_lift(key)
    if field=='binary_backend':
        r=store.advance_lift(key)
    assert r[field]=='z3' and r['z3_calls']==1 and r['z3_executed']
    if field=='middle_backend':
        assert store.read(key)['state']['children']
    else:
        assert r['solved']


def test_unknown_representative_is_retained_after_equivalent_join_exhaustion(tmp_path,monkeypatch):
    # Only fake the child timeout, never the middle enumeration or key.
    from solver.compressed_lift import CompressedLiftResult
    unknown=CompressedLiftResult('UNKNOWN',12,1,1,6,5,4,0,0,True,reason='controlled timeout')
    monkeypatch.setattr(staged,'solve_compressed_lift',lambda *_a,**_k:unknown)
    store,key=setup_store(tmp_path,affine_lift=False,max_middle_models=100,maximum_ms=1000)
    for _ in range(30):
        r=store.advance_lift(key)
        if r['phase']!='READY':break
    state=store.read(key)['state']
    assert state['enum']['status']=='UNSAT'
    assert state['enum']['mitm']['duplicates']>0
    assert state['children'] and all(c['status']=='WAITING' for c in state['children'])
    assert r['phase']=='WAITING' and r['status']=='LIFT_UNKNOWN'
    assert store.recover()==[key]


def test_parent_unsat_requires_every_child_proof(tmp_path,monkeypatch):
    from solver.compressed_lift import CompressedLiftResult
    proof=CompressedLiftResult('UNSAT',12,1,1,6,5,4,0,0,True,reason='controlled child proof')
    monkeypatch.setattr(staged,'solve_compressed_lift',lambda *_a,**_k:proof)
    store,key=setup_store(tmp_path,affine_lift=False,max_middle_models=100)
    for _ in range(30):
        r=store.advance_lift(key)
        if r['phase']!='READY':break
    state=store.read(key)['state']
    assert state['enum']['status']=='UNSAT'
    assert state['children'] and all(c['status']=='UNSAT' for c in state['children'])
    assert r['phase']=='UNSAT'


def test_legacy_checkpoint_resume_and_mitm_cursor_persistence(tmp_path):
    legacy,key=setup_store(tmp_path/'legacy',middle_mitm=False,affine_lift=False)
    legacy.advance_lift(key)
    old_child=legacy.read(key)['state']['children'][0]
    resumed=staged.CompletionWorkStore(tmp_path/'legacy',12,123,
        staged.SlicePolicy(middle_mitm=True,affine_lift=True,rectangle_search=False))
    assert resumed.recover()==[key]
    assert resumed.read(key)['state']['children'][0]==old_child
    report=resumed.advance_lift(key)
    assert report['binary_backend']=='affine' and report['solved']
    current,key=setup_store(tmp_path/'current')
    current.advance_lift(key)
    before=current.read(key)['state']
    restored=staged.CompletionWorkStore(tmp_path/'current',12,123,current.policy)
    assert restored.recover()==[key]
    assert restored.read(key)['state']['enum']['mitm']==before['enum']['mitm']
    assert restored.advance_lift(key)['solved']


def test_affine_cursor_resumes_a_real_L68_child_without_replaying(tmp_path,monkeypatch):
    # Fixed middle fixture from the prior UNSOLVED compression experiment.
    # Only two assignments per slice, not a long benchmark or a PQCP seed.
    a=(2,0,0,0,0,2,2,0,0,0,-2,0,2,-2,0,2,0,2,-2,0,-2,0,0,2,-2,2,0,-2,-2,0,2,-2,-2,0)
    b=(-2,-2,-2,-2,2,0,0,2,0,2,0,2,0,0,0,0,2,2,0,0,2,0,-2,2,0,2,2,0,0,0,0,2,0,0)
    p=TargetContentProfile(68,1,1,19,15,16,12)
    def source(w):
        return tuple(int(x==-2) for x in w)+tuple(int(x!=2) for x in w)
    j=CompressedLiftJob(a,b,p,source(a),source(b),2)
    def sliced(*args,**kwargs):
        return solve_affine_lift(*args,**kwargs,max_steps=2)
    monkeypatch.setattr('solver.affine_lift.solve_affine_lift',sliced)
    policy=staged.SlicePolicy(affine_lift=True,rectangle_search=False,maximum_ms=1000)
    store=staged.CompletionWorkStore(tmp_path,68,123,policy)
    key=store.register('lift',asdict(j))
    first=store.advance_lift(key)
    assert first['affine_next_index']==2 and first['phase']=='WAITING'
    restored=staged.CompletionWorkStore(tmp_path,68,123,policy)
    assert restored.recover()==[key]
    second=restored.advance_lift(key)
    assert second['affine_next_index']==4 and second['affine_examined']==2
    assert second['status']=='LIFT_UNKNOWN' and second['z3_calls']==0
    assert not (tmp_path/'68.txt').exists()


def test_main_defaults_enable_stages_and_flags_disable_them(tmp_path):
    for L in (44,68):
        args=main.parse_args(['--L',str(L)])
        bridge=main._CCompletionBridge(args,L,tmp_path/str(L))
        try:
            assert bridge._staged_store.policy.middle_mitm
            assert bridge._staged_store.policy.affine_lift
        finally:bridge.close()
    args=main.parse_args(['--L','68','--no-z3-middle-mitm','--no-z3-affine-lift'])
    bridge=main._CCompletionBridge(args,68,tmp_path/'rollback')
    try:
        assert not bridge._staged_store.policy.middle_mitm
        assert not bridge._staged_store.policy.affine_lift
    finally:bridge.close()


def test_main_background_worker_uses_new_stages_and_correct_call_counter(tmp_path,capsys):
    store,key=setup_store(tmp_path)
    done=threading.Event()
    args=main.parse_args(['--L','12','--seed','123','--no-z3-rectangle-search'])
    bridge=main._CCompletionBridge(args,12,tmp_path,
        lambda r:done.set() if r.get('solved') else None)
    try:
        assert done.wait(5)
    finally:bridge.close()
    assert bridge.z3_execution_count==0
    assert bridge.valid_result_count==1
    assert store.read(key)['phase']=='SAT'
    output=capsys.readouterr().out
    assert '中間層 MITM：FOUND' in output
    assert 'child=SAT parent=LIFT_SAT' in output


@pytest.mark.parametrize('options',({'middle_mitm':1},{'affine_lift':1},
                                    {'mitm_index_entries':0},{'affine_max_assignments':0}))
def test_invalid_policy(options):
    with pytest.raises(ValueError):staged.SlicePolicy(**options)
