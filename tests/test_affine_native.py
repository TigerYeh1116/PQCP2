"""C/reference equivalence, including >64-bit words, SAT order and resume."""

import random

import pytest

from solver.affine_lift import solve_affine_lift
from solver.affine_native import NativeAffineScan, NativeUnavailable, _library
from solver.correlation import full_correlation_profile
from solver.target_profiles import TargetContentProfile


@pytest.fixture(scope='module')
def native():
    try:
        return _library()
    except (NativeUnavailable,OSError) as error:
        pytest.skip(str(error))


def state_at(base,deltas,index):
    gray=index^(index>>1)
    a,b=base
    for j,(da,db) in enumerate(deltas):
        if gray>>j&1:a^=da;b^=db
    return a,b


@pytest.mark.parametrize('L',(4,8,12,44,64,68,94,128))
def test_native_complete_first_match_and_cursor_vs_scalar_pacf(native,L):
    rng=random.Random(606+L)
    for free in (0,1,5):
        base=(rng.getrandbits(L),rng.getrandbits(L))
        deltas=tuple((rng.getrandbits(L),rng.getrandbits(L)) for _ in range(free))
        size=1<<free
        profiles=[]
        for i in range(size):
            a,b=state_at(base,deltas,i)
            aa,bb=(tuple((w>>p)&1 for p in range(L)) for w in (a,b))
            profiles.append(tuple(full_correlation_profile(aa,bb)))
        for chosen in (0,size//2,size-1):
            wanted=tuple(L-v//2 for v in profiles[chosen][1:L//2])
            scanner=NativeAffineScan(L,base,deltas,wanted,native)
            for start,end in ((0,size),(size//2,size),(0,size//2),(size,size)):
                matches=[i for i in range(start,end)
                         if profiles[i][1:L//2]==profiles[chosen][1:L//2]]
                cursor,found=scanner.scan(start,end)
                assert cursor==(matches[0]+1 if matches else end)
                assert found==(state_at(base,deltas,matches[0]) if matches else None)


def small_case():
    return (-2,0,-2,-2,0,2),(0,-2,-2,0,0,0),TargetContentProfile(12,1,1,5,3,4,4)


@pytest.mark.parametrize('cross',(None,()))
def test_exact_solver_python_native_same_sat_and_exhaustion(native,cross):
    a,b,p=small_case()
    results=[solve_affine_lift(a,b,1,1,2000,p,cross_compressions=cross,backend=backend)
             for backend in ('python','native')]
    x,y=results
    assert (x.status,x.a,x.b,x.profile,x.examined,x.next_index)==(
        y.status,y.a,y.b,y.profile,y.examined,y.next_index)
    assert x.scan_backend=='python' and y.scan_backend=='native'


def test_resume_can_switch_backends_without_replay(native):
    a,b,p=small_case()
    expected=solve_affine_lift(a,b,1,1,2000,p,cross_compressions=(),backend='python')
    cursor=examined=0
    for backend in ('native','python','native','python','native','python'):
        r=solve_affine_lift(a,b,1,1,2000,p,cross_compressions=(),backend=backend,
                            start_index=cursor,max_steps=3)
        assert r.next_index==cursor+r.examined
        cursor=r.next_index;examined+=r.examined
        if r.status=='UNSAT':break
        assert r.status=='UNKNOWN'
    assert r.status==expected.status=='UNSAT'
    assert cursor==examined==expected.examined


def test_missing_compiler_auto_falls_back_and_explicit_native_fails(monkeypatch):
    from solver import affine_native
    a,b,p=small_case()
    monkeypatch.setattr(affine_native,'optional_library',lambda:None)
    r=solve_affine_lift(a,b,1,1,2000,p)
    assert r.verified and r.scan_backend=='python'
    def fail():raise NativeUnavailable('test toolchain unavailable')
    monkeypatch.setattr(affine_native,'_library',fail)
    with pytest.raises(NativeUnavailable):
        solve_affine_lift(a,b,1,1,2000,p,backend='native')


def test_native_error_is_never_silently_converted_to_unsat(native,monkeypatch):
    a,b,p=small_case()
    def bad(*_):raise RuntimeError('invalid native cursor')
    monkeypatch.setattr(NativeAffineScan,'scan',bad)
    with pytest.raises(RuntimeError,match='cursor'):
        solve_affine_lift(a,b,1,1,2000,p,backend='auto')


def test_bounds_reject_truncation(native):
    with pytest.raises(ValueError):NativeAffineScan(132,(0,0),(),(0,)*65,native)
    with pytest.raises(ValueError):NativeAffineScan(68,(1<<68,0),(),(0,)*33,native)
    with pytest.raises(ValueError):NativeAffineScan(68,(0,0),((0,0),)*63,(0,)*33,native)
    scan=NativeAffineScan(68,(0,0),(),(0,)*33,native)
    with pytest.raises(ValueError):scan.scan(0,2)
    with pytest.raises(ValueError):scan.scan(-1,0)
