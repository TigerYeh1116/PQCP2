"""Exact pattern specialization, complete native decisions, partial rollback."""
import ctypes
from itertools import product
from random import Random

import pytest

from solver.structured_energy import integer_periodic_autocorrelation_profile as pacf
from solver.ternary_system_native import _library,prepare_native_system,run_native_system


def bound_functions(lib):
    extended=lib.pqcp_quadratic_extended_plan
    extended.argtypes=lib.pqcp_quadratic_plan.argtypes;extended.restype=ctypes.c_int
    specialize=lib.pqcp_specialize_quadratic_plan;u=ctypes.c_uint32
    specialize.argtypes=[u,ctypes.POINTER(u),ctypes.POINTER(u),ctypes.c_int,ctypes.c_int,ctypes.c_uint64,ctypes.POINTER(u)]
    specialize.restype=ctypes.c_int
    return extended,specialize


def constructed_plan(d,weights,negative,seed):
    rng=Random(seed);positions=tuple(rng.sample(range(d),w) for w in weights)
    neg=tuple(set(rng.sample(p,n)) for p,n in zip(positions,negative))
    rows=tuple(tuple((-1 if i in neg[s] else 1) if i in positions[s] else 0 for i in range(d)) for s in range(2))
    masks=tuple(sum(1<<i for i in p) for p in positions)
    wanted=tuple(x+y for x,y in zip(pacf(rows[0]),pacf(rows[1])))
    return prepare_native_system(masks,d,wanted,negative,force_side=0),rows


@pytest.mark.parametrize('d',(3,5,7,9,11))
def test_every_pattern_specializes_to_independent_integer_coefficient_construction(d):
    lib=_library();extended,specialize=bound_functions(lib);u=ctypes.c_uint32;word=ctypes.c_uint64
    p,_=constructed_plan(d,(d//2,d-d//2),(d//4,(d-d//2)//2),864000+d)
    assert p['consistent'];patterns=int(p['meta'][5]);free=int(p['meta'][4]);total=patterns+free
    toggles=tuple(p['patterns'][:2*patterns])+tuple(p['toggles'][:2*free])
    constant=u();linear=(u*total)();quadratic=(u*(total*total))()
    assert extended(p['mask_array'],p['base'],(word*(2*total))(*toggles),total,d,p['wanted'],p['negative'],
                    ctypes.byref(constant),linear,quadratic)==0
    for pattern in range(1<<patterns):
        initial=list(p['base'])
        for j in range(patterns):
            if pattern>>j&1:
                for s in range(2):initial[s]^=p['patterns'][2*j+s]
        c=u();l=(u*free)();q=(u*(free*free))()
        assert lib.pqcp_quadratic_plan(p['mask_array'],(word*2)(*initial),p['toggles'],free,d,p['wanted'],p['negative'],
                                      ctypes.byref(c),l,q)==0
        packed=(u*601)()
        assert specialize(constant,linear,quadratic,patterns,free,pattern,packed)==0
        assert packed[0]==c.value and tuple(packed[1:1+free])==tuple(l)
        assert tuple(packed[25:25+free*free])==tuple(q)


@pytest.mark.parametrize('seed',(864301,864302,864303))
def test_large_pattern_bank_keeps_exact_integer_controls_and_partial_assignment_order(seed):
    # Native arbitrary-norm controls, not claimed strict PQCPs.
    p,reference=constructed_plan(43,(14,29),(5,13),seed)
    assert p['consistent'] and p['selected']==0 and 7<=p['meta'][4]<=24
    assert p['compatible_pattern_count']>64
    complete=int(p['estimated_assignments'])
    old=run_native_system(p,complete,conditioned_hybrid=False)
    new=run_native_system(p,complete)
    assert new.get('conditioned_hybrid') and old['status']==new['status']=='SAT_COMPRESSED'
    for out in (old,new):
        assert tuple(w.count(-1) for w in out['words'])==tuple(p['negative'])
        assert tuple(x+y for x,y in zip(pacf(out['words'][0]),pacf(out['words'][1])))==tuple(p['wanted'])
    for cap in (1,complete-1):
        baseline=run_native_system(p,cap,conditioned_hybrid=False)
        bounded=run_native_system(p,cap)
        assert (bounded['status'],bounded['words'],bounded['assignments'])==(baseline['status'],baseline['words'],baseline['assignments'])
    # Keep the original GF2 / mod8 space, but violate the exact DC norm
    # identity by changing a symmetric target pair by eight. No word with
    # these two fixed negative counts can have the changed profile sum.
    unsigned=tuple(tuple(int(m>>i&1) for i in range(43)) for m in p['masks'])
    bounds=tuple(x+y for x,y in zip(pacf(unsigned[0]),pacf(unsigned[1])))
    fake=list(p['wanted'])
    changed=False
    for shift in range(1,22):
        for delta in (-8,8):
            if abs(fake[shift]+delta)<=bounds[shift]:
                fake[shift]+=delta;fake[43-shift]+=delta;changed=True;break
        if changed:break
    assert changed and sum(fake)!=sum(sum(w)**2 for w in reference)
    contradiction=dict(p,wanted=(ctypes.c_int*43)(*fake))
    baseline=run_native_system(contradiction,complete,conditioned_hybrid=False)
    cached=run_native_system(contradiction,complete)
    assert baseline['status']==cached['status']=='UNSAT_FIXED_SUPPORT'
    assert cached.get('conditioned_hybrid') and cached['words'] is None


@pytest.mark.parametrize('d',(3,5,7,9))
def test_complete_small_weight_fibres_against_all_original_literal_signs(d):
    p,_=constructed_plan(d,(d//2,d-d//2),(d//4,(d-d//2)//2),864200+d)
    positions=[(s,i) for s in range(2) for i in range(d) if p['masks'][s]>>i&1]
    feasible=set()
    for values in product((-1,1),repeat=d):
        rows=[[0]*d for _ in range(2)]
        for (s,i),v in zip(positions,values):rows[s][i]=v
        rows=tuple(tuple(w) for w in rows)
        if tuple(w.count(-1) for w in rows)!=tuple(p['negative']):continue
        if tuple(x+y for x,y in zip(pacf(rows[0]),pacf(rows[1])))==tuple(p['wanted']):feasible.add(rows)
    new=run_native_system(p,max(1,int(p['estimated_assignments'])),conditioned_hybrid=True,hybrid_tail=6)
    assert new['status']==('SAT_COMPRESSED' if feasible else 'UNSAT_FIXED_SUPPORT')
    if feasible:assert new['words'] in feasible
