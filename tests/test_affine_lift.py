"""Exact arithmetic and resumable affine lift checks; no long benchmarks."""

from dataclasses import replace
from itertools import islice, product
import random

import pytest

from solver.affine_lift import (
    affine_basis, affine_binary_pairs, orientation_parity_system, solve_affine_lift,
)
from solver.correlation import full_correlation_profile
from solver.structured_energy import (
    fold_profile, integer_periodic_autocorrelation_profile as pacf, target_profile,
)
from solver.target_profiles import TargetContentProfile
from solver.verifier import verify_pqcp


def literal_lifts(word):
    """Independent bit-array oracle, not Gray or bitset based."""
    d = len(word)
    positions = [i for i,v in enumerate(word) if v == 0]
    for choices in product((0,1),repeat=len(positions)):
        bits = [int(v == -2) for v in word]*2
        for i,b in zip(positions,choices):
            bits[i],bits[i+d] = b,1-b
        yield tuple(bits)


def content(a,b,k,eta):
    counts = tuple(sum(2 if v==-2 else 1 if v==0 else 0 for v in w[p::2])
                   for w in (a,b) for p in (0,1))
    return TargetContentProfile(2*len(a),k,eta,*counts)


def small_sat():
    return ((-2,0,-2,-2,0,2),(0,-2,-2,0,0,0),
            TargetContentProfile(12,1,1,5,3,4,4))


def test_mod4_system_and_gray_suffix_against_literal_assignments():
    rng = random.Random(968)
    for _ in range(30):
        a,b = (tuple(rng.choice((-2,0,2)) for i in range(6)) for s in (0,1))
        h = [tuple(rng.choice((-1,1)) if x==0 else 0 for x in w) for w in (a,b)]
        target = tuple(sum(w[i]*w[(i+u)%6]*(1 if i+u<6 else -1)
                           for w in h for i in range(6)) for u in range(6))
        system = orientation_parity_system(a,b,target)
        assert system['consistent']
        expected = set()
        for aa,bb in product(literal_lifts(a),literal_lifts(b)):
            hs = [tuple((1-2*w[i]-(1-2*w[i+6]))//2 for i in range(6)) for w in (aa,bb)]
            observed = tuple(sum(w[i]*w[(i+u)%6]*(1 if i+u<6 else -1)
                                 for w in hs for i in range(6)) for u in range(6))
            if all((observed[u]-target[u])%4 == 0 for u in (1,2)):
                expected.add(tuple(sum(v<<i for i,v in enumerate(w)) for w in (aa,bb)))
        generated = list(affine_binary_pairs(a,b,system))
        assert len(generated) == len(set(generated)) == system['affine_assignments']
        assert set(generated) == expected
        for cursor in (0,1,len(generated)//2,len(generated)):
            assert list(affine_binary_pairs(a,b,system,cursor)) == generated[cursor:]
        particular,kernel = affine_basis(system)
        assert len(kernel) == len(system['variables'])-system['rank']
        assert all((bin(particular&e['mask']).count('1')&1)==e['rhs'] for e in system['equations'])


def test_full_lift_against_exhaustive_binary_oracle():
    # Forty exact compressed matches spanning every L12 target/sign, built
    # from the entire ternary space rather than an existing PQCP archive.
    words = list(product((-2,0,2),repeat=6))
    random.Random(12).shuffle(words)
    signatures = {w:pacf(w) for w in words}
    statuses = set()
    for k,eta in product(range(1,6),(-1,1)):
        target = target_profile(12,k,eta)
        folded = fold_profile(target,2)
        buckets = {}
        for w,sig in signatures.items():
            buckets.setdefault(sig,[]).append(w)
        matches = ((a,b) for a in words
                   for b in buckets.get(tuple(t-x for t,x in zip(folded,signatures[a])),()))
        for a,b in islice(matches,4):
            possible = any(tuple(full_correlation_profile(aa,bb)) == target
                           for aa,bb in product(literal_lifts(a),literal_lifts(b)))
            result = solve_affine_lift(a,b,k,eta,2000,content(a,b,k,eta))
            assert result.status == ('SAT' if possible else 'UNSAT')
            statuses.add(result.status)
            assert not result.z3_executed
            if result.verified:
                assert verify_pqcp(result.a,result.b).is_valid
    assert statuses == {'SAT','UNSAT'}


def test_contradictory_mod4_conditions_are_exact_unsat():
    a,b = (-2,-2,0,-2,2,0),(-2,0,0,2,0,0)
    r = solve_affine_lift(a,b,3,-1,1000,content(a,b,3,-1))
    assert r.status == 'UNSAT' and r.examined == 0 and not r.z3_executed
    assert 'inconsistent' in r.reason


def test_resuming_all_gray_slices_is_identical_to_continuous():
    a,b,p = small_sat()
    # Empty cross disjunction excludes every lift; this tests exhaustion and
    # cursor bookkeeping independently of where a SAT would stop the search.
    whole = solve_affine_lift(a,b,p.k,p.eta,1000,p,cross_compressions=())
    cursor = total = 0
    while True:
        part = solve_affine_lift(a,b,p.k,p.eta,1000,p,cross_compressions=(),
                                 start_index=cursor,max_steps=3)
        assert part.next_index == cursor+part.examined
        cursor = part.next_index
        total += part.examined
        if part.status != 'UNKNOWN':
            break
        assert not part.deferred and cursor < part.assignments
    assert whole.status == part.status == 'UNSAT'
    assert total == whole.examined == whole.assignments == cursor


def test_cap_timeout_content_and_cross_are_not_ignored(monkeypatch):
    a,b,p = small_sat()
    r = solve_affine_lift(a,b,p.k,p.eta,1000,p,max_assignments=1)
    assert r.status == 'UNKNOWN' and r.deferred and r.examined == 0
    assert solve_affine_lift(a,b,p.k,p.eta,1000,p).verified
    bad_content = replace(p,a_even_ones=p.a_even_ones-1)
    assert solve_affine_lift(a,b,p.k,p.eta,1000,bad_content).status == 'UNSAT'
    ticks = iter((0,1,2,3,4,5))
    monkeypatch.setattr('solver.affine_lift.perf_counter',lambda:next(ticks))
    r = solve_affine_lift(a,b,p.k,p.eta,1,p)
    assert r.status == 'UNKNOWN' and not r.deferred and r.next_index == 0


def test_odd_half_length_defers_to_z3():
    # A matched factor-2 problem with L6, not an invalid profile.
    desired = fold_profile(target_profile(6,1,1),2)
    words = list(product((-2,0,2),repeat=3))
    a,b = next((a,b) for a,b in product(words,repeat=2)
               if tuple(x+y for x,y in zip(pacf(a),pacf(b)))==desired)
    r = solve_affine_lift(a,b,1,1,1000)
    assert r.status == 'UNKNOWN' and r.deferred


@pytest.mark.parametrize('kwargs',({'timeout_ms':0},{'max_assignments':0},
                                   {'start_index':-1},{'max_steps':0}))
def test_invalid_budgets(kwargs):
    a,b,p = small_sat()
    options = dict(timeout_ms=1000)
    options.update(kwargs)
    with pytest.raises(ValueError):
        solve_affine_lift(a,b,1,1,content_profile=p,**options)


def test_verifier_failure_is_an_error_not_a_saved_sat(monkeypatch):
    from types import SimpleNamespace
    a,b,p = small_sat()
    monkeypatch.setattr('solver.affine_lift.verify_pqcp',
                        lambda *_:SimpleNamespace(is_valid=False))
    with pytest.raises(RuntimeError,match='independent'):
        solve_affine_lift(a,b,1,1,1000,p)
