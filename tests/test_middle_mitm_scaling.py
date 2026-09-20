"""Exhaustive small-domain proofs for diagnostics; no long search in pytest."""
from collections import Counter
from itertools import product, islice
import random

import pytest

from experiments.analyze_middle_mitm_scaling import (
    negacyclic_profile, split_channels, orientations_with_rows, residual_target,
    middle_metrics, probe_binary,
    orientation_parity_system, affine_basis, enumerate_affine_lift,
    affine_binary_pairs,
)
from solver.structured_energy import integer_periodic_autocorrelation_profile as pacf, target_profile
from solver.structured_energy import fold_profile
from solver.correlation import full_correlation_profile
from solver.target_profiles import TargetContentProfile
from dataclasses import asdict


@pytest.mark.parametrize('L',(4,8,12))
def test_full_cyclic_decomposition_exhaustive(L):
    d=L//2
    for bits in product((0,1),repeat=L):
        c,h=split_channels(bits)
        p,n= pacf(c),negacyclic_profile(h)
        full=pacf(tuple(1-2*x for x in bits))
        assert all(2*full[u]==p[u]+4*n[u] for u in range(d))
        assert all(2*full[u+d]==p[u]-4*n[u] for u in range(d))
        assert all(n[d-u]==-n[u] for u in range(1,d))
        assert n[d//2]==0


@pytest.mark.parametrize('n',(1,3))
def test_binomial_count_equals_all_binary_lifts(n):
    counts=Counter()
    for bits in product((0,1),repeat=4*n):
        c,_=split_channels(bits)
        rows=tuple(sum(1-2*b for b in bits[r::4]) for r in range(4))
        counts[c,rows]+=1
    for (c,rows),count in counts.items():
        assert orientations_with_rows(c,rows)==count
    assert orientations_with_rows((0,)*(2*n),(n,n,n,n))==0


@pytest.mark.parametrize('L',(44,68))
def test_target_aliasing_and_independent_residual_shifts(L):
    d=L//2
    for k in (1,2,4,L//4):
        for eta in (-1,1):
            residual=residual_target(L,k,eta)
            assert residual[0]==d and residual[d//2]==0
            assert all(residual[d-u]==-residual[u] for u in range(1,d))
            if k==L//4:
                assert not any(residual[1:])
            else:
                assert residual[k]==eta and residual[d-k]==-eta


def test_wrong_alphabets_and_lengths_rejected():
    with pytest.raises(ValueError): split_channels((0,1,0))
    with pytest.raises(ValueError): split_channels((0,2))
    with pytest.raises(ValueError): orientations_with_rows((3,0),(1,1,1,1))
    with pytest.raises(ValueError): negacyclic_profile(())


def test_exact_matched_middle_count_and_probe():
    p=TargetContentProfile(4,1,1,0,0,1,1)
    metrics=middle_metrics((2,2),(0,0),p)
    assert metrics['orientation_bits']==2
    assert metrics['cross_constrained_orientations']==4
    result=probe_binary(dict(target=asdict(p),middle_a=(2,2),middle_b=(0,0)),2)
    assert result['status']=='SAT' and result['verified']


def test_mismatched_middle_rejected():
    p=TargetContentProfile(4,1,1,0,0,1,1)
    with pytest.raises(ValueError,match='match'):
        middle_metrics((2,2),(2,2),p)
    with pytest.raises(ValueError,match='content'):
        middle_metrics((2,2),(0,0),TargetContentProfile(4,1,1,1,0,1,1))


def test_parity_affine_basis_and_mod4_equations_against_exhaustive_signs():
    rng=random.Random(19068)
    for _ in range(20):
        ma,mb=(tuple(rng.choice((-2,0,2)) for i in range(6)) for s in range(2))
        initial=[tuple(rng.choice((-1,1)) if x==0 else 0 for x in w) for w in (ma,mb)]
        target=tuple(a+b for a,b in zip(negacyclic_profile(initial[0]),negacyclic_profile(initial[1])))
        system=orientation_parity_system(ma,mb,target)
        assert system['consistent']
        particular,kernel=affine_basis(system)
        generated=set()
        for choices in product((0,1),repeat=len(kernel)):
            word=particular
            for include,delta in zip(choices,kernel):
                if include:word^=delta
            generated.add(word)
        oracle=set()
        for word in range(1<<len(system['variables'])):
            hs=[[0]*6,[0]*6]
            for bit,(s,i) in enumerate(system['variables']):hs[s][i]=1-2*((word>>bit)&1)
            actual=tuple(a+b for a,b in zip(negacyclic_profile(hs[0]),negacyclic_profile(hs[1])))
            if all((actual[u]-target[u])%4==0 for u in (1,2)):oracle.add(word)
        assert generated==oracle
        assert len(generated)==system['affine_assignments']


def test_cross_compatible_middle_need_not_have_a_binary_solution():
    # Derived by independent exhaustive L12 orientation enumeration, not a
    # known PQCP. Both compressed levels and row margins pass. Two remaining
    # sign equations demand the SAME four-sign product be -1 and +1.
    ma,mb=(-2,-2,0,-2,2,0),(-2,0,0,2,0,0)
    p=TargetContentProfile(12,3,-1,3,5,4,2)
    metrics=middle_metrics(ma,mb,p)
    assert metrics['cross_constrained_orientations']==32
    system=orientation_parity_system(ma,mb,metrics['residual_target'])
    assert not system['consistent'] and system['affine_assignments']==0
    request=dict(target=asdict(p),middle_a=ma,middle_b=mb)
    assert enumerate_affine_lift(request,1)['status']=='UNSAT'


def test_affine_probe_sat_and_time_limit():
    request=dict(target=asdict(TargetContentProfile(4,1,1,0,0,1,1)),middle_a=(2,2),middle_b=(0,0))
    assert enumerate_affine_lift(request,1)['verified']
    result=enumerate_affine_lift(request,1e-12)
    assert result['status']=='UNKNOWN' and not result['verified']


def test_gray_enumerator_against_all_lifts_of_small_matched_pairs():
    # Independent literal bit-array construction and baseline correlation,
    # not the bitset/Gray implementation under test. No prior solution file.
    from experiments.research_compressed_mitm import signature_join
    words=list(product((-2,0,2),repeat=6))
    random.Random(20260919).shuffle(words)
    seen_status=set()
    def lifts(word):
        positions=[i for i,v in enumerate(word) if v==0]
        for choices in product((0,1),repeat=len(positions)):
            bits=[int(v==-2) for v in word]*2
            for i,b in zip(positions,choices):bits[i]=b;bits[i+6]=1-b
            yield tuple(bits)
    for k,eta in product(range(1,6),(-1,1)):
        target=target_profile(12,k,eta)
        for ma,mb in islice(signature_join(words,words,fold_profile(target,2)),4):
            content=tuple(sum(2 if v==-2 else 1 if v==0 else 0 for v in w[p::2])
                          for w in (ma,mb) for p in (0,1))
            p=TargetContentProfile(12,k,eta,*content)
            possible=any(tuple(full_correlation_profile(a,b))==target
                         for a,b in product(tuple(lifts(ma)),tuple(lifts(mb))))
            result=enumerate_affine_lift(dict(target=asdict(p),middle_a=ma,middle_b=mb),2)
            assert result['status']==('SAT' if possible else 'UNSAT')
            if not possible and result['parity_system']['consistent']:
                assert result['enumerated']==result['parity_system']['affine_assignments']
                seen_status.add('complete_nonlinear_unsat')
            seen_status.add(result['status'])
    assert {'SAT','UNSAT'}<=seen_status


def test_gray_word_mapping_is_bijective_and_matches_literal_lifts():
    rng=random.Random(68)
    for _ in range(30):
        ma,mb=(tuple(rng.choice((-2,0,2)) for i in range(6)) for s in range(2))
        hs=[tuple(rng.choice((-1,1)) if x==0 else 0 for x in w) for w in (ma,mb)]
        target=tuple(a+b for a,b in zip(negacyclic_profile(hs[0]),negacyclic_profile(hs[1])))
        system=orientation_parity_system(ma,mb,target)
        actual=list(affine_binary_pairs(ma,mb,system))
        expected=set()
        for assignment in range(1<<len(system['variables'])):
            if any((bin(assignment & eq['mask']).count('1')&1)!=eq['rhs'] for eq in system['equations']):continue
            arrays=[[int(v==-2) for v in w]*2 for w in (ma,mb)]
            for bit,(s,i) in enumerate(system['variables']):
                value=(assignment>>bit)&1
                arrays[s][i]=value;arrays[s][i+6]=1-value
            expected.add(tuple(sum(v<<i for i,v in enumerate(bits)) for bits in arrays))
        assert len(actual)==len(set(actual))==system['affine_assignments']
        assert set(actual)==expected
