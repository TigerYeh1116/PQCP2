import ctypes
from itertools import product
from random import Random

import pytest

from solver.structured_energy import integer_periodic_autocorrelation_profile as pacf
from solver.ternary_system_native import _library,prepare_native_system,run_native_system


def count_filtered(plan,budget,tail):
    u=ctypes.c_uint64;output=(u*2)();stats=(u*7)();q=int(plan['meta'][2])
    code=_library().pqcp_quadratic_count_hybrid_signs(plan['mask_array'],plan['base'],plan['toggles'],q,
        plan['d'],plan['wanted'],plan['negative'],tail,budget,output,stats)
    assert code in (0,1,2)
    words=None if code!=1 else tuple(tuple((-1 if output[s]>>i&1 else 1) if plan['masks'][s]>>i&1 else 0
        for i in range(plan['d'])) for s in range(2))
    return ('UNSAT_FIXED_SUPPORT','SAT_COMPRESSED','UNKNOWN')[code],words,tuple(map(int,stats))


def test_linear_count_quota_eliminates_half_an_actual_affine_space():
    plan=prepare_native_system((1,15),5,(5,1,-1,-1,1),(0,1),conditioning=False)
    assert plan['consistent'] and plan['meta'][2]==2
    status,words,stats=count_filtered(plan,4,6)
    assert status=='SAT_COMPRESSED'
    assert tuple(w.count(-1) for w in words)==(0,1)
    assert tuple(x+y for x,y in zip(pacf(words[0]),pacf(words[1])))==(5,1,-1,-1,1)
    assert 2<stats[0]<=4  # half of the original assignments excluded by the necessary count equation


@pytest.mark.parametrize('d',(3,5,7,9,11))
def test_count_affine_filter_preserves_all_original_exact_sign_solutions_and_partial_order(d):
    rng=Random(866000+d)
    for case in range(10):
        active=set(rng.sample(range(2*d),d))
        masks=tuple(sum(1<<i for i in range(d) if s*d+i in active) for s in range(2))
        references=tuple(tuple(rng.choice((-1,1)) if m>>i&1 else 0 for i in range(d)) for m in masks)
        wanted=tuple(x+y for x,y in zip(pacf(references[0]),pacf(references[1])))
        if case%2:
            limit=sum(pacf(tuple(int(m>>i&1) for i in range(d)))[1] for m in masks)
            alternatives=[v for v in (wanted[1]-4,wanted[1]+4) if abs(v)<=limit]
            if alternatives:wanted=(wanted[0],alternatives[0])+wanted[2:-1]+(alternatives[0],)
        negative=tuple(w.count(-1) for w in references)
        plan=prepare_native_system(masks,d,wanted,negative,conditioning=False)
        if not plan['consistent']:continue
        positions=[(s,i) for s in range(2) for i in range(d) if masks[s]>>i&1]
        feasible=set()
        for values in product((-1,1),repeat=d):
            rows=[[0]*d for _ in range(2)]
            for (s,i),v in zip(positions,values):rows[s][i]=v
            rows=tuple(tuple(w) for w in rows)
            if (tuple(w.count(-1) for w in rows)==negative
                    and tuple(x+y for x,y in zip(pacf(rows[0]),pacf(rows[1])))==wanted):feasible.add(rows)
        complete=1<<plan['meta'][2]
        for tail in (1,3,6,8):
            status,words,stats=count_filtered(plan,complete,tail)
            assert status==('SAT_COMPRESSED' if feasible else 'UNSAT_FIXED_SUPPORT')
            if feasible:assert words in feasible
            else:assert words is None and stats[0]==complete
            if complete>1:
                old=run_native_system(plan,complete-1,quadratic=True,hybrid_tail=tail)
                bounded,bounded_words,bounded_stats=count_filtered(plan,complete-1,tail)
                assert (bounded,bounded_words,bounded_stats[0])==(old['status'],old['words'],old['assignments'])


@pytest.mark.parametrize('seed,weights,negative',((862601,(3,40),(1,17)),(862602,(3,40),(1,17)),
    (862603,(3,40),(1,17)),(863010,(10,33),(2,15)),(863011,(10,33),(2,15)),(863012,(10,33),(2,15))))
def test_l43_affine_count_reduction_preserves_every_modular_model_and_halves_prefixes(seed,weights,negative):
    # Constructed arbitrary integer norm: these are native-kernel controls,
    # not strict PQCP solutions. One row is fixed in a constant-weight fibre.
    rng=Random(seed);d=43
    positions=tuple(rng.sample(range(d),w) for w in weights)
    neg=tuple(set(rng.sample(pos,n)) for pos,n in zip(positions,negative))
    rows=tuple(tuple((-1 if i in neg[s] else 1) if i in positions[s] else 0 for i in range(d)) for s in range(2))
    masks=tuple(sum(1<<i for i in p) for p in positions)
    wanted=tuple(x+y for x,y in zip(pacf(rows[0]),pacf(rows[1])))
    p=prepare_native_system(masks,d,wanted,negative,force_side=0)
    assert p['consistent'] and p['selected']==0
    q=int(p['meta'][4]);assert q>=11
    pattern=sum(int(rows[0][i]==-1)<<j for j,i in enumerate(sorted(positions[0])))
    assert all(bin(pattern&int(p['checks'][j])).count('1')%2==p['rhs'][j] for j in range(p['meta'][6]))
    base=list(map(int,p['base']))
    for j in range(p['meta'][5]):
        if pattern>>j&1:
            for s in range(2):base[s]^=int(p['patterns'][2*j+s])
    space=[tuple(map(int,p['toggles'][2*j:2*j+2])) for j in range(q)]
    pop=lambda x:bin(x).count('1')
    assert all(((pop(space[i][1])*pop(space[j][1]))^pop(space[i][1]&space[j][1]))%2==0
               for i in range(q) for j in range(i+1,q))
    linear=[((pop(base[1]^t[1])>>1)^(pop(base[1])>>1))&1 for t in space]
    rhs=((pop(base[1])>>1)^(negative[1]>>1))&1;pivot=linear.index(1)
    new_base=tuple(base[s]^(space[pivot][s] if rhs else 0) for s in range(2))
    remaining=[i for i in range(q) if i!=pivot]
    new_space=[tuple(space[i][s]^(space[pivot][s] if linear[i] else 0) for s in range(2)) for i in remaining]
    lib=_library();u=ctypes.c_uint64;model=lib.pqcp_quadratic_bit_tail_models
    model.argtypes=lib.pqcp_quadratic_hybrid_signs.argtypes[:-2]+[
        ctypes.POINTER(ctypes.c_uint32),ctypes.POINTER(u)];model.restype=ctypes.c_int
    def all_models(basis_base,basis):
        output=(ctypes.c_uint32*65536)();count=u();dim=len(basis)
        assert model(p['mask_array'],(u*2)(*basis_base),(u*(2*dim))(*(x for t in basis for x in t)),
            dim,d,p['wanted'],p['negative'],6,65536,output,ctypes.byref(count))==0
        assert count.value<=65536
        return set(map(int,output[:count.value]))
    original=all_models(base,space);reduced=all_models(new_base,new_space)
    mapped=set()
    for f in reduced:
        old=rhs<<pivot
        for j,i in enumerate(remaining):
            if f>>j&1:old^=(1<<i)^(linear[i]<<pivot)
        mapped.add(old)
    assert original==mapped and original
    exact_profiles=set()
    for f in original:
        signs=list(base)
        for j,t in enumerate(space):
            if f>>j&1:
                for s in range(2):signs[s]^=t[s]
        words=tuple(tuple((-1 if signs[s]>>i&1 else 1) if masks[s]>>i&1 else 0 for i in range(d)) for s in range(2))
        exact_profiles.add(tuple(x+y for x,y in zip(pacf(words[0]),pacf(words[1]))))
    # Alter a target by eight: all original modular models stay identical.
    # Pick a profile absent from the complete original modular model set.
    fake=None
    unsigned=tuple(tuple(int(m>>i&1) for i in range(d)) for m in masks)
    limits=tuple(x+y for x,y in zip(pacf(unsigned[0]),pacf(unsigned[1])))
    for shift in range(1,d//2+1):
        for delta in (-8,8):
            proposed=list(wanted);proposed[shift]+=delta;proposed[d-shift]+=delta
            if abs(proposed[shift])<=limits[shift] and tuple(proposed) not in exact_profiles:
                fake=tuple(proposed);break
        if fake is not None:break
    assert fake is not None
    output=(u*2)();stats=(u*7)()
    assert lib.pqcp_quadratic_count_hybrid_signs(p['mask_array'],(u*2)(*base),p['toggles'],q,d,
        (ctypes.c_int*d)(*fake),p['negative'],6,1<<q,output,stats)==0
    assert stats[0]==1<<q
    assert stats[4]==1<<(q-1-6)  # one exact dimension removed; keep all 64 tails
