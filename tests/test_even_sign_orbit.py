"""Complete raw fibres, exact original trit norms, partial-budget rollback."""
from itertools import product
import json
from pathlib import Path
from random import Random

import pytest

from solver.odd_lift_native import difference_sign_problem
from solver.structured_energy import integer_periodic_autocorrelation_profile as pacf
from solver.target_profiles import TargetContentProfile
from solver.ternary_support_search import support_mask
from solver.ternary_system_native import prepare_native_system,run_native_system


@pytest.mark.parametrize('d',[3,5,7,9,11])
def test_all_small_original_sign_words_and_partial_budget(d):
    rng=Random(1000+d);used=False
    for case in range(12):
        occupied=set(rng.sample(range(2*d),d))
        masks=tuple(sum(1<<i for i in range(d) if s*d+i in occupied) for s in range(2))
        reference=tuple(tuple(rng.choice((-1,1)) if m>>i&1 else 0 for i in range(d)) for m in masks)
        wanted=tuple(x+y for x,y in zip(pacf(reference[0]),pacf(reference[1])))
        if case%2:
            limit=sum(pacf(tuple(int(m>>i&1) for i in range(d)))[1] for m in masks)
            alternatives=[v for v in (wanted[1]-4,wanted[1]+4) if abs(v)<=limit]
            if alternatives:wanted=(wanted[0],alternatives[0])+wanted[2:-1]+(alternatives[0],)
        negative=tuple(w.count(-1) for w in reference)
        plan=prepare_native_system(masks,d,wanted,negative,conditioning=False)
        if not plan['consistent']:continue
        positions=[(s,i) for s in range(2) for i in range(d) if masks[s]>>i&1]
        feasible=set()
        for values in product((-1,1),repeat=d):
            pair=[[0]*d for _ in range(2)]
            for (s,i),v in zip(positions,values):pair[s][i]=v
            pair=tuple(tuple(w) for w in pair)
            if tuple(w.count(-1) for w in pair)==negative and tuple(x+y for x,y in zip(pacf(pair[0]),pacf(pair[1])))==wanted:
                feasible.add(pair)
        complete=1<<plan['meta'][2]
        for tail in (1,3,6,8):
            old=run_native_system(plan,complete,quadratic=True,hybrid_tail=tail)
            new=run_native_system(plan,complete,quadratic=True,hybrid_tail=tail,even_sign_orbit=True)
            assert old['status']==new['status']==('SAT_COMPRESSED' if feasible else 'UNSAT_FIXED_SUPPORT')
            if feasible:assert new['words'] in feasible
            else:assert new['assignments']==old['assignments']==complete
            if new.get('even_sign_orbit_pruning'):
                used=True;assert new['orbit_reduced_free_variables']==plan['meta'][2]-1
            for cap in {1,max(1,complete-1)}:
                if cap>=complete:continue
                original=run_native_system(plan,cap,quadratic=True,hybrid_tail=tail)
                rollback=run_native_system(plan,cap,quadratic=True,hybrid_tail=tail,even_sign_orbit=True)
                assert (rollback['status'],rollback['words'],rollback['assignments'])==(original['status'],original['words'],original['assignments'])
                assert not rollback['even_sign_orbit_pruning']
    assert used


def test_actual_l43_c_witnesses_and_dead_g_domains_cover_sigma_four_and_six():
    root=Path(__file__).resolve().parents[1]
    sources=json.loads((root/'results/L86_independent_norm_phase_million_20261008/coarse_witnesses.json').read_text())['rows']
    seen=set()
    for source in sources:
        target=TargetContentProfile(**source['target'])
        if target.eta in seen:continue
        c=tuple(tuple(w) for w in source['result']['attempts']['coarse']['words']);masks=tuple(support_mask(w) for w in c)
        wanted=tuple(x+y for x,y in zip(pacf(c[0]),pacf(c[1])))
        mg,wg,ng=difference_sign_problem(masks,target)
        for kind,supports,profile,negative in (('C',masks,wanted,tuple(w.count(-1) for w in c)),('G',mg,wg,ng)):
            plan=prepare_native_system(supports,43,profile,negative,conditioning=False)
            old=run_native_system(plan,2**23)
            new=run_native_system(plan,2**23,even_sign_orbit=True)
            assert old['status']==new['status']==('SAT_COMPRESSED' if kind=='C' else 'UNSAT_FIXED_SUPPORT')
            assert new['even_sign_orbit_pruning']
            if kind=='C':
                assert tuple(w.count(-1) for w in new['words'])==negative
                assert tuple(x+y for x,y in zip(pacf(new['words'][0]),pacf(new['words'][1])))==profile
                assert tuple(support_mask(w) for w in new['words'])==supports
            else:
                assert new['assignments']==old['assignments']==1<<plan['meta'][2]
                assert new['prefix_branches']<=old['prefix_branches']
                assert new['prefix_branches']*(1<<new['orbit_tail_variables'])==1<<(plan['meta'][2]-1)
        seen.add(target.eta)
        if len(seen)==2:break
    assert seen=={-1,1}
