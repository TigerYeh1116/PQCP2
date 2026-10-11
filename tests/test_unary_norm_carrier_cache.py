from itertools import product

from solver.coarse_row_completion import complete_fixed_coarse_row_target
from solver.unary_norm_carrier_cache import UnaryNormCarrierCache
from solver.verifier import verify_pqcp


def test_cached_complete_small_domains_match_all_original_targets(monkeypatch):
    import solver.coarse_row_completion as module
    real_g=module.run_native_system
    hits=0
    for d in (3,5):
        for c in product((-1,0,1),repeat=d):
            cache=UnaryNormCarrierCache()
            # A capped G stage must not discard a completed C_B domain.
            # Warm actual C_B carriers, then compare real full G outputs.
            monkeypatch.setattr(module,'run_native_system',lambda *args,**kwargs:dict(status='UNKNOWN',words=None,assignments=1))
            for eta in (-1,1):
                complete_fixed_coarse_row_target(c,k=1,eta=eta,support_bounds=True,two_adic=True,dihedral_support=True,norm_carrier_cache=cache)
            monkeypatch.setattr(module,'run_native_system',real_g)
            for k in range(1,d):
                for eta in (-1,1):
                    options=dict(k=k,eta=eta,support_bounds=True,two_adic=True,dihedral_support=True)
                    old=complete_fixed_coarse_row_target(c,**options)
                    new=complete_fixed_coarse_row_target(c,norm_carrier_cache=cache,**options)
                    assert (new['status'],new['complete'],new['words'])==(old['status'],old['complete'],old['words'])
                    if new['words']:assert verify_pqcp(*new['words']).is_valid
            hits+=cache.hits
    assert hits>0


def test_l43_g_only_exclusions_are_recomputed_at_opposite_original_parity(monkeypatch):
    import solver.coarse_row_completion as module
    old=(0,-1,0,1,0,0,0,0,1,-1,0,1,1,-1,0,0,0,0,0,0,0,1,0,0,0,-1,0,0,1,1,1,0,1,0,-1,1,1,-1,0,-1,1,0,1)
    c=tuple(old[2*j%43] for j in range(43));cache=UnaryNormCarrierCache();actual=module.run_native_system;targets=[]
    def observed(plan,budget):
        targets.append(tuple(plan['wanted']));return actual(plan,budget)
    monkeypatch.setattr(module,'run_native_system',observed)
    a=complete_fixed_coarse_row_target(c,k=1,eta=1,norm_carrier_cache=cache,support_bounds=True,dihedral_support=True,two_adic=True)
    assert a['complete'] and not a['words'] and a['coarse_witness_supports'] and not a['c_b_norm_unsat']
    targets.clear()
    b=complete_fixed_coarse_row_target(c,k=42,eta=1,norm_carrier_cache=cache,support_bounds=True,dihedral_support=True,two_adic=True)
    assert b['norm_carrier_cache_hit'] and b['reused_c_b_carriers'] and b['complete'] and not b['words']
    assert a['profile']['wanted_c_b']==b['profile']['wanted_c_b']
    assert a['profile']['wanted_g_pair']!=b['profile']['wanted_g_pair']
    assert targets and all(t==b['profile']['wanted_g_pair'] for t in targets)
    assert cache.slots<=cache.slot_budget


def test_unknown_supports_remain_available_and_can_later_produce_a_full_solution(monkeypatch):
    import solver.coarse_row_completion as module
    cache=UnaryNormCarrierCache();c=(0,1,0);actual=module.unary_sign_completion;seen=[]
    def capped(*args,**kwargs):
        seen.append(args[0]);return dict(status='UNKNOWN',words=None,assignments=1)
    monkeypatch.setattr(module,'unary_sign_completion',capped)
    for k in (1,2):
        out=complete_fixed_coarse_row_target(c,k=k,eta=1,norm_carrier_cache=cache)
        assert out['status']=='UNKNOWN_FIXED_SIGNED_C_A' and not out['complete'] and not out['c_b_norm_unsat']
    assert len(seen)==2 and cache.hits==1 and all(word is None for mask,word in next(iter(cache._items.values()))['candidates'])
    monkeypatch.setattr(module,'unary_sign_completion',actual)
    out=complete_fixed_coarse_row_target(c,k=2,eta=1,norm_carrier_cache=cache)
    assert out['status']=='SAT_FULL_PQCP' and verify_pqcp(*out['words']).is_valid


def test_first_full_sat_does_not_close_unvisited_supports():
    cache=UnaryNormCarrierCache()
    out=complete_fixed_coarse_row_target((0,1,0),k=2,eta=1,norm_carrier_cache=cache)
    assert out['words'] and not out['complete'] and not cache._items and cache.slots==0
