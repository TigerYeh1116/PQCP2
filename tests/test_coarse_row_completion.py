from itertools import product

from solver.coarse_row_completion import complete_fixed_coarse_row_target
from solver.full_row_completion import crt_trit_rows
from solver.verifier import verify_pqcp


def test_fixed_coarse_row_all_remaining_binary_bits_against_entire_l6_space():
    binary=tuple(product((0,1),repeat=6));expected={}
    for a,b in product(binary,repeat=2):
        check=verify_pqcp(a,b)
        if check.is_valid:
            key=crt_trit_rows(a)[0],check.nonzero_shifts[0],check.profile[check.nonzero_shifts[0]]//4
            expected.setdefault(key,set()).add((a,b))
    positive=negative=0
    for c in product((-1,0,1),repeat=3):
        for k,eta in product((1,2),(-1,1)):
            outputs=[]
            for bounds in (False,True):
                out=complete_fixed_coarse_row_target(c,k=k,eta=eta,support_bounds=bounds)
                assert bool(out['words'])==bool(expected.get((c,k,eta),()))
                if out['words']:
                    assert out['words'] in expected[c,k,eta] and crt_trit_rows(out['words'][0])[0]==c
                    assert verify_pqcp(*out['words']).is_valid;positive+=1
                else:assert out['complete'] and out['status']=='UNSAT_FIXED_SIGNED_C_A';negative+=1
                outputs.append(out)
            assert (outputs[0]['status'],outputs[0]['words'],outputs[0]['complete'])==(outputs[1]['status'],outputs[1]['words'],outputs[1]['complete'])
            lifted=complete_fixed_coarse_row_target(c,k=k,eta=eta,support_bounds=True,two_adic=True,dihedral_support=True,cut_projection=True)
            assert bool(lifted['words'])==bool(expected.get((c,k,eta),()))
            if lifted['words']:assert lifted['words'] in expected[c,k,eta]
            else:assert lifted['complete']
    assert positive and negative


def test_cache_reuses_only_single_c_b_exhaustion_across_opposite_original_parities():
    import json
    from pathlib import Path
    from solver.coarse_row_completion import CompleteUnaryNormUnsatCache
    # Read-only independently audited non-solution input used by the actual
    # long run, not a saved PQCP or synthetic positive performance seed.
    from solver.full_row_completion import crt_trit_rows
    path=Path(__file__).resolve().parents[1]/'results/L86_latest_two_halves_radius2_top8_20261008/input_centers.json'
    c=crt_trit_rows(json.loads(path.read_text())['seeds'][0]['words'][0])[0]
    cache=CompleteUnaryNormUnsatCache()
    a=complete_fixed_coarse_row_target(c,k=1,eta=1,norm_unsat_cache=cache,two_adic=True,support_bounds=True)
    b=complete_fixed_coarse_row_target(c,k=42,eta=1,norm_unsat_cache=cache,two_adic=True,support_bounds=True)
    assert a['c_b_norm_unsat'] and b['c_b_norm_unsat'] and b['norm_unsat_cache_hit'] and cache.hits==1
    assert a['profile']['wanted_c_b']==b['profile']['wanted_c_b']
    assert a['profile']['wanted_g_pair']!=b['profile']['wanted_g_pair']


def test_g_only_unsat_never_enters_the_single_norm_cache():
    from solver.coarse_row_completion import CompleteUnaryNormUnsatCache
    # Archived exact coarse C_A; it has actual C_B norm partners while both
    # G lifts at opposite original parities are independently complete UNSAT.
    old=(0,-1,0,1,0,0,0,0,1,-1,0,1,1,-1,0,0,0,0,0,0,0,1,0,0,0,-1,0,0,1,1,1,0,1,0,-1,1,1,-1,0,-1,1,0,1)
    c=tuple(old[2*j%43] for j in range(43));cache=CompleteUnaryNormUnsatCache()
    for k in (1,42):
        out=complete_fixed_coarse_row_target(c,k=k,eta=1,norm_unsat_cache=cache,support_bounds=True)
        assert out['coarse_witness_supports']>0 and not out['c_b_norm_unsat'] and not out['norm_unsat_cache_hit']
    assert not cache._items and cache.hits==0
