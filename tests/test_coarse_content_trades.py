from itertools import combinations,product

from solver.coarse_content_trades import exact_coarse_content_trades,replace_two_delta
from solver.structured_energy import integer_periodic_autocorrelation_profile as pacf,fold_profile,target_profile
from solver.target_profiles import canonical_target_content_profiles
from solver.torch_compressed_search import search_paired_contents


def test_two_site_delta_matches_every_small_trit_replacement():
    for row in product((-1,0,1),repeat=3):
        base=pacf(row)
        for p,q in combinations(range(3),2):
            for vp,vq in product((-1,0,1),repeat=2):
                changed=tuple(vp if i==p else vq if i==q else v for i,v in enumerate(row))
                assert replace_two_delta(row,p,q,vp,vq)==tuple(x-y for x,y in zip(pacf(changed)[1:2],base[1:2]))


def test_content_trades_match_literal_small_pair_space():
    rows=tuple(product((-1,0,1),repeat=3))
    for target in canonical_target_content_profiles(6):
        wanted=tuple(v//4 for v in fold_profile(target_profile(6,target.k,target.eta),2))
        allowed=search_paired_contents(target)
        feasible=[(a,b) for a,b in product(rows,repeat=2)
            if tuple((w.count(-1),w.count(0),w.count(1)) for w in (a,b)) in allowed
            and tuple(x+y for x,y in zip(pacf(a),pacf(b)))==wanted]
        for a,b in feasible:
            expected=set()
            for x,y in feasible:
                old=(a,b);new=(x,y)
                if any(sum(v!=w for v,w in zip(before,after))!=2 for before,after in zip(old,new)):continue
                changes=tuple(after.count(-1)-before.count(-1) for before,after in zip(old,new))
                if changes not in ((1,-1),(-1,1)):continue
                # Only zeros->+/- and +/- -> zeros, not other replacements.
                if all(all(v==w or (v==0 if change==1 else w==0) for v,w in zip(before,after))
                       for before,after,change in zip(old,new,changes)):expected.add((x,y))
            out=exact_coarse_content_trades(a,b,target)
            assert out['complete'] and set(out['words'])==expected
