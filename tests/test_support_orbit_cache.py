from itertools import product

from solver.pqcp_support_solver import complete_pqcp_supports
from solver.structured_energy import integer_periodic_autocorrelation_profile
from solver.support_orbit_cache import FULL_SUPPORT_ORBITS,SupportOrbitCache,row_dihedral_mask
from solver.target_profiles import TargetContentProfile


def test_row_dihedral_orbits_preserve_every_literal_signed_profile_and_count():
    d=5
    for word in product((-1,0,1),repeat=d):
        mask=sum(int(v!=0)<<i for i,v in enumerate(word))
        key=row_dihedral_mask(mask,d)
        expected=integer_periodic_autocorrelation_profile(word)
        for reverse in (1,-1):
            for shift in range(d):
                changed=tuple(word[(reverse*i+shift)%d] for i in range(d))
                other=sum(int(v!=0)<<i for i,v in enumerate(changed))
                assert integer_periodic_autocorrelation_profile(changed)==expected
                assert changed.count(-1)==word.count(-1)
                assert row_dihedral_mask(other,d)==key
                complement=((1<<d)-1)^other
                original_complement=((1<<d)-1)^mask
                assert row_dihedral_mask(complement,d)==row_dihedral_mask(original_complement,d)


def test_full_l43_unsat_orbit_reuses_only_a_completed_proof_and_has_rollback():
    target=TargetContentProfile(86,1,-1,16,22,18,21)
    reference=(tuple([-1]*2+[0]*34+[1]*7),tuple([-1]*15+[0]*9+[1]*19))
    masks=(2216287141996,4362344590963)
    rotated=tuple(((m<<shift)|(m>>(43-shift)))&((1<<43)-1) for m,shift in zip(masks,(3,7)))
    FULL_SUPPORT_ORBITS.clear()
    partial=complete_pqcp_supports(*reference,target,masks,max_assignments=1,orbit_pruning=True)
    assert partial['status']=='UNKNOWN' and not FULL_SUPPORT_ORBITS.entries
    first=complete_pqcp_supports(*reference,target,masks,orbit_pruning=True)
    second=complete_pqcp_supports(*reference,target,rotated,orbit_pruning=True)
    rollback=complete_pqcp_supports(*reference,target,rotated,orbit_pruning=False)
    assert first['status']==second['status']==rollback['status']=='UNSAT_BINARY_SUPPORT'
    assert second['excluded_by']=='dihedral_orbit' and second['attempts']=={}
    assert second['orbit_certificate']['source_masks']==masks and rollback['attempts']
    FULL_SUPPORT_ORBITS.clear()


def test_orbit_cache_is_bounded_and_does_not_share_targets_or_counts():
    cache=SupportOrbitCache(2)
    target=TargetContentProfile(6,2,-1,1,2,1,1)
    first=cache.key((1,3),target,(0,1))
    assert first!=cache.key((1,3),target,(1,1))
    assert first!=cache.key((1,3),TargetContentProfile(6,2,1,1,2,1,1),(0,1))
    for i in range(3):cache.remember(('case',i),(1,3),'coarse')
    assert len(cache.entries)==2 and cache.get(('case',0)) is None
