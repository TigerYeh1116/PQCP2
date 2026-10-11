import random

import pytest

from solver.alternation_orbits import alternate_canonical_profile
from solver.pqcp_support_solver import complete_pqcp_supports
from solver.support_orbit_cache import FULL_SUPPORT_ORBITS, SupportOrbitCache
from solver.target_profiles import canonical_target_content_profiles, TargetContentProfile
from solver.ternary_support_search import support_mask
from solver.torch_compressed_search import search_paired_contents
from solver.torch_joint_search import joint_binary_words, binary_joint_rows
from solver.correlation import full_correlation_profile


def test_full_literal_modulation_transforms_every_c_g_content_and_cache_key():
    rng=random.Random(56)
    cache=SupportOrbitCache()
    for target in canonical_target_content_profiles(86):
        for contents in search_paired_contents(target):
            c,g=[],[]
            for (n,z,p),(e,o) in zip(contents,((target.a_even_ones,target.a_odd_ones),(target.b_even_ones,target.b_odd_ones))):
                positions=rng.sample(range(43),n+p)
                negative=set(rng.sample(positions,n))
                cs=tuple(-1 if i in negative else 1 if i in positions else 0 for i in range(43))
                inactive=[i for i in range(43) if i not in positions]
                ng=set(rng.sample(inactive,e-n))
                gs=tuple(0 if i in positions else -1 if i in ng else 1 for i in range(43))
                c.append(cs);g.append(gs)
            words=joint_binary_words(c,g,target)
            changed=tuple(tuple(v^(i%2) for i,v in enumerate(w)) for w in words)
            if target.k%2==0:changed=changed[::-1]
            partner=alternate_canonical_profile(target)
            cc,gg=binary_joint_rows(changed,partner)
            assert cc==tuple(g[::-1] if target.k%2==0 else g)
            assert gg==tuple(c[::-1] if target.k%2==0 else c)
            assert tuple(full_correlation_profile(*changed))==tuple(
                (-1 if u%2 else 1)*v for u,v in enumerate(full_correlation_profile(*words)))
            original_key=cache.key(tuple(support_mask(w) for w in c),target,tuple(w.count(-1) for w in c),modulation=True)
            changed_key=cache.key(tuple(support_mask(w) for w in cc),partner,tuple(w.count(-1) for w in cc),modulation=True)
            assert original_key==changed_key


def test_completed_odd_unsat_can_prune_the_opposite_sign_but_unknown_cannot():
    target=TargetContentProfile(86,1,-1,16,22,18,21)
    reference=(tuple([-1]*2+[0]*34+[1]*7),tuple([-1]*15+[0]*9+[1]*19))
    masks=(2216287141996,4362344590963)
    partner=alternate_canonical_profile(target)
    changed_masks=tuple(((1<<43)-1)^m for m in masks)
    changed=(tuple([-1]*14+[0]*9+[1]*20),tuple([-1]*3+[0]*34+[1]*6))
    FULL_SUPPORT_ORBITS.clear()
    partial=complete_pqcp_supports(*reference,target,masks,max_assignments=1,
                                 orbit_pruning=True,modulation_orbit_pruning=True)
    assert partial['status']=='UNKNOWN' and not FULL_SUPPORT_ORBITS.entries
    original=complete_pqcp_supports(*reference,target,masks,orbit_pruning=True,modulation_orbit_pruning=True)
    reused=complete_pqcp_supports(*changed,partner,changed_masks,max_assignments=1,
                                 orbit_pruning=True,modulation_orbit_pruning=True)
    rollback=complete_pqcp_supports(*changed,partner,changed_masks)
    assert original['status']==reused['status']==rollback['status']=='UNSAT_BINARY_SUPPORT'
    assert reused['attempts']=={} and reused['orbit_certificate']['source_target']==vars(target)
    assert reused['orbit_certificate']['source_negative_counts']==(2,15)
    assert rollback['attempts']
    FULL_SUPPORT_ORBITS.clear()


def test_modulation_cannot_be_enabled_without_proven_orbit_storage():
    target=TargetContentProfile(86,1,-1,16,22,18,21)
    reference=(tuple([-1]*2+[0]*34+[1]*7),tuple([-1]*15+[0]*9+[1]*19))
    with pytest.raises(ValueError,match='modulation orbit pruning'):
        complete_pqcp_supports(*reference,target,(2216287141996,4362344590963),
                              modulation_orbit_pruning=True)
