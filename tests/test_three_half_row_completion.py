from itertools import product

import pytest

from solver.half_row_completion import complete_three_parity_rows
from solver.three_half_row_completion import three_half_row_fibers,three_half_triple_count
from solver.verifier import verify_pqcp


def halves(pair):return pair[0][::2],pair[0][1::2],pair[1][::2],pair[1][1::2]


def test_entire_declared_three_half_region_matches_literal_l6_oracle():
    small=tuple(product((0,1),repeat=6));pairs=tuple(product(small,repeat=2))
    valid=tuple(p for p in pairs if verify_pqcp(*p).is_valid);positive=0
    for center in (pairs[0],pairs[543],pairs[1445]):
        original=halves(center)
        for radius in (2,3,4):
            expected=set()
            for pair in valid:
                distances=tuple(sum(a!=b for a,b in zip(u,v)) for u,v in zip(original,halves(pair)))
                if any(distances[fixed]==0 and all(distances[i]>0 for i in range(4) if i not in (free,fixed))
                       and sum(distances[i] for i in range(4) if i not in (free,fixed))<=radius
                       for free in range(4) for fixed in range(4) if free!=fixed):expected.add(pair)
            jobs=list(three_half_row_fibers(*center,radius=radius))
            assert len(jobs)==three_half_triple_count(3,radius)
            keys=set();actual=set()
            for job in jobs:
                if job['kind']!='ELIGIBLE':continue
                key=job['free'],job['fixed_masks'];assert key not in keys;keys.add(key)
                out=complete_three_parity_rows(job['halves'],free_component=job['free'],
                                               finite_trace=True,joint_gram=True,batched_psd=True)
                assert out['complete'];actual.update(hit['words'] for hit in out['witnesses'])
            assert actual==expected;positive+=bool(expected)
    assert positive


def test_three_half_rejects_invalid_radius_or_saved_solution():
    for radius in (True,0,1,5):
        with pytest.raises(ValueError):list(three_half_row_fibers((0,)*6,(0,)*6,radius=radius))
    with pytest.raises(ValueError):list(three_half_row_fibers((0,0,0,0,0,1),(0,0,0,0,1,1)))


def test_four_half_complete_region_all_roles_against_literal_l6_oracle():
    from solver.three_half_row_completion import four_half_row_fibers,four_half_triple_count
    small=tuple(product((0,1),repeat=6));pairs=tuple(product(small,repeat=2))
    valid=tuple(p for p in pairs if verify_pqcp(*p).is_valid);positive=0
    for center in (pairs[0],pairs[543],pairs[1445]):
        for radius in (3,4):
            original=halves(center);expected=set()
            for pair in valid:
                distances=tuple(sum(x!=y for x,y in zip(a,b)) for a,b in zip(original,halves(pair)))
                if any(all(distances[i]>0 for i in range(4) if i!=free)
                       and sum(distances[i] for i in range(4) if i!=free)<=radius for free in range(4)):expected.add(pair)
            jobs=list(four_half_row_fibers(*center,radius=radius));actual=set();keys=set()
            assert len(jobs)==four_half_triple_count(3,radius)
            for job in jobs:
                if job['kind']!='ELIGIBLE':continue
                key=job['free'],job['fixed_masks'];assert key not in keys;keys.add(key)
                out=complete_three_parity_rows(job['halves'],free_component=job['free'],finite_trace=True,
                                               joint_gram=True,batched_psd=True)
                actual.update(hit['words'] for hit in out['witnesses'])
            assert actual==expected;positive+=bool(actual)
    assert positive
