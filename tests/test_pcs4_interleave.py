from itertools import product

from solver.interleaving import interleave_bits
from solver.pcs4_interleave import (pcs4_ingredient_key,search_pcs4_interleave,
                                    mask_signs,reflected_mask)
from solver.solution_equivalence import pqcp_equivalence_key
from solver.verifier import verify_pqcp


def rotate(mask,r,d):return ((mask<<r)|(mask>>(d-r)))&((1<<d)-1) if r else mask


def test_ingredient_key_matches_entire_literal_small_group_orbit():
    for quad in ((0,3,4,2),(1,3,5,6),(7,0,2,5)):
        orbit=[]
        for q in (1,2):
            transformed=tuple(sum(((m>>((q*i)%3))&1)<<i for i in range(3)) for m in quad)
            for flags in product((False,True),repeat=4):
                rows=tuple(reflected_mask(m,3) if r else m for m,r in zip(transformed,flags))
                for rotations in product(range(3),repeat=4):
                    a,b,c,d=tuple(rotate(m,r,3) for m,r in zip(rows,rotations))
                    orbit.extend(((a,b,c,d),(a,b,d,c)))
        assert pcs4_ingredient_key(quad,3)==min(orbit)
        assert all(pcs4_ingredient_key(image,3)==min(orbit) for image in orbit[::23])


def test_relative_odd_reflections_cover_all_four_row_reflections_and_phases():
    quad=(0,3,4,2);actual=set()
    for assignment in (quad,quad[:2]+quad[2:][::-1]):
        for flags in product((False,True),repeat=4):
            rows=tuple(mask_signs(reflected_mask(m,3) if r else m,3) for m,r in zip(assignment,flags))
            for da,db in product(range(3),repeat=2):
                words=(interleave_bits(rows[0],rows[2],da),interleave_bits(rows[1],rows[3],db,-1))
                if verify_pqcp(*words).is_valid:actual.add(pqcp_equivalence_key(*words))
    out=search_pcs4_interleave(quad,3)
    assert actual and out['complete']
    assert {pqcp_equivalence_key(*hit['words']) for hit in out['witnesses']}==actual


def test_published_l43_ingredient_keeps_non_solution_seeds_and_all_images():
    quad=(1687788056856,3946068457116,3493597975730,1089223681038)
    out=search_pcs4_interleave(quad)
    assert out['complete'] and out['seeds'] and out['relative_images']>=2
    assert min(r['target_energy'] for r in out['seeds'])==8
    assert all(not verify_pqcp(*r['words']).is_valid for r in out['seeds'])
