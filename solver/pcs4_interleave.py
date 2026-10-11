"""Full-class-preserving symmetries and exact interleaving of PCS4 ingredients.

Independent reflections of E are absorbed by full-row reflection together
with O reflection and relative phase. Retain BOTH O assignments and FOUR
relative O reflections. No homometric rows are merged.
"""
from functools import lru_cache
from itertools import product

from .interleaving import interleave_bits,sparse_odd_matches,best_sparse_odd_candidate,periodic_sign_profile
from .sds_norm_completion import native_odd_cross_vectors
from .structured_energy import target_profile
from .support_orbit_cache import row_dihedral_mask
from .verifier import verify_pqcp


def mask_signs(mask,d):return tuple(1-2*((mask>>i)&1) for i in range(d))


def reflected_mask(mask,d):return sum(((mask>>i)&1)<<((-i)%d) for i in range(d))


@lru_cache(maxsize=4096)
def _unit_dihedral_keys(mask,d):
    return tuple(row_dihedral_mask(sum(((mask>>((q*i)%d))&1)<<i for i in range(d)),d)
                 for q in range(1,d))


def pcs4_ingredient_key(quad,d=43):
    """Common units, independent dihedral rows and O exchange, with label E fixed."""
    quad=tuple(quad)
    if d not in (3,5,11,43) or len(quad)!=4 or any(type(m) is not int or not 0<=m<1<<d for m in quad):
        raise ValueError('four literal masks and supported prime length required')
    images=tuple(_unit_dihedral_keys(m,d) for m in quad)
    return min((a,b,*sorted((c,e))) for a,b,c,e in zip(*images))


def relative_odd_images(quad,d):
    ea,eb,oa,ob=quad;seen=set()
    for a,b in ((oa,ob),(ob,oa)):
        for ra,rb in product((False,True),repeat=2):
            pair=(reflected_mask(a,d) if ra else a,reflected_mask(b,d) if rb else b)
            if pair in seen:continue
            seen.add(pair);yield (ea,eb,*pair)


@lru_cache(maxsize=4096)
def _vectors(e,o,d):return native_odd_cross_vectors(e,o,d)


def search_pcs4_interleave(quad,d=43):
    """All full witnesses in reflected/assigned/phase images of this ingredient.

    One best seed per relative-reflection image is selected heuristically;
    it never replaces the exhaustive exact sparse-target join.
    """
    quad=tuple(quad)
    if len(quad)!=4 or d not in (3,5,11,43) or any(type(m) is not int or not 0<=m<1<<d for m in quad):
        raise ValueError('supported PCS4 ingredient masks required')
    rows=tuple(mask_signs(m,d) for m in quad)
    if tuple(map(sum,zip(*(periodic_sign_profile(w) for w in rows))))!=(4*d,)+(0,)*(d-1):
        raise ValueError('independently verified four-row summed PACF zero required')
    witnesses=[];seeds=[];seen=set();queries=images=0
    for ea,eb,oa,ob in relative_odd_images(quad,d):
        images+=1
        left,right=_vectors(ea,oa,d),_vectors(eb,ob,d)
        queries+=(d//2)*len(left)
        e_rows=mask_signs(ea,d),mask_signs(eb,d)
        o_rows=mask_signs(oa,d),mask_signs(ob,d)
        for k,da,db in sparse_odd_matches(left,right):
            words=(interleave_bits(e_rows[0],o_rows[0],da),interleave_bits(e_rows[1],o_rows[1],db,-1))
            check=verify_pqcp(*words)
            if not check.is_valid or tuple(check.profile)!=target_profile(2*d,k,1):
                raise RuntimeError('PCS4 interleave failed independent original-length target audit')
            if words in seen:continue
            seen.add(words);witnesses.append(dict(words=words,k=k,eta=1,odd_masks=(oa,ob),offsets=(da,db)))
        best=best_sparse_odd_candidate(left,right)
        if best is not None and best['full_energy']>0:
            words=(interleave_bits(e_rows[0],o_rows[0],best['left_preimage']),
                   interleave_bits(e_rows[1],o_rows[1],best['right_preimage'],-1))
            check=verify_pqcp(*words);wanted=target_profile(2*d,best['k'],1)
            if check.profile[d] or sum(((x-y)//4)**2 for x,y in zip(check.profile[1:d+1],wanted[1:d+1]))!=best['full_energy']:
                raise RuntimeError('PCS4 near-seed failed independent full integer energy audit')
            seeds.append(dict(words=words,k=best['k'],sign=1,target_energy=best['full_energy'],
                              odd_masks=(oa,ob),profile=check.profile))
    return dict(complete=True,witnesses=witnesses,seeds=seeds,relative_images=images,odd_lookups=queries,
                scope='All odd assignments, relative reflections and phases of this PCS4 ingredient. '
                      'One near-seed per image; not all PCS4s or all near-seed preimages.')
