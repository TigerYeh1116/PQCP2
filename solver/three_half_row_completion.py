"""Vary two known halves within a total flip budget; complete a third.

All 12 roles (one free, an unordered pair of nontrivially changed known
halves) are included. Only the fourth half stays fixed. This expands beyond
the two-half search without modifying the exact fixed-three completion.
"""
from itertools import combinations,product
from math import comb,prod

from .correlation import normalize_binary_sequence
from .half_row_completion import _target_index
from .two_half_row_completion import iter_half_flip_masks
from .verifier import verify_pqcp


def three_half_triple_count(d,radius):
    return 12*sum(comb(d,a)*comb(d,b) for a in range(1,min(d,radius-1)+1)
                  for b in range(1,min(d,radius-a)+1))


def three_half_row_fibers(a,b,*,radius=2):
    """All literal triples with two nonempty flip masks, total size <=r.

    The free half can change every bit and count. The remaining known half
    is held literally fixed. No restriction to adjacent flips, swaps or a
    selected target. The declared region is finite, not a global search.
    """
    a,b=tuple(normalize_binary_sequence(a)),tuple(normalize_binary_sequence(b));length=len(a);d=length//2
    if (len(b)!=length or length!=2*d or d not in (3,5,11,43)
            or type(radius) is not int or not 2<=radius<=min(4,2*d)):
        raise ValueError('supported equal twice-prime binary rows and total flip radius 2..4 required')
    if verify_pqcp(a,b).is_valid:raise ValueError('three-half centers must be non-solutions')
    halves=(a[::2],a[1::2],b[::2],b[1::2]);masks=tuple(sum(v<<i for i,v in enumerate(row)) for row in halves)
    weights=tuple(map(sum,halves));ordinal=0
    flip_groups={r:tuple(m for m in iter_half_flip_masks(d,r) if bin(m).count('1')==r)
                 for r in range(1,min(d,radius-1)+1)}
    for free in range(4):
        target_index=_target_index(length,free)
        for varied in combinations(tuple(i for i in range(4) if i!=free),2):
            x,y=varied
            for a_count in flip_groups:
                for b_count in flip_groups:
                    if a_count+b_count>radius:continue
                    for x_flip in flip_groups[a_count]:
                        x_mask=masks[x]^x_flip;x_weight=bin(x_mask).count('1')
                        for y_flip in flip_groups[b_count]:
                            ordinal+=1;y_mask=masks[y]^y_flip;counts=list(weights)
                            counts[x]=x_weight;counts[y]=bin(y_mask).count('1')
                            job=dict(varied=varied,free=free,flips=(x_flip,y_flip),ordinal=ordinal)
                            if tuple(v for i,v in enumerate(counts) if i!=free) not in target_index:
                                yield dict(job,kind='CONTENT_EXCLUDED');continue
                            trial=list(halves);trial[free]=None
                            trial[x]=tuple((x_mask>>i)&1 for i in range(d))
                            trial[y]=tuple((y_mask>>i)&1 for i in range(d))
                            fixed_masks=tuple(x_mask if i==x else y_mask if i==y else masks[i]
                                              for i in range(4) if i!=free)
                            # Two nonzero masks uniquely specify the changed
                            # known-row pair. XOR is injective in each role.
                            yield dict(job,kind='ELIGIBLE',halves=tuple(trial),fixed_masks=fixed_masks)
    if ordinal!=three_half_triple_count(d,radius):raise RuntimeError('three-half region enumeration incomplete')


def four_half_triple_count(d,radius):
    return 4*sum(prod(comb(d,c) for c in counts) for counts in product(range(1,min(d,radius-2)+1),repeat=3)
                 if sum(counts)<=radius)


def four_half_row_fibers(a,b,*,radius=3):
    """Three known halves change nontrivially; fourth is entirely free.

    No known row is held fixed. All four free roles are enumerated; the
    three bounded masks share a total flip budget and arbitrary positions.
    """
    a,b=tuple(normalize_binary_sequence(a)),tuple(normalize_binary_sequence(b));length=len(a);d=length//2
    if (len(b)!=length or length!=2*d or d not in (3,5,11,43)
            or type(radius) is not int or not 3<=radius<=4):
        raise ValueError('supported equal twice-prime binary rows and total flip radius 3..4 required')
    if verify_pqcp(a,b).is_valid:raise ValueError('four-half centers must be non-solutions')
    halves=(a[::2],a[1::2],b[::2],b[1::2]);masks=tuple(sum(v<<i for i,v in enumerate(row)) for row in halves)
    ordinal=0
    flip_groups={r:tuple(m for m in iter_half_flip_masks(d,r) if bin(m).count('1')==r)
                 for r in range(1,min(d,radius-2)+1)}
    for free in range(4):
        varied=tuple(i for i in range(4) if i!=free);target_index=_target_index(length,free)
        for counts in product(flip_groups,repeat=3):
            if sum(counts)>radius:continue
            for flips in product(*(flip_groups[c] for c in counts)):
                ordinal+=1;changed=tuple(masks[i]^f for i,f in zip(varied,flips))
                job=dict(varied=varied,free=free,flips=flips,ordinal=ordinal)
                if tuple(bin(m).count('1') for m in changed) not in target_index:
                    yield dict(job,kind='CONTENT_EXCLUDED');continue
                trial=[None]*4
                for i,mask in zip(varied,changed):trial[i]=tuple((mask>>u)&1 for u in range(d))
                yield dict(job,kind='ELIGIBLE',halves=tuple(trial),fixed_masks=changed)
    if ordinal!=four_half_triple_count(d,radius):raise RuntimeError('four-half region enumeration incomplete')
