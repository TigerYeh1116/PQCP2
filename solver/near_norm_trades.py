"""Complete content-constrained sign trades and exact integer profile joins.

Only odd d <= 47 is supported. Packing is the LINEAR integer map
Phi(v)=sum(v[u]*512**u), not digit-wise modular arithmetic. For a pair of
real sign profiles and a target bounded by 2d, their difference is bounded
by 4d < 511 in each coordinate. Hence integer equality is injective:
the highest nonzero coordinate outweighs every lower coordinate combined.
"""
from itertools import combinations

BASE_BITS=9


def profile(row):
    return tuple(sum(row[i]*row[(i+u)%len(row)] for i in range(len(row)))
                 for u in range(1,len(row)//2+1))


def packed(vector):
    return sum(value<<(BASE_BITS*i) for i,value in enumerate(vector))


def content_trade_groups(row,target_sum,max_flips):
    """Every literal row within the Hamming bound and requested exact sum."""
    row=tuple(row);d=len(row)
    if (not 3<=d<=47 or d%2!=1 or any(type(v) is not int or v not in (-1,1) for v in row)
            or type(target_sum) is not int or not -d<=target_sum<=d or (d-target_sum)%2
            or type(max_flips) is not int or not 0<=max_flips<=d):
        raise ValueError('bounded odd sign row, exact content and Hamming radius required')
    negative=tuple(i for i,v in enumerate(row) if v==-1)
    positive=tuple(i for i,v in enumerate(row) if v==1)
    original=sum(1<<i for i in negative);base=packed(profile(row));delta=(target_sum-sum(row))//2
    singles=tuple(packed(tuple(-2*row[i]*(row[(i+u)%d]+row[(i-u)%d])
                              for u in range(1,d//2+1))) for i in range(d))
    groups={};tested=0
    for flips in range(max_flips+1):
        if (flips+delta)%2:continue
        nminus,nplus=(flips+delta)//2,(flips-delta)//2
        if not 0<=nminus<=len(negative) or not 0<=nplus<=len(positive):continue
        for remove in combinations(negative,nminus):
            for add in combinations(positive,nplus):
                positions=remove+add;mask=original;code=base
                for i in positions:mask^=1<<i;code+=singles[i]
                for i,j in combinations(positions,2):
                    distance=min(abs(i-j),d-abs(i-j))
                    code+=(4*row[i]*row[j])<<(BASE_BITS*(distance-1))
                groups.setdefault(code,[]).append(dict(mask=mask,flips=tuple(sorted(positions))))
                tested+=1
    return dict(d=d,target_sum=target_sum,max_flips=max_flips,groups=groups,literal_variants=tested)


def exact_profile_pairs(left,right,target):
    """All preimages of the COMPLETE independent-shift target, no cap."""
    d=left['d'];target=tuple(target)
    if (right['d']!=d or len(target)!=d//2 or
            any(type(v) is not int or abs(v)>2*d for v in target)):
        raise ValueError('matching row lengths and physically bounded pair target required')
    wanted=packed(target)
    for code,models in left['groups'].items():
        others=right['groups'].get(wanted-code,())
        for a in models:
            for b in others:yield a,b


def signs(mask,d):
    return tuple(-1 if mask>>i&1 else 1 for i in range(d))


def zero_odd_offset_pairs(e0,o0,e1,o1,*,statistics=None):
    """All offsets with zero odd sum for interleave(e0,o0), interleave(e1,-o1).

    This fixes only the odd correlations. The caller must separately prove
    the even target, then verify the original binary pair.
    """
    from .interleaving import odd_cross_vectors
    if len({len(row) for row in (e0,o0,e1,o1)})!=1:
        raise ValueError('all four sign rows must have equal lengths')
    left,right=odd_cross_vectors(e0,o0),odd_cross_vectors(e1,o1)
    if statistics is not None:statistics['profile_lookups']=0
    for vector,offsets in left.items():
        if statistics is not None:statistics['profile_lookups']+=1
        for first in offsets:
            for second in right.get(vector,()):yield first,second
