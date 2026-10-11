"""Exact one-shift projections of fixed-support, fixed-cardinality signs.

N_U(u) = I_S(u) - 2 D_M(u), where D is the edge cut of negative
vertices M in the support graph with edges {j,j+u}. At prime length
the graph is a cycle restricted to S: disjoint paths, or the whole cycle.
Dynamic programming retains ALL attainable cuts, not just their bounds.
Different shifts still share signs; passing projections is not a witness.
"""
from functools import lru_cache


def _popcount(value):return bin(value).count("1")


def _validate(mask,d,negative):
    if (d not in (3,5,11,43) or type(mask) is not int or not 0<=mask<1<<d
            or type(negative) is not int or not 0<=negative<=_popcount(mask)):
        raise ValueError('supported prime, literal support and exact negative count required')


@lru_cache(maxsize=43)
def _path_cuts(length):
    # dp[k][end] is a bitset of transition counts on a path.
    dp=[[0,0] for _ in range(length+1)];dp[0][0]=1;dp[1][1]=1
    for size in range(1,length):
        nxt=[[0,0] for _ in range(length+1)]
        for k in range(size+1):
            left,right=dp[k]
            nxt[k][0]|=left|(right<<1)
            nxt[k+1][1]|=right|(left<<1)
        dp=nxt
    return tuple(left|right for left,right in dp)


def _sumsets(left,right):
    """Bitset Minkowski sum of finite nonnegative integer sets."""
    if _popcount(left)>_popcount(right):left,right=right,left
    out=0
    while left:
        bit=left&-left;out|=right<<(bit.bit_length()-1);left^=bit
    return out


@lru_cache(maxsize=8192)
def _component_cuts(lengths,negative):
    dp=[1]+[0]*negative;used=0
    for length in lengths:
        cuts=_path_cuts(length);nxt=[0]*(negative+1)
        for before in range(min(used,negative)+1):
            if not dp[before]:continue
            for add in range(min(length,negative-before)+1):
                nxt[before+add]|=_sumsets(dp[before],cuts[add])
        dp=nxt;used+=length
    return dp[negative]


def _literal_cycle_cuts(mask,d,negative,shift):
    """Independent vertex DP, also handles the closing edge of full support."""
    vertices=tuple(bool(mask&(1<<(j*shift%d))) for j in range(d));out=0
    for first in range(2 if vertices[0] else 1):
        if first>negative:continue
        dp=[[0,0] for _ in range(negative+1)];dp[first][first]=1
        for at in range(1,d):
            nxt=[[0,0] for _ in range(negative+1)]
            for count in range(negative+1):
                for previous in range(2):
                    bits=dp[count][previous]
                    if not bits:continue
                    for value in range(2 if vertices[at] else 1):
                        if count+value>negative:continue
                        cut=int(vertices[at-1] and vertices[at] and previous!=value)
                        nxt[count+value][value]|=bits<<cut
            dp=nxt
        for last in range(2):
            out|=dp[negative][last]<<int(vertices[-1] and vertices[0] and last!=first)
    return out


def attainable_cuts(mask,d,negative,shift):
    _validate(mask,d,negative)
    if type(shift) is not int or not 1<=shift<d:raise ValueError('nonzero literal shift required')
    negative=min(negative,_popcount(mask)-negative) # Whole-row negation preserves every cut.
    if mask==(1<<d)-1:return _literal_cycle_cuts(mask,d,negative,shift)
    ordered=tuple(bool(mask&(1<<(j*shift%d))) for j in range(d))
    missing=ordered.index(False);lengths=[];run=0
    for at in range(1,d+1):
        if ordered[(missing+at)%d]:run+=1
        elif run:lengths.append(run);run=0
    return _component_cuts(tuple(sorted(lengths)),negative)


def _intersection(mask,d,shift):
    return _popcount(mask&(((mask<<shift)|(mask>>(d-shift)))&((1<<d)-1)))


def verify_cut_certificate(proof):
    """Recompute exclusions using a separate literal cycle-vertex DP."""
    masks=tuple(proof['masks']);negative=tuple(proof['negative']);d=proof['d'];shift=proof['shift']
    if len(masks) not in (1,2) or len(masks)!=len(negative) or type(shift) is not int or not 1<=shift<=d//2:
        return False
    bits=1;intersection=0
    for mask,count in zip(masks,negative):
        _validate(mask,d,count)
        bits=_sumsets(bits,_literal_cycle_cuts(mask,d,min(count,_popcount(mask)-count),shift))
        intersection+=_intersection(mask,d,shift)
    difference=intersection-proof['wanted']
    return (bits==proof['attainable_cut_sums'] and intersection==proof['unsigned_intersection_sum']
            and (difference<0 or difference%2 or not bits&(1<<(difference//2))))


def support_cut_obstruction(masks,d,wanted,negative):
    """Necessary condition for ONE norm or a SUM of TWO unknown norms.

    Each summand is projected independently, then all possible cut sums
    are combined. This does not apply a single-norm test to a norm sum.
    Every exclusion is audited by the independent literal vertex DP.
    """
    masks=tuple(masks);negative=tuple(negative);wanted=tuple(wanted)
    if len(masks) not in (1,2) or len(masks)!=len(negative):raise ValueError('one or two signed support rows required')
    for mask,count in zip(masks,negative):_validate(mask,d,count)
    if (len(wanted)!=d or any(type(v) is not int for v in wanted)
            or wanted[0]!=sum(_popcount(m) for m in masks) or any(wanted[u]!=wanted[-u] for u in range(1,d))):
        raise ValueError('symmetric integer norm target with exact total weight required')
    for shift in range(1,d//2+1):
        bits=1;intersection=0
        for mask,count in zip(masks,negative):
            bits=_sumsets(bits,attainable_cuts(mask,d,count,shift));intersection+=_intersection(mask,d,shift)
        difference=intersection-wanted[shift]
        if difference<0 or difference%2 or not bits&(1<<(difference//2)):
            proof=dict(d=d,masks=masks,negative=negative,shift=shift,wanted=wanted[shift],
                unsigned_intersection_sum=intersection,attainable_cut_sums=bits,
                reason='EXACT_CARDINALITY_SUPPORT_CUT_PROJECTION',
                scope='All cut counts at this shift with these literal supports and negative counts. Passing other shifts is not inferred.')
            if not verify_cut_certificate(proof):raise RuntimeError('support cut projection failed independent literal cycle audit')
            return proof
    return None
