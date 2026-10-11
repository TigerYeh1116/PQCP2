"""Odd sidelobes forced exactly by an impulse difference and arbitrary skew O.

g has +1 at +/-1; c is complementary and has sum 9. Sum-one skew O gives
O_i+O_-i=2 delta_0, so odd full PACFs are exactly +4 at 41/45. Mod-4 even
equations determine every reflected sign-pair XOR of c; exact cardinality
leaves only 122880 c rows. This is an explicit restricted construction.
"""
from itertools import combinations


def impulse_pair_certificate():
    d=43;all_bits=(1<<d)-1;holes=(1<<1)|(1<<42);active=all_bits^holes
    indices=tuple(range(2,22));equations=[]
    for u in range(1,22):
        coeff=sum((((active>>((p+u)%d))^(active>>((p-u)%d)))&1)<<i for i,p in enumerate(indices))
        rotated=((active<<u)|(active>>(d-u)))&all_bits
        baseline=bin(active&rotated).count('1');desired=0 if u==2 else 1
        if (desired-baseline)%2:raise RuntimeError('C mod-4 coefficient lattice failed')
        rhs=((desired-baseline)//2)%2
        equations.append(coeff|(rhs<<20))
    reduced=list(equations);rank=0;pivots=[]
    for column in range(20):
        pivot=next((i for i in range(rank,len(reduced)) if (reduced[i]>>column)&1),None)
        if pivot is None:continue
        reduced[rank],reduced[pivot]=reduced[pivot],reduced[rank]
        for i in range(len(reduced)):
            if i!=rank and (reduced[i]>>column)&1:reduced[i]^=reduced[rank]
        pivots.append(column);rank+=1
    if rank!=20 or any(row==(1<<20) for row in reduced):raise RuntimeError('unique reflected-pair XOR pattern not certified')
    solution=sum(((reduced[i]>>20)&1)<<column for i,column in enumerate(pivots))
    if any((bin((row&((1<<20)-1))&solution).count('1')%2)!=((row>>20)&1) for row in equations):
        raise RuntimeError('C pair XOR solution failed all original equations')
    opposite=tuple(p for i,p in enumerate(indices) if solution>>i&1)
    equal=tuple(p for i,p in enumerate(indices) if not solution>>i&1)
    return dict(active=active,holes=holes,rank=rank,equations=tuple(equations),pair_xor_mask=solution,
                opposite_pairs=opposite,equal_pairs=equal,negative_count=16,
                scope='g=delta_1+delta_-1, arbitrary sum-one skew O, c support complementary and sum 9.')


def iter_impulse_c_masks(certificate):
    opposite=certificate['opposite_pairs'];equal=certificate['equal_pairs'];negative=certificate['negative_count']
    for origin in (0,1):
        residual=negative-len(opposite)-origin
        if residual%2 or not 0<=residual//2<=len(equal):continue
        for chosen in combinations(equal,residual//2):
            base=origin|sum((1<<p)|(1<<(43-p)) for p in chosen)
            for orientations in range(1<<len(opposite)):
                mask=base
                for i,p in enumerate(opposite):mask|=1<<(p if orientations>>i&1 else 43-p)
                yield mask


def required_skew_intersections(negative,certificate):
    d=43;all_bits=(1<<d)-1;holes=certificate['holes']
    if (type(negative) is not int or not 0<=negative<1<<d or negative&holes
            or bin(negative).count('1')!=16):raise ValueError('literal complementary C negative support required')
    ea=negative;eb=negative|holes
    values=[]
    for u in range(1,22):
        ra=((ea<<u)|(ea>>(d-u)))&all_bits;rb=((eb<<u)|(eb>>(d-u)))&all_bits
        doubled=33-bin(ea&ra).count('1')-bin(eb&rb).count('1')
        if doubled%2:raise RuntimeError('C row failed necessary binary-O mod-4 lattice')
        values.append(doubled//2)
    return (21,)+tuple(values)+tuple(values[::-1])
