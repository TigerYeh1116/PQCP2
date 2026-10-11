"""Exact mod-4 affine C fibers for g of three plus signs and one minus.

Restricted centered construction E_A=c+g,E_B=c-g, actual odd rows O,-O.
No skew or perfect requirement on O. L43 scalar sums c=9,g=2,O=1.
"""
from itertools import combinations


def normalized_g_rows():
    """All 11480 literal g with its unique minus at zero; no reflection quotient."""
    for plus in combinations(range(1,43),3):
        yield sum(1<<i for i in plus),1


def c_affine_certificate(plus,minus,d=43,negative_weight=15,*,g_weight=4):
    if (type(d) is not int or d<5 or not d%2 or any(type(v) is not int or not 0<=v<1<<d for v in (plus,minus))
            or g_weight not in (2,4) or plus&minus
            or bin(plus).count('1')!=(g_weight+2)//2 or bin(minus).count('1')!=(g_weight-2)//2):
        raise ValueError('disjoint literal sum-two masks of declared weight two or four required')
    active=((1<<d)-1)^(plus|minus)
    positions=tuple(i for i in range(d) if active>>i&1);m=len(positions)
    if type(negative_weight) is not int or not 0<=negative_weight<=m:
        raise ValueError('C negative weight outside complementary support')
    g=tuple(int(plus>>i&1)-int(minus>>i&1) for i in range(d))
    rows=[]
    for u in range(1,d//2+1):
        baseline=sum(int(active>>i&1)*int(active>>((i+u)%d)&1) for i in range(d))
        pg=sum(g[i]*g[(i+u)%d] for i in range(d))
        wanted=d-2-pg
        if (baseline-wanted)%2:
            return dict(complete=True,inconsistent=True,reason='mod-four baseline parity contradiction',
                        plus=plus,minus=minus,d=d,g_weight=g_weight,negative_weight=negative_weight,equations=())
        coeff=sum((((active>>((i+u)%d))^(active>>((i-u)%d)))&1)<<j for j,i in enumerate(positions))
        rows.append(coeff|((((baseline-wanted)//2)%2)<<m))
    rows.append(((1<<m)-1)|((negative_weight%2)<<m))
    reduced=list(rows);rank=0;pivots=[]
    for column in range(m):
        pivot=next((i for i in range(rank,len(reduced)) if reduced[i]>>column&1),None)
        if pivot is None:continue
        reduced[rank],reduced[pivot]=reduced[pivot],reduced[rank]
        for i in range(len(reduced)):
            if i!=rank and reduced[i]>>column&1:reduced[i]^=reduced[rank]
        pivots.append(column);rank+=1
    common=dict(complete=True,plus=plus,minus=minus,d=d,g_weight=g_weight,negative_weight=negative_weight,
                active=active,positions=positions,equations=tuple(rows),rank=rank)
    if any(row==(1<<m) for row in reduced):return dict(common,inconsistent=True,reason='inconsistent affine mod-four C system')
    def literal(mask):return sum(((mask>>i)&1)<<position for i,position in enumerate(positions))
    base=sum(((reduced[i]>>m)&1)<<column for i,column in enumerate(pivots))
    basis=[]
    for column in range(m):
        if column in pivots:continue
        basis.append((1<<column)|sum(((reduced[i]>>column)&1)<<p for i,p in enumerate(pivots)))
    for row in rows:
        coeff=row&((1<<m)-1)
        if bin(coeff&base).count('1')%2!=(row>>m)&1 or any(bin(coeff&v).count('1')%2 for v in basis):
            raise RuntimeError('affine C base/kernel failed original mod-four equations')
    return dict(common,inconsistent=False,base=literal(base),basis=tuple(map(literal,basis)),
                nullity=len(basis),raw_affine_assignments=1<<len(basis))


def iter_c_masks(certificate):
    if certificate['inconsistent']:return
    mask=certificate['base'];basis=certificate['basis']
    for index in range(1<<len(basis)):
        if index:
            bit=index&-index;mask^=basis[bit.bit_length()-1]
        if bin(mask).count('1')==certificate['negative_weight']:yield mask


def required_o_intersections(negative,plus,minus,d=43):
    g_weight=bin(plus|minus).count('1')
    if (d!=43 or plus&minus or bin(plus).count('1')-bin(minus).count('1')!=2
            or g_weight not in (2,4) or negative&(plus|minus) or bin(negative).count('1')!=17-g_weight//2):
        raise ValueError('literal L43 complementary C of required negative weight and sum-two g required')
    all_bits=(1<<d)-1;ea=negative|minus;eb=negative|plus;half=[]
    for u in range(1,22):
        ra=((ea<<u)|(ea>>(d-u)))&all_bits;rb=((eb<<u)|(eb>>(d-u)))&all_bits
        doubled=33-bin(ea&ra).count('1')-bin(eb&rb).count('1')
        if doubled%2:raise RuntimeError('C affine row failed even-lag mod-four condition')
        half.append(doubled//2)
    return (21,)+tuple(half)+tuple(half[::-1])


def odd_integer_match(o_mask,plus,minus,d=43):
    """ALL centered odd integer equations, target at centered +/-1."""
    indices=tuple((i,1 if plus>>i&1 else -1) for i in range(d) if (plus|minus)>>i&1)
    return all(sum(sign*(((o_mask>>((i+u)%d))&1)+((o_mask>>((i-u)%d))&1)) for i,sign in indices)
               ==2-int(u==1) for u in range(d//2+1))


def impulse_symmetry_response(s,d=43):
    """Unique rational M in (delta_s+delta_-s)*M=delta_1+delta_-1.

    The step-2s odd cycle determines M(0) exactly with denominator two.
    Every binary O requires (O+O*)/2=M, including signs of forced pairs.
    """
    from math import gcd
    if type(d) is not int or d<3 or not d%2 or type(s) is not int or not 1<=s<=d//2 or gcd(s,d)!=1:
        raise ValueError('unit impulse displacement at an odd length required')
    rhs=tuple(int((2*s*k+s)%d in (1,d-1)) for k in range(d))
    doubled=sum((-1)**(d-1-k)*v for k,v in enumerate(rhs))
    if doubled%2:raise RuntimeError('two-impulse response nonintegral')
    value=doubled//2;origin=value;response=[0]*d
    for k in range(d):response[2*s*k%d]=value;value=rhs[k]-value
    if value!=origin or any(response[(j+s)%d]+response[(j-s)%d]!=int(j in (1,d-1)) for j in range(d)):
        raise RuntimeError('two-impulse response failed original convolution')
    if any(v not in (-1,0,1) for v in response) or abs(response[0])!=1 or sum(response)!=1:
        raise RuntimeError('two-impulse response lacks required binary-compatible symmetry')
    return tuple(response)
