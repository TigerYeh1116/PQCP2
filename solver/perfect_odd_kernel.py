"""Certified integral kernel for identical perfect-43 odd interleave rows.

Let O have sum 1 and PACF -1. g_t=(shift_t O-shift_-t O)/2 satisfies
Gram(g)=22 I and symmetric_cross(g_t,O)=0. Ranks 21 modulo 2 and 11
certify the basis is primitive: any integer kernel vector has integral
coefficients. Trit vectors of norm <=43 are therefore 0 or +/- one g_t.
This is a restricted-family theorem, never a global L86 exclusion.
"""
from .interleaving import interleave_bits,periodic_sign_profile


def rank_mod_prime(rows,p):
    a=[list(row) for row in rows]
    rank=0
    for column in range(len(a[0]) if a else 0):
        pivot=next((i for i in range(rank,len(a)) if a[i][column]%p),None)
        if pivot is None:continue
        a[rank],a[pivot]=a[pivot],a[rank]
        inverse=pow(a[rank][column]%p,-1,p)
        a[rank]=[v*inverse%p for v in a[rank]]
        for i in range(len(a)):
            if i==rank:continue
            value=a[i][column]%p
            if value:a[i]=[(x-value*y)%p for x,y in zip(a[i],a[rank])]
        rank+=1
    return rank


def perfect43_kernel_certificate(o):
    o=tuple(o)
    if len(o)!=43 or sum(o)!=1 or periodic_sign_profile(o)!=(43,)+(-1,)*42:
        raise ValueError('independently verified perfect sign row of sum 1 required')
    basis=tuple(tuple((o[(i-t)%43]-o[(i+t)%43])//2 for i in range(43)) for t in range(1,22))
    gram=tuple(tuple(sum(x*y for x,y in zip(a,b)) for b in basis) for a in basis)
    if any(v!=(22 if i==j else 0) for i,row in enumerate(gram) for j,v in enumerate(row)):
        raise RuntimeError('perfect-row kernel Gram identity failed')
    ranks={p:rank_mod_prime(basis,p) for p in (2,11)}
    if ranks!={2:21,11:21}:
        raise RuntimeError('integral kernel basis not certified primitive; cannot exclude other vectors')
    # Audit the antisymmetric convolution for EVERY basis vector, too.
    for t,g in enumerate(basis,1):
        cross=tuple(sum(g[(i+u)%43]*o[i] for i in range(43)) for u in range(43))
        if cross!=tuple(22*(int(u==t)-int(u==43-t)) for u in range(43)):
            raise RuntimeError('perfect-row odd-kernel convolution failed')
    return dict(basis=basis,gram_diagonal=22,ranks=ranks,primitive=True,
        kernel_dimension=21,scope='Same perfect-43 O in both odd rows, up to independent row rotations.')


def interleave_from_complementary_trits(c,g,o):
    d=len(o)
    if len(c)!=d or len(g)!=d or d%2!=1 or any((a!=0)+(b!=0)!=1 for a,b in zip(c,g)):
        raise ValueError('complementary trit rows and an odd O length required')
    ea,eb=tuple(a+b for a,b in zip(c,g)),tuple(a-b for a,b in zip(c,g))
    return (interleave_bits(ea,o,(d+1)//2),interleave_bits(eb,o,(d+1)//2,-1))
