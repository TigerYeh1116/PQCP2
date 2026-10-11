"""ALL skew binary support preimages of a cyclic F2 norm target.

U+U*=J+1 fixes one negative symbol in each nonzero reflected pair. In every
self-reciprocal field F_(q^2), U*=U^q, so U^q=U+1 and norm=U(U+1).
Enumerating the full affine Frobenius coset gives TWO roots per admissible
norm, not 129 unrestricted phases. L43 has at most 2^3=8 literal preimages.
No rotation quotient is allowed: the skew center is fixed at coordinate zero.
"""
from functools import lru_cache
from itertools import product

from .cyclic_support import support_ring,field_pow,field_mul,poly_mul,poly_divmod
from .sds_norm_completion import cyclic_binary_norm


@lru_cache(maxsize=8)
def _skew_field_roots(factor):
    degree=factor.bit_length()-1;q=1<<(degree//2)
    if degree%2:raise RuntimeError('even extension degree required for skew Frobenius')
    basis=tuple(field_pow(1<<i,q,factor) for i in range(degree))
    value=frobenius=0;groups={};count=0
    # Gray enumeration of EVERY field element, auditing the full linear coset.
    for cursor in range(1<<degree):
        if cursor:
            bit=(cursor&-cursor).bit_length()-1
            value^=1<<bit;frobenius^=basis[bit]
        if frobenius!=value^1:continue
        count+=1;norm=field_mul(value,value^1,factor)
        if field_pow(norm,q,factor)!=norm:raise RuntimeError('skew norm outside fixed field')
        groups.setdefault(norm,[]).append(value)
    if count!=q or len(groups)!=q//2 or any(len(v)!=2 or v[0]^v[1]!=1 for v in groups.values()):
        raise RuntimeError('complete affine Frobenius-coset count failed')
    return {n:tuple(sorted(v)) for n,v in groups.items()}


@lru_cache(maxsize=512)
def _lifted_roots(d,factor,idempotent,norm):
    modulus=(1<<d)|1
    return tuple(poly_divmod(poly_mul(v,idempotent),modulus)[1]
                 for v in _skew_field_roots(factor).get(norm,()))


def skew_norm_preimages(desired,d=43):
    """Complete literal masks; every retained row has sum +1 and fixed skew center."""
    ring=support_ring(d)
    if type(desired) is not int or not 0<=desired<1<<d:
        raise ValueError('literal cyclic F2 norm target required')
    reverse=sum(((desired>>i)&1)<<((-i)%d) for i in range(d))
    if reverse!=desired:raise ValueError('symmetric cyclic norm target required')
    parity=((d-1)//2)%2
    if bin(desired).count('1')%2!=parity:return dict(complete=True,masks=(),reason='skew row scalar parity obstruction')
    groups=[]
    for factor,idempotent in zip(ring.factors[1:],ring.idempotents[1:]):
        norm=poly_divmod(desired,factor)[1]
        group=_lifted_roots(d,factor,idempotent,norm)
        if not group:return dict(complete=True,masks=(),reason='fixed-field norm has no affine skew roots',factor=factor)
        groups.append(group)
    masks=[]
    for parts in product(*groups):
        mask=parity*ring.idempotents[0]
        for part in parts:mask^=part
        if (mask&1 or bin(mask).count('1')!=(d-1)//2
                or any(((mask>>i)^(mask>>(d-i)))&1!=1 for i in range(1,d))
                or cyclic_binary_norm(mask,d)!=desired):
            raise RuntimeError('skew norm reconstruction failed literal ALL-coordinate/norm audit')
        masks.append(mask)
    return dict(complete=True,masks=tuple(masks),norm_target=desired,
                maximum_preimages=1<<len(groups),rotation_quotient=False,
                scope='ALL sum-one skew binary supports for this norm target, center zero. '
                      'Skew is an explicit construction restriction, not a general PQCP filter.')
