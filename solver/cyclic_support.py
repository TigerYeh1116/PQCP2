"""Exact characteristic-two cyclic support norms (lengths 3, 5, 11, 43).

For odd prime d with -1 in <2> mod d, each nontrivial factor of X^d+1
is a self-reciprocal irreducible of even degree m. Cyclic reversal is
Frobenius^(m/2), so U U* + V V* = 1+X^k+X^-k is a field norm equation.
CRT parameterizes ALL parity-compatible V for fixed U. This is only a
necessary condition for signed ternary completion and the binary lift.
"""
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Tuple


def poly_mul(a, b):
    result = 0
    while b:
        if b & 1: result ^= a
        a <<= 1; b >>= 1
    return result


def poly_divmod(a, b):
    if not b: raise ZeroDivisionError('zero polynomial')
    quotient = 0; degree = b.bit_length()-1
    while a and a.bit_length()-1 >= degree:
        shift = a.bit_length()-1-degree
        quotient ^= 1 << shift; a ^= b << shift
    return quotient, a


def field_mul(a, b, modulus):
    return poly_divmod(poly_mul(a, b), modulus)[1]


def field_pow(a, exponent, modulus):
    result = 1
    while exponent:
        if exponent & 1: result = field_mul(result, a, modulus)
        a = field_mul(a, a, modulus); exponent >>= 1
    return result


def poly_gcd(a, b):
    while b: a, b = b, poly_divmod(a, b)[1]
    return a


def irreducible(f):
    degree = f.bit_length()-1; x = poly_divmod(2, f)[1]; value = x
    for i in range(1, degree+1):
        value = field_mul(value, value, f)
        if i <= degree//2 and poly_gcd(value ^ x, f) != 1: return False
    return value == x


def _prime_divisors(n):
    result = []; p = 2
    while p*p <= n:
        if n % p == 0:
            result.append(p)
            while n % p == 0: n //= p
        p += 1
    if n > 1: result.append(n)
    return result


@dataclass(frozen=True)
class CyclicSupportRing:
    d: int
    degree: int
    factors: Tuple[int, ...]  # X+1 first, followed by irreducibles
    idempotents: Tuple[int, ...]
    norm_generators: Tuple[int, ...]


@lru_cache(maxsize=4)
def support_ring(d):
    if type(d) is not int or d not in (3, 5, 11, 43):
        raise ValueError('supported odd prime half-lengths: 3,5,11,43')
    degree = 1
    while pow(2, degree, d) != 1: degree += 1
    if degree % 2 or pow(2, degree//2, d) != d-1:
        raise RuntimeError('reversal is not the required Frobenius involution')
    remaining = (1 << d)-1; factors = [3]
    for middle in range(1 << (degree-1)):
        factor = (1 << degree) | (middle << 1) | 1
        quotient, residue = poly_divmod(remaining, factor)
        if residue == 0:
            if not irreducible(factor): raise RuntimeError('reducible field factor')
            factors.append(factor); remaining = quotient
            if remaining == 1: break
    modulus = (1 << d) | 1
    product = 1
    for factor in factors: product = poly_mul(product, factor)
    if remaining != 1 or product != modulus:
        raise RuntimeError('incomplete cyclic factorization')
    idempotents = []
    for factor in factors:
        quotient, residue = poly_divmod(modulus, factor)
        if residue: raise RuntimeError('CRT divisor mismatch')
        reduced = poly_divmod(quotient, factor)[1]
        inverse = field_pow(reduced, (1 << (factor.bit_length()-1))-2, factor)
        element = poly_divmod(poly_mul(quotient, inverse), modulus)[1]
        for other in factors:
            if poly_divmod(element, other)[1] != int(factor == other):
                raise RuntimeError('CRT idempotent failed exact audit')
        idempotents.append(element)
    q = 1 << (degree//2); generators = []
    for factor in factors[1:]:
        for candidate in range(2, 1 << degree):
            h = field_pow(candidate, q-1, factor)
            if (field_pow(h, q+1, factor) == 1 and
                    all(field_pow(h, (q+1)//p, factor) != 1 for p in _prime_divisors(q+1))):
                generators.append(h); break
        else: raise RuntimeError('norm-one subgroup generator not found')
    return CyclicSupportRing(d, degree, tuple(factors), tuple(idempotents), tuple(generators))


@dataclass(frozen=True)
class NormSupportFamily:
    d: int
    support_a: int
    k: int
    base: int
    choices: Tuple[Tuple[int, ...], ...]
    norm_target: Optional[int] = None

    @property
    def size(self):
        value = 1
        for group in self.choices: value *= len(group)
        return value


@lru_cache(maxsize=512)
def _norm_options(d, degree, factor, idempotent, generator, norm):
    """A bounded exact cache: L43 has only 3*128 possible field norms."""
    q = 1 << (degree//2)
    modulus = (1 << d) | 1
    if field_pow(norm, q, factor) != norm:
        raise RuntimeError('target norm outside fixed field')
    root = field_pow(norm, q//2, factor)
    if field_mul(root, field_pow(root, q, factor), factor) != norm:
        raise RuntimeError('norm square root failed exact audit')
    options = []
    value = root
    for _ in range(q+1 if norm else 1):
        options.append(poly_divmod(poly_mul(value, idempotent), modulus)[1])
        value = field_mul(value, generator, factor)
    if len(set(options)) != len(options):
        raise RuntimeError('duplicate field phase')
    return tuple(options)


def parity_support_family(support_a, d, k):
    """ALL V with U U*+V V*=1+X^k+X^(d-k), before weight/sign constraints."""
    ring = support_ring(d)
    if type(support_a) is not int or not 0 <= support_a < 1 << d:
        raise ValueError('support_a must be a d-bit nonnegative integer')
    if type(k) is not int or not 1 <= k <= d//2:
        raise ValueError('k must be an independent nonzero shift')
    q = 1 << (ring.degree//2)
    desired = 1 ^ (1 << k) ^ (1 << (d-k))
    base = (1 ^ (bin(support_a).count('1') & 1))*ring.idempotents[0]
    groups = []
    for factor, idempotent, generator in zip(ring.factors[1:], ring.idempotents[1:], ring.norm_generators):
        u = poly_divmod(support_a, factor)[1]
        norm = poly_divmod(desired, factor)[1] ^ field_mul(u, field_pow(u, q, factor), factor)
        groups.append(_norm_options(d, ring.degree, factor, idempotent, generator, norm))
    return NormSupportFamily(d, support_a, k, base, tuple(groups))


def cyclic_norm_family(desired, d):
    """ALL U with U U*=desired, for any symmetric characteristic-two target.

    support_a/k are unused for this generic family. Its explicit norm_target
    prevents confusing it with the fixed-side PQCP support problem.
    """
    ring = support_ring(d)
    if type(desired) is not int or not 0 <= desired < 1 << d:
        raise ValueError('d-bit characteristic-two norm target required')
    reverse = sum(((desired >> i) & 1) << ((d-i) % d) for i in range(d))
    if reverse != desired:
        raise ValueError('norm target must be cyclically symmetric')
    base = (bin(desired).count('1') & 1)*ring.idempotents[0]
    groups = []
    for factor, idempotent, generator in zip(ring.factors[1:], ring.idempotents[1:], ring.norm_generators):
        norm = poly_divmod(desired, factor)[1]
        groups.append(_norm_options(d, ring.degree, factor, idempotent, generator, norm))
    return NormSupportFamily(d, 0, 0, base, tuple(groups), desired)


def iter_family(family):
    """Literal Python CRT enumeration, retained as an independent small oracle."""
    from itertools import product
    for choice in product(*family.choices):
        value = family.base
        for component in choice: value ^= component
        yield value


def rotation_representatives(family):
    """Exactly one V per cyclic-rotation orbit, with fixed U unchanged.

    If a field component is nonzero, rotation adds a generator of the
    order-d subgroup to its norm phase. Representatives of those cosets
    are phases 0..(q+1)/d-1. All-zero field components give a constant V
    whose orbit has size one. This is a symmetry quotient, not invariance.
    It must not be used as a literal fixed-centre Hamming-ball enumeration.
    """
    ring = support_ring(family.d)
    order = (1 << (ring.degree//2))+1
    if order % family.d:
        raise RuntimeError('rotation subgroup does not divide norm kernel')
    groups = list(family.choices)
    for i, group in enumerate(groups):
        if len(group) == order:
            groups[i] = group[:order//family.d]
            return NormSupportFamily(family.d, family.support_a, family.k, family.base, tuple(groups), family.norm_target)
        if len(group) != 1:
            raise ValueError('unmodified norm family required for rotation quotient')
    return family
