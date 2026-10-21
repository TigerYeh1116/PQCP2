"""Characteristic-two support families U U* + V V* = 1+X^k+X^-k for any odd prime d.

Generalizes solver/cyclic_support.py, which needs -1 in <2> mod d (d = 43).
Over R = GF(2)[X]/(X^d+1) = GF(2) x K_1 x ... x K_r (CRT), reversal * sends
the component of an irreducible factor f to that of its reciprocal f*:

- f = f* (degree m even): V_f V_f^(2^(m/2)) = r_f, a field norm equation;
  its solutions are one root times the (2^(m/2)+1) norm-one elements.
- f != f* (d = 47: two degree-23 factors): V_f * rho(V_f*) = r_f with
  rho(a) = (lift a)* mod f, so V_f may be chosen freely and V_f* follows.

The exact solution set is a product over factor classes, so sampling each
class uniformly samples V uniformly; rejecting wrong weights then gives a
uniform V among the weight-matching family. Only a necessary condition for
signed ternary completion; nothing here excludes or proves a PQCP.
"""
from dataclasses import dataclass
from functools import lru_cache
import random as _random

from .cyclic_support import field_mul, field_pow, irreducible, poly_divmod, poly_gcd, poly_mul


def _reverse(value, d):
    """Coefficient of X^i moves to X^(-i mod d)."""
    out = value & 1
    for i in range(1, d):
        if value >> i & 1:
            out |= 1 << (d-i)
    return out


def _monic_reciprocal(f):
    degree = f.bit_length()-1
    return int(format(f, '0{}b'.format(degree+1))[::-1], 2)


def _split_equal_degree(f, m, rng):
    """Cantor-Zassenhaus over GF(2): all irreducible factors of degree m of squarefree f."""
    if f.bit_length()-1 == m:
        return [f]
    while True:
        a = rng.randrange(2, 1 << (f.bit_length()-1))
        trace, power = 0, a
        for _ in range(m):
            trace ^= power
            power = poly_divmod(poly_mul(power, power), f)[1]
        g = poly_gcd(f, trace)
        if 0 < g.bit_length()-1 < f.bit_length()-1:
            other = poly_divmod(f, g)[0]
            return _split_equal_degree(g, m, rng)+_split_equal_degree(other, m, rng)


@dataclass(frozen=True)
class GF2SupportRing:
    d: int
    m: int
    factors: tuple       # X+1 first, then irreducibles of degree m
    idempotents: tuple
    partner: tuple       # partner[i] = index of the reciprocal factor (i itself if self-reciprocal)
    kernel: tuple        # norm-one generator for self-reciprocal factors, else 0


@lru_cache(maxsize=8)
def gf2_support_ring(d):
    if type(d) is not int or d < 3 or any(d % p == 0 for p in range(2, int(d**0.5)+1)):
        raise ValueError('odd prime d required')
    m = 1
    while pow(2, m, d) != 1:
        m += 1
    modulus = (1 << d) | 1
    cyclotomic = poly_divmod(modulus, 3)[0]
    found = sorted(_split_equal_degree(cyclotomic, m, _random.Random(d)))
    factors = [3]+found
    product = 1
    for f in factors:
        product = poly_mul(product, f)
    if product != modulus or any(not irreducible(f) for f in found) or len(found) != (d-1)//m:
        raise RuntimeError('cyclotomic factorization failed its exact audit')
    idempotents = []
    for f in factors:
        quotient = poly_divmod(modulus, f)[0]
        inverse = field_pow(poly_divmod(quotient, f)[1], (1 << (f.bit_length()-1))-2, f)
        e = poly_divmod(poly_mul(quotient, inverse), modulus)[1]
        if any(poly_divmod(e, g)[1] != int(g == f) for g in factors):
            raise RuntimeError('CRT idempotent failed its exact audit')
        idempotents.append(e)
    partner, kernel = [0], [0]
    for f in factors[1:]:
        j = factors.index(_monic_reciprocal(f))
        partner.append(j)
        if j != len(partner)-1:
            kernel.append(0)
            continue
        q = 1 << (m//2)
        order_primes = [p for p in range(2, q+2) if (q+1) % p == 0 and all(p % r for r in range(2, int(p**0.5)+1))]
        for candidate in range(2, 1 << m):
            h = field_pow(candidate, q-1, f)
            if field_pow(h, q+1, f) == 1 and all(field_pow(h, (q+1)//p, f) != 1 for p in order_primes):
                kernel.append(h)
                break
        else:
            raise RuntimeError('norm-one generator not found')
    return GF2SupportRing(d, m, tuple(factors), tuple(idempotents), tuple(partner), tuple(kernel))


def _component_options(ring, i, r_i):
    """(size, chooser) for one factor class; chooser(index) -> {factor index: component}."""
    f, m, j = ring.factors[i], ring.m, ring.partner[i]
    if j == i:  # norm equation v^(q+1) = r_i
        q = 1 << (m//2)
        if r_i == 0:
            return 1, lambda index: {i: 0}
        root = field_pow(r_i, q//2, f)
        if field_mul(root, field_pow(root, q, f), f) != r_i:
            raise RuntimeError('norm root failed its exact audit')
        h = ring.kernel[i]
        return q+1, lambda index: {i: field_mul(root, field_pow(h, index, f), f)}
    g = ring.factors[j]
    size = 1 << m

    def other_from(b):  # the V_g with rho(V_g) = b in K_f
        return poly_divmod(_reverse(poly_divmod(poly_mul(b, ring.idempotents[i]), (1 << ring.d) | 1)[1], ring.d), g)[1]

    if r_i:
        def choose(index):
            v = index+1
            return {i: v, j: other_from(field_mul(r_i, field_pow(v, size-2, f), f))}
        return size-1, choose

    def choose_zero(index):  # V_f = 0 with V_g free, or V_g = 0 with V_f free (nonzero)
        return {i: 0, j: index} if index < size else {i: index-size+1, j: 0}
    return 2*size-1, choose_zero


def support_family(ring, support_a, k):
    """Exact factorized family of all V with U U* + V V* = 1+X^k+X^-k (any weight)."""
    d = ring.d
    if type(support_a) is not int or not 0 <= support_a < 1 << d or type(k) is not int or not 1 <= k <= d//2:
        raise ValueError('d-bit support and independent shift 1<=k<=d/2 required')
    modulus = (1 << d) | 1
    desired = 1 ^ (1 << k) ^ (1 << (d-k))
    target = desired ^ poly_divmod(poly_mul(support_a, _reverse(support_a, d)), modulus)[1]
    classes = []
    for i in range(1, len(ring.factors)):
        if ring.partner[i] < i:
            continue
        classes.append(_component_options(ring, i, poly_divmod(target, ring.factors[i])[1]))
    base = (1 ^ bin(support_a).count('1') & 1)*ring.idempotents[0]
    return base, tuple(classes)


def _assemble(ring, base, components):
    modulus = (1 << ring.d) | 1
    value = base
    for i, v in components.items():
        value ^= poly_divmod(poly_mul(v, ring.idempotents[i]), modulus)[1]
    return value


def iter_support_family(ring, support_a, k):
    """Every family member, for small-d audits."""
    from itertools import product
    base, classes = support_family(ring, support_a, k)
    for indices in product(*(range(size) for size, _ in classes)):
        components = {}
        for (size, choose), index in zip(classes, indices):
            components.update(choose(index))
        yield _assemble(ring, base, components)


def sample_support(ring, support_a, k, weight, rng, max_tries=200000):
    """Uniform V of the given weight in the family, or None after max_tries rejections."""
    base, classes = support_family(ring, support_a, k)
    for _ in range(max_tries):
        components = {}
        for size, choose in classes:
            components.update(choose(rng.randrange(size)))
        value = _assemble(ring, base, components)
        if bin(value).count('1') == weight:
            return value
    return None
