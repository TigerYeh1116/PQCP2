"""Exact orbit keys for fixed compressed PQCP subproblems, not full pairs.

For L=m*d and c[r]=sum_j x[r+j*d], let
    x'[i] = s*x[epsilon*q*i+t mod L],
where s,epsilon are independent row signs/reflections and gcd(q,L)=1.
Compression commutes: c'[r]=s*c[epsilon*q*r+t mod d]. Each full row
transformation is bijective. Both rows must use the SAME q; then
C'(u)=C(q*u), so the target shifts become +/-q^{-1}*k, with eta unchanged.

L is even, hence q is odd. An odd t swaps the row's even/odd one counts;
s=-1 replaces each count w by L/2-w. Row exchange exchanges BOTH patterns
and their content. These transformations biject entire binary-lift solution
sets, not just their compressed PACF signatures.

If d is odd, t and t+d have identical compressed permutations but opposite
parity. Both content assignments must therefore be included. If d is even,
rotation parity is fixed; arbitrary content swaps would be UNSAFE.

Only this specified symmetry group is used. Different compression factors,
different eta, independent nontrivial decimations, arbitrary permutations,
and equal PACFs alone are never identified. Bounded-radius/partially fixed
problems need additional constraints transformed and must not use this key.
"""

from functools import lru_cache
from math import gcd
from typing import Sequence, Tuple

from .target_profiles import TargetContentProfile


RowKey = Tuple[Tuple[int, ...], int, int]
CompressedJobKey = Tuple[int, int, int, int, RowKey, RowKey]


@lru_cache(maxsize=8192)
def _row_key(word: Tuple[int, ...], even: int, odd: int, half: int) -> RowKey:
    """Minimize independent row operations, carrying content with positions."""
    d = len(word)
    best = None
    # Reflection r -> -r fixes parity; word[::-1] would add an odd offset.
    for oriented in (word, tuple(word[-i % d] for i in range(d))):
        for offset in range(d):
            rotated = oriented[offset:] + oriented[:offset]
            parities = (0, 1) if d % 2 else (offset % 2,)
            for parity in parities:
                e, o = (odd, even) if parity else (even, odd)
                for row in ((rotated, e, o),
                            (tuple(-x for x in rotated), half-e, half-o)):
                    if best is None or row < best:
                        best = row
    return best


@lru_cache(maxsize=256)
def _target_transforms(L: int, k: int) -> Tuple[int, Tuple[int, ...]]:
    """Only units with the smallest target shift can minimize the full key."""
    transforms = []
    for q in range(1, L):
        if gcd(q, L) == 1:
            shift = (pow(q, -1, L) * k) % L
            transforms.append((min(shift, L-shift), q))
    shift = min(value for value, _ in transforms)
    return shift, tuple(q for value, q in transforms if value == shift)


@lru_cache(maxsize=4096)
def _pair_key(a: Tuple[int, ...], b: Tuple[int, ...], target: TargetContentProfile,
              factor: int) -> CompressedJobKey:
    L, d = target.L, len(a)
    k, units = _target_transforms(L, target.k)
    best = None
    # q modulo d suffices for patterns; ALL full q above are odd and preserve
    # full-sequence parity. Do not infer parity from q modulo odd d.
    for q in sorted({q % d for q in units}):
        aa = tuple(a[q*i % d] for i in range(d))
        bb = tuple(b[q*i % d] for i in range(d))
        rows = tuple(sorted((
            _row_key(aa, target.a_even_ones, target.a_odd_ones, L//2),
            _row_key(bb, target.b_even_ones, target.b_odd_ones, L//2),
        )))
        if best is None or rows < best:
            best = rows
    return L, factor, target.eta, k, best[0], best[1]


def compressed_job_equivalence_key(
    a: Sequence[int], b: Sequence[int], target: TargetContentProfile, factor: int,
) -> CompressedJobKey:
    """Canonical key of (patterns, full target, parity content, factor).

    This does not transform the stored representative or assert it has a
    solution. Equal keys mean equivalent unrestricted fixed-compression lift
    problems. Distinct keys need not rule out some other, unimplemented
    equivalence. Input feasibility is left to the existing exact solvers.
    """
    if not isinstance(target, TargetContentProfile):
        raise ValueError("target must be a TargetContentProfile")
    L = target.L
    if type(L) is not int or L < 4 or L % 2:
        raise ValueError("compressed equivalence requires even L >= 4")
    if type(factor) is not int or factor not in (2, 4) or L % factor:
        raise ValueError("factor must be 2 or 4 and divide L")
    if (type(target.k) is not int or not 0 < target.k < L//2
            or type(target.eta) is not int or target.eta not in (-1, 1)):
        raise ValueError("invalid target shift/sign")
    content = (target.a_even_ones, target.a_odd_ones,
               target.b_even_ones, target.b_odd_ones)
    if any(type(w) is not int or not 0 <= w <= L//2 for w in content):
        raise ValueError("invalid parity content")
    a, b = tuple(a), tuple(b)
    if len(a) != L//factor or len(b) != L//factor:
        raise ValueError("compressed length does not match L/factor")
    if any(type(x) is not int or abs(x) > factor or (x-factor) % 2 for x in a+b):
        raise ValueError("invalid compressed alphabet")
    return _pair_key(a, b, target, factor)
