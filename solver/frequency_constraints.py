"""Exact necessary Fourier conditions for the ORIGINAL two-sidelobe target.

These are feasibility tests for an entire fixed-content profile, not a score
threshold. No existing solution, approximate arithmetic or solver is used.
"""

from functools import lru_cache
from typing import FrozenSet

from .target_profiles import TargetContentProfile


@lru_cache(maxsize=512)
def _quarter_squares(length: int, weight: int) -> FrozenSet[int]:
    """Squared half Fourier coordinates attainable by one parity group."""
    n = length // 4
    return frozenset((weight - 2 * c) ** 2
                     for c in range(max(0, weight - n), min(n, weight) + 1))


def quarter_frequency_feasible(profile: TargetContentProfile) -> bool:
    """Whether the four parity counts can satisfy the quarter-frequency PSD.

    Only 4|L is handled; other even lengths return True (no conclusion).
    Set n=L/4 and let w be the number of ones in one parity group. Its
    residue-0 or residue-1 subgroup has c ones, where
    max(0,w-n)<=c<=min(n,w). The other subgroup has w-c ones. Consequently
    Re X(L/4)/2=w_even-2*c_even and Im X(L/4)/2=w_odd-2*c_odd,
    using positive-sine Fourier coordinates. For A and B together:

      sum_(four groups) (w_j-2*c_j)^2 = L/2 + 2*eta*cos(pi*k/2).

    The right side is the Fourier transform of T(0)=2L and
    T(k)=T(L-k)=4*eta, divided by 4; the cosine is exactly 1,0,-1,0.
    The four groups are disjoint. The finite sumset therefore enumerates ALL
    possibilities at this frequency for the profile. Discarding intermediate
    sums above the target is safe because remaining squared terms are >=0.
    False proves this content cannot realize its assigned (k,eta) PQCP target.
    It says nothing about different targets with the same content. True is
    only necessary, never a substitute for the full independent verifier.
    """
    length = profile.L
    weights = (profile.a_even_ones, profile.a_odd_ones,
               profile.b_even_ones, profile.b_odd_ones)
    if (not isinstance(length, int) or isinstance(length, bool) or length < 4
            or length % 2 or not isinstance(profile.eta, int)
            or isinstance(profile.eta, bool) or profile.eta not in (-1, 1)
            or not isinstance(profile.k, int) or isinstance(profile.k, bool)
            or not 0 < profile.k < length // 2
            or any(not isinstance(w, int) or isinstance(w, bool)
                   or not 0 <= w <= length // 2 for w in weights)):
        raise ValueError("invalid target/content profile")
    # Validate before caching: Python treats True==1 and False==0 as equal
    # keys, but malformed boolean profile fields must still be rejected.
    return _quarter_frequency_feasible_validated(profile)


@lru_cache(maxsize=4096)
def _quarter_frequency_feasible_validated(profile: TargetContentProfile) -> bool:
    length = profile.L
    if length % 4:
        return True
    weights = (profile.a_even_ones, profile.a_odd_ones,
               profile.b_even_ones, profile.b_odd_ones)
    target = length // 2 + 2 * profile.eta * (1, 0, -1, 0)[profile.k % 4]
    reachable = {0}
    for weight in weights:
        reachable = {partial + square for partial in reachable
                     for square in _quarter_squares(length, weight)
                     if partial + square <= target}
        if not reachable:
            return False
    return target in reachable
