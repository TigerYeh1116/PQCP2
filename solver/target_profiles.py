"""Safe Project 2 target/content profiles derived from two Fourier characters.

Write binary sequences as signs ``x_i=(-1)^a_i`` and ``y_i=(-1)^b_i``.
For a target whose only nonzero sidelobes are ``4*eta`` at ``k,L-k``:

``sum_u C(u) = (sum_i x_i)^2 + (sum_i y_i)^2 = 2L + 8*eta``

and, for even L, evaluation at the alternating character gives

``sum_u (-1)^u C(u) = Alt(x)^2 + Alt(y)^2
                      = 2L + 8*eta*(-1)^k``.

After dividing all four sequence sums by two, these are exactly the two
sum-of-two-squares identities used below.  They are necessary conditions,
not sufficient PQCP tests; the independent verifier remains authoritative.
"""

from dataclasses import dataclass
from math import gcd
from typing import Iterable, Optional, Sequence, Tuple

from .correlation import normalize_binary_sequence


@dataclass(frozen=True, order=True)
class TargetContentProfile:
    """One target shift/sign and exact even/odd one counts for A and B."""

    L: int
    k: int
    eta: int
    a_even_ones: int
    a_odd_ones: int
    b_even_ones: int
    b_odd_ones: int

    @property
    def target_value(self) -> int:
        return 4 * self.eta

    @property
    def weight_pair(self) -> Tuple[int, int]:
        return (
            self.a_even_ones + self.a_odd_ones,
            self.b_even_ones + self.b_odd_ones,
        )

    @property
    def alternating_halves(self) -> Tuple[int, int]:
        """Return Alt(A)/2 and Alt(B)/2 in sign notation."""
        return (
            self.a_odd_ones - self.a_even_ones,
            self.b_odd_ones - self.b_even_ones,
        )


def decimation_shift_representatives(length: int) -> Tuple[int, ...]:
    """Return one nonzero target shift per unit-decimation orbit.

    Multiplication ``i -> q*i mod L`` for ``gcd(q,L)=1`` permutes sequence
    coordinates and maps a target shift to ``q*k mod L``.  Unit orbits in
    ``Z/LZ`` are classified by ``gcd(k,L)``.  Periodic symmetry identifies
    ``k`` and ``L-k``, so representatives are chosen from ``1..L/2-1``.

    This reduces target-position cases only; it does not canonicalize complete
    sequence pairs or justify pruning a bounded ball around an unchanged
    centre.
    """
    _validate_even_length(length)
    representatives = {}
    for shift in range(1, length // 2):
        representatives.setdefault(gcd(shift, length), shift)
    return tuple(representatives[key] for key in sorted(representatives))


def target_content_profiles(
    length: int,
    decimation_reduced: bool = False,
) -> Tuple[TargetContentProfile, ...]:
    """Enumerate every necessary parity-content profile for Project 2.

    ``decimation_reduced=True`` retains one target ``k`` per proven
    decimation orbit.  All signs and ordered A/B contents are still included.
    """
    _validate_even_length(length)
    half = length // 2
    shifts: Iterable[int] = (
        decimation_shift_representatives(length)
        if decimation_reduced else range(1, half)
    )
    profiles = set()
    for eta in (-1, 1):
        ordinary_pairs = _signed_square_pairs(half + 2 * eta, half)
        for k in shifts:
            alternating_pairs = _signed_square_pairs(
                half + 2 * eta * (-1 if k % 2 else 1), half
            )
            for ordinary_a, ordinary_b in ordinary_pairs:
                weight_a, weight_b = half - ordinary_a, half - ordinary_b
                for alternate_a, alternate_b in alternating_pairs:
                    counts = _parity_counts(
                        half, weight_a, alternate_a, weight_b, alternate_b
                    )
                    if counts is not None:
                        profiles.add(TargetContentProfile(
                            length, k, eta,
                            counts[0], counts[1], counts[2], counts[3],
                        ))
    return tuple(sorted(profiles))


def canonical_target_content_profiles(length: int) -> Tuple[TargetContentProfile, ...]:
    """Reduce profiles under exact PACF-preserving pair symmetries.

    A/B exchange, individual sequence complementation and an independent
    one-position cyclic rotation of either sequence all preserve each
    periodic autocorrelation and hence the target ``k,eta``.  A one-position
    rotation exchanges that sequence's even/odd one counts.  Odd unit
    decimations preserve index parity for even L, so these reductions safely
    combine with the target-position representatives.  This function only
    chooses one *content* representative; it does not claim to canonicalize
    complete words.
    """
    half = length // 2
    canonical = set()
    for profile in target_content_profiles(length, decimation_reduced=True):
        variants = []
        for complement_a in (False, True):
            for complement_b in (False, True):
                ae = half - profile.a_even_ones if complement_a else profile.a_even_ones
                ao = half - profile.a_odd_ones if complement_a else profile.a_odd_ones
                be = half - profile.b_even_ones if complement_b else profile.b_even_ones
                bo = half - profile.b_odd_ones if complement_b else profile.b_odd_ones
                for rotate_a in (False, True):
                    a_counts = (ao, ae) if rotate_a else (ae, ao)
                    for rotate_b in (False, True):
                        b_counts = (bo, be) if rotate_b else (be, bo)
                        variants.append(a_counts + b_counts)
                        variants.append(b_counts + a_counts)
        ae, ao, be, bo = min(variants)
        canonical.add(TargetContentProfile(length, profile.k, profile.eta, ae, ao, be, bo))
    return tuple(sorted(canonical))


def pair_content(a: Sequence[int], b: Sequence[int]) -> Tuple[int, int, int, int]:
    """Return ``A_even,A_odd,B_even,B_odd`` one counts."""
    a_bits, b_bits = normalize_binary_sequence(a), normalize_binary_sequence(b)
    if len(a_bits) != len(b_bits) or len(a_bits) % 2:
        raise ValueError("a and b must have equal positive even length")
    return (
        sum(a_bits[0::2]), sum(a_bits[1::2]),
        sum(b_bits[0::2]), sum(b_bits[1::2]),
    )


def profile_matches_pair_content(
    profile: TargetContentProfile,
    a: Sequence[int],
    b: Sequence[int],
) -> bool:
    """Check only exact parity content; this is not a PQCP verification."""
    return profile.L == len(a) == len(b) and pair_content(a, b) == (
        profile.a_even_ones, profile.a_odd_ones,
        profile.b_even_ones, profile.b_odd_ones,
    )


def _signed_square_pairs(target: int, bound: int) -> Tuple[Tuple[int, int], ...]:
    if target < 0:
        return ()
    return tuple(
        (left, right)
        for left in range(-bound, bound + 1)
        for right in range(-bound, bound + 1)
        if left * left + right * right == target
    )


def _parity_counts(
    half: int,
    weight_a: int,
    alternate_a: int,
    weight_b: int,
    alternate_b: int,
) -> Optional[Tuple[int, int, int, int]]:
    numerators = (
        weight_a - alternate_a, weight_a + alternate_a,
        weight_b - alternate_b, weight_b + alternate_b,
    )
    if any(value % 2 for value in numerators):
        return None
    counts = tuple(value // 2 for value in numerators)
    if any(not 0 <= value <= half for value in counts):
        return None
    return counts[0], counts[1], counts[2], counts[3]


def _validate_even_length(length: int) -> None:
    if not isinstance(length, int) or isinstance(length, bool) or length < 4 or length % 2:
        raise ValueError("target profiles require an even integer length at least four")
