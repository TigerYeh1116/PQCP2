"""Exact half-shift feasibility and construction for even-length PQCP seeds.

For a binary word ``a`` write its signs as ``x[i] = (-1)**a[i]`` and put
``d=L/2``.  If ``z(a)`` is the number of unordered pairs ``(i,i+d)`` whose
bits differ, then

``rho(a; d) = L - 4*z(a)``.

The Project 2 target has two distinct nonzero shifts ``k,L-k`` and therefore
the self-symmetric shift ``d`` is zero.  Every solution consequently obeys

``z(a) + z(b) = d``.

The constructor below samples *only* words ``b`` with prescribed even/odd
Hamming content that satisfy this identity with a supplied ``a``.  It is an
exact seed construction, not a verifier and not a heuristic score filter.
"""

import random
from typing import Optional, Sequence, Tuple

from .correlation import normalize_binary_sequence


BinaryWord = Tuple[int, ...]


def half_shift_mismatch_count(sequence: Sequence[int]) -> int:
    """Return the number of unequal unordered pairs ``(i,i+L/2)``."""
    bits = normalize_binary_sequence(sequence)
    if not bits or len(bits) % 2:
        raise ValueError("half-shift pairs require a positive even length")
    distance = len(bits) // 2
    return sum(bits[index] != bits[index + distance]
               for index in range(distance))


def construct_half_shift_zero_partner(
    a: Sequence[int],
    even_ones: int,
    odd_ones: int,
    rng: random.Random,
) -> Optional[BinaryWord]:
    """Construct an exact-content ``b`` with ``rho_a(L/2)+rho_b(L/2)=0``.

    ``None`` is returned if this particular ``a`` cannot be paired with the
    requested content at the half shift.  The feasibility calculation is
    exhaustive over the pair types ``00,01,10,11``; no unassigned bit is
    guessed when deciding impossibility.
    """
    a_bits = normalize_binary_sequence(a)
    length = len(a_bits)
    if not length or length % 2:
        raise ValueError("a must have positive even length")
    if not isinstance(rng, random.Random):
        raise TypeError("rng must be random.Random")
    half = length // 2
    for name, value in (("even_ones", even_ones), ("odd_ones", odd_ones)):
        if (not isinstance(value, int) or isinstance(value, bool)
                or not 0 <= value <= half):
            raise ValueError("{} must be an integer in 0..L/2".format(name))

    required_mismatches = half - half_shift_mismatch_count(a_bits)
    if not 0 <= required_mismatches <= half:  # defensive; count contract
        raise AssertionError("derived half-shift mismatch count is invalid")

    if half % 2 == 0:
        return _same_parity_partner(
            length, required_mismatches, even_ones, odd_ones, rng
        )
    return _opposite_parity_partner(
        length, required_mismatches, even_ones, odd_ones, rng
    )


def _same_parity_partner(
    length: int,
    mismatches: int,
    even_ones: int,
    odd_ones: int,
    rng: random.Random,
) -> Optional[BinaryWord]:
    """Construct the case where opposite-half endpoints have equal parity."""
    half = length // 2
    pairs_per_parity = half // 2

    def possible(group_ones: int) -> Tuple[int, ...]:
        values = []
        for unequal in range(pairs_per_parity + 1):
            remaining = group_ones - unequal
            if remaining < 0 or remaining % 2:
                continue
            double_ones = remaining // 2
            if double_ones <= pairs_per_parity - unequal:
                values.append(unequal)
        return tuple(values)

    allocations = tuple(
        (even_mismatches, mismatches - even_mismatches)
        for even_mismatches in possible(even_ones)
        if mismatches - even_mismatches in possible(odd_ones)
    )
    if not allocations:
        return None
    even_mismatches, odd_mismatches = rng.choice(allocations)
    bits = [0] * length
    for parity, group_ones, unequal in (
        (0, even_ones, even_mismatches),
        (1, odd_ones, odd_mismatches),
    ):
        pairs = list(range(parity, half, 2))
        rng.shuffle(pairs)
        double_ones = (group_ones - unequal) // 2
        for index in pairs[:double_ones]:
            bits[index] = bits[index + half] = 1
        for index in pairs[double_ones:double_ones + unequal]:
            if rng.randrange(2):
                bits[index] = 1
            else:
                bits[index + half] = 1
    return _checked_partner(bits, even_ones, odd_ones, mismatches)


def _opposite_parity_partner(
    length: int,
    mismatches: int,
    even_ones: int,
    odd_ones: int,
    rng: random.Random,
) -> Optional[BinaryWord]:
    """Construct the case where opposite-half endpoints have opposite parity."""
    half = length // 2
    numerator = even_ones + odd_ones - mismatches
    if numerator < 0 or numerator % 2:
        return None
    double_ones = numerator // 2
    even_only = even_ones - double_ones
    odd_only = odd_ones - double_ones
    if min(double_ones, even_only, odd_only) < 0:
        return None
    if double_ones + even_only + odd_only > half:
        return None

    pairs = list(range(half))
    rng.shuffle(pairs)
    bits = [0] * length
    cursor = 0
    for index in pairs[cursor:cursor + double_ones]:
        bits[index] = bits[index + half] = 1
    cursor += double_ones
    for index in pairs[cursor:cursor + even_only]:
        even_position = index if index % 2 == 0 else index + half
        bits[even_position] = 1
    cursor += even_only
    for index in pairs[cursor:cursor + odd_only]:
        odd_position = index if index % 2 else index + half
        bits[odd_position] = 1
    return _checked_partner(bits, even_ones, odd_ones, mismatches)


def _checked_partner(
    bits: Sequence[int], even_ones: int, odd_ones: int, mismatches: int,
) -> BinaryWord:
    result = tuple(bits)
    if (sum(result[0::2]), sum(result[1::2])) != (even_ones, odd_ones):
        raise RuntimeError("half-shift construction changed requested content")
    if half_shift_mismatch_count(result) != mismatches:
        raise RuntimeError("half-shift construction produced the wrong mismatch count")
    return result


__all__ = (
    "construct_half_shift_zero_partner",
    "half_shift_mismatch_count",
)
