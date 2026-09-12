"""Mathematically safe pruning for complete or partial binary PQCP inputs.

No score threshold is used here.  A rejection is made only when a stated
necessary condition for a Project 2 solution is violated.
"""

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, Union

from .objective import (
    ALLOWED_NONZERO_SHIFT_VALUES,
    REQUIRED_NONZERO_SHIFTS,
    _validate_profile,
)
from .structured_energy import target_profile
from .target_profiles import TargetContentProfile


PartialBit = Optional[Union[int, str]]
PartialSequence = Sequence[PartialBit]


@dataclass(frozen=True)
class PruningResult:
    """Whether pruning is proven safe, together with the violated conditions."""

    should_prune: bool
    reasons: Tuple[str, ...]


@dataclass(frozen=True)
class CorrelationBounds:
    """Conservative lower/upper bounds for each pair-correlation shift."""

    lower: Tuple[int, ...]
    upper: Tuple[int, ...]


@dataclass(frozen=True)
class FixedContentCorrelationBounds:
    """Pair-correlation bounds conditional on exact even/odd one counts.

    ``content_feasible`` is false when the known bits already use too many
    ones, or the remaining unknown positions cannot supply enough ones.  In
    that case no completion exists and ``lower``/``upper`` are empty.
    """

    content_feasible: bool
    lower: Tuple[int, ...]
    upper: Tuple[int, ...]


def prune_complete_profile(profile: Sequence[int]) -> PruningResult:
    """Safely reject a complete profile that cannot be a Project 2 target.

    Conditions are necessary: S[0]=2L; binary pair correlations obey
    S[u] == S[L-u] and S[u] == 2L (mod 4); nonzero shifts must be +/-4; and
    exactly two actual nonzero shift indices are required.  The count follows
    the Project 2 convention, so a symmetric pair u and L-u counts as two.
    """
    values = _validate_profile(profile)
    length = len(values)
    reasons: List[str] = []

    if values[0] != 2 * length:
        reasons.append("S[0] is not 2L")
    if any(values[shift] != values[(-shift) % length] for shift in range(1, length)):
        reasons.append("periodic symmetry S[u] = S[L-u] is violated")

    required_residue = (2 * length) % 4
    if any(values[shift] % 4 != required_residue for shift in range(1, length)):
        reasons.append("binary pair-correlation congruence modulo 4 is violated")

    nonzero_shifts = tuple(shift for shift in range(1, length) if values[shift] != 0)
    if any(abs(values[shift]) != 4 for shift in nonzero_shifts):
        reasons.append("a nonzero shift has magnitude other than 4")
    if len(nonzero_shifts) != REQUIRED_NONZERO_SHIFTS:
        reasons.append("the actual nonzero-shift count is not two")

    return PruningResult(bool(reasons), tuple(reasons))


def partial_correlation_bounds(a: PartialSequence, b: PartialSequence) -> CorrelationBounds:
    """Return conservative bounds for every S(u) of a partial binary pair.

    A known term contributes exactly +1 or -1.  A term involving one or two
    unspecified bits is bounded independently by [-1, +1].  Dependencies
    between unknown terms are deliberately ignored, widening rather than
    narrowing the interval; therefore every binary completion remains within
    these bounds.
    """
    a_bits = _normalize_partial_sequence(a)
    b_bits = _normalize_partial_sequence(b)
    if len(a_bits) != len(b_bits):
        raise ValueError("a and b must have equal lengths")
    length = len(a_bits)
    lower = []
    upper = []
    for shift in range(length):
        known_sum = 0
        unknown_terms = 0
        for sequence in (a_bits, b_bits):
            for index in range(length):
                left = sequence[index]
                right = sequence[(index + shift) % length]
                if left is None or right is None:
                    unknown_terms += 1
                else:
                    known_sum += 1 if left == right else -1
        lower.append(known_sum - unknown_terms)
        upper.append(known_sum + unknown_terms)
    return CorrelationBounds(tuple(lower), tuple(upper))


def fixed_content_partial_correlation_bounds(
    a: PartialSequence,
    b: PartialSequence,
    target: TargetContentProfile,
) -> FixedContentCorrelationBounds:
    """Bound every ``S(u)`` using the target's fixed parity contents.

    For one sequence, write ``rho(u)=L-2D(u)``, where ``D(u)`` is the number
    of directed mismatch edges ``i -> i+u``.  Known-known edges contribute
    exactly.  For each known-unknown edge we aggregate the cost of assigning
    that unknown bit 0 or 1.  Because the number of remaining ones is fixed
    separately on even and odd positions, sorting these per-bit cost changes
    gives exact extrema for the known-unknown part.

    Unknown-unknown edges are conservatively bounded.  Each mismatching edge
    consumes an incidence at one remaining 1 and one remaining 0, and every
    position has at most two directed incidences.  Thus their count is at most
    ``min(E, 2*ones, 2*zeros)``; for parity-preserving shifts the same bound is
    applied separately to even and odd vertices.  Ignoring dependencies
    between the boundary and unknown-unknown extrema can only widen the final
    interval, so every fixed-content completion is guaranteed to remain in
    the returned bounds.

    The target fixes four disjoint contents: A-even, A-odd, B-even and B-odd.
    This routine changes neither the Project target nor its counting
    convention and performs no heuristic score pruning.
    """
    a_bits = _normalize_partial_sequence(a)
    b_bits = _normalize_partial_sequence(b)
    if len(a_bits) != len(b_bits):
        raise ValueError("a and b must have equal lengths")
    if not isinstance(target, TargetContentProfile) or target.L != len(a_bits):
        raise ValueError("target must be a matching TargetContentProfile")
    # Reuse the established target validation, including k and eta.
    target_profile(target.L, target.k, target.eta)

    a_remaining = _remaining_parity_content(
        a_bits, target.a_even_ones, target.a_odd_ones
    )
    b_remaining = _remaining_parity_content(
        b_bits, target.b_even_ones, target.b_odd_ones
    )
    if a_remaining is None or b_remaining is None:
        return FixedContentCorrelationBounds(False, (), ())

    lower = []
    upper = []
    for shift in range(target.L):
        a_lower, a_upper = _fixed_content_sequence_rho_bounds(
            a_bits, a_remaining, shift
        )
        b_lower, b_upper = _fixed_content_sequence_rho_bounds(
            b_bits, b_remaining, shift
        )
        lower.append(a_lower + b_lower)
        upper.append(a_upper + b_upper)
    return FixedContentCorrelationBounds(True, tuple(lower), tuple(upper))


def prune_partial_assignment(
    a: PartialSequence,
    b: PartialSequence,
    target: Optional[TargetContentProfile] = None,
) -> PruningResult:
    """Safely reject only partial pairs whose completions cannot meet the target.

    For each nonzero shift, its conservative interval must contain at least
    one allowed final value in {-4, 0, +4}.  Also, if zero lies outside an
    interval, that actual shift is forced nonzero; more than two forced
    nonzero indices cannot be repaired by any completion.  These tests are
    necessary conditions, not objective-based heuristics.
    """
    if target is not None:
        return prune_fixed_content_partial_assignment(a, b, target)

    bounds = partial_correlation_bounds(a, b)
    length = len(bounds.lower)
    reasons: List[str] = []

    required_residue = (2 * length) % 4
    compatible_targets = tuple(
        value for value in ALLOWED_NONZERO_SHIFT_VALUES if value % 4 == required_residue
    )
    if not compatible_targets:
        reasons.append("no Project 2 target value satisfies binary modulo-4 congruence")
        return PruningResult(True, tuple(reasons))

    impossible_shifts = tuple(
        shift for shift in range(1, length)
        if not any(bounds.lower[shift] <= value <= bounds.upper[shift]
                   for value in compatible_targets)
    )
    if impossible_shifts:
        reasons.append("a shift bound cannot reach any allowed target value")

    forced_nonzero = sum(
        0 < bounds.lower[shift] or bounds.upper[shift] < 0
        for shift in range(1, length)
    )
    if forced_nonzero > REQUIRED_NONZERO_SHIFTS:
        reasons.append("more than two actual shifts are forced nonzero")

    return PruningResult(bool(reasons), tuple(reasons))


def prune_fixed_content_partial_assignment(
    a: PartialSequence,
    b: PartialSequence,
    target: TargetContentProfile,
) -> PruningResult:
    """Safely prune a partial pair against one exact target/content case.

    A rejection occurs only when the partial bits cannot meet one of the four
    exact parity counts, or an assigned Project target value lies outside the
    conservative fixed-content correlation interval for some shift.
    """
    bounds = fixed_content_partial_correlation_bounds(a, b, target)
    if not bounds.content_feasible:
        return PruningResult(
            True,
            ("partial assignment cannot meet the fixed parity content",),
        )

    desired = target_profile(target.L, target.k, target.eta)
    impossible = tuple(
        shift for shift, value in enumerate(desired)
        if not bounds.lower[shift] <= value <= bounds.upper[shift]
    )
    reasons = () if not impossible else (
        "a fixed-content shift bound cannot reach its assigned target value",
    )
    return PruningResult(bool(reasons), reasons)


def _remaining_parity_content(
    bits: Tuple[Optional[int], ...],
    even_ones: int,
    odd_ones: int,
) -> Optional[Tuple[int, int]]:
    """Return required ones among unknown even/odd positions, or infeasible."""
    half = len(bits) // 2
    if len(bits) % 2 or any(
        not isinstance(value, int) or isinstance(value, bool)
        or not 0 <= value <= half
        for value in (even_ones, odd_ones)
    ):
        raise ValueError("fixed parity contents require even L and valid counts")
    remaining = []
    for parity, target_ones in enumerate((even_ones, odd_ones)):
        known = sum(bit == 1 for bit in bits[parity::2] if bit is not None)
        unknown = sum(bit is None for bit in bits[parity::2])
        needed = target_ones - known
        if not 0 <= needed <= unknown:
            return None
        remaining.append(needed)
    return remaining[0], remaining[1]


def _fixed_content_sequence_rho_bounds(
    bits: Tuple[Optional[int], ...],
    remaining_ones: Tuple[int, int],
    shift: int,
) -> Tuple[int, int]:
    """Return safe min/max rho for one fixed-content partial sequence."""
    length = len(bits)
    unknown_positions = tuple(index for index, bit in enumerate(bits) if bit is None)
    boundary_base = {index: 0 for index in unknown_positions}
    boundary_delta = {index: 0 for index in unknown_positions}
    known_mismatches = 0
    unknown_edges = [0, 0]

    for index in range(length):
        partner = (index + shift) % length
        left, right = bits[index], bits[partner]
        if left is not None and right is not None:
            known_mismatches += left != right
        elif left is None and right is None:
            if index != partner:
                # For an even shift both endpoints have this parity.  For an
                # odd shift the combined count is all that is needed below.
                unknown_edges[index % 2] += 1
        else:
            unknown = index if left is None else partner
            known = right if left is None else left
            cost_zero = int(known == 1)
            cost_one = int(known == 0)
            boundary_base[unknown] += cost_zero
            boundary_delta[unknown] += cost_one - cost_zero

    boundary_min = sum(boundary_base.values())
    boundary_max = boundary_min
    unknown_counts = []
    for parity in (0, 1):
        positions = [index for index in unknown_positions if index % 2 == parity]
        unknown_counts.append(len(positions))
        deltas = sorted(boundary_delta[index] for index in positions)
        ones = remaining_ones[parity]
        boundary_min += sum(deltas[:ones])
        boundary_max += sum(deltas[len(deltas) - ones:]) if ones else 0

    if shift % 2 == 0:
        unknown_upper = sum(
            min(
                unknown_edges[parity],
                2 * remaining_ones[parity],
                2 * (unknown_counts[parity] - remaining_ones[parity]),
            )
            for parity in (0, 1)
        )
    else:
        even_ones, odd_ones = remaining_ones
        even_zeros = unknown_counts[0] - even_ones
        odd_zeros = unknown_counts[1] - odd_ones
        unknown_upper = min(
            sum(unknown_edges),
            2 * min(even_ones, odd_zeros)
            + 2 * min(even_zeros, odd_ones),
        )

    mismatch_min = known_mismatches + boundary_min
    mismatch_max = min(
        length,
        known_mismatches + boundary_max + unknown_upper,
    )
    return length - 2 * mismatch_max, length - 2 * mismatch_min


def _normalize_partial_sequence(sequence: Iterable[PartialBit]) -> Tuple[Optional[int], ...]:
    """Normalize known bits while preserving None as an unspecified bit."""
    try:
        values = tuple(sequence)
    except TypeError as error:
        raise ValueError("a partial binary sequence must be iterable") from error
    if not values:
        raise ValueError("a partial binary sequence must be non-empty")

    normalized = []
    for index, value in enumerate(values):
        if value is None:
            normalized.append(None)
        elif value in (0, "0") and not isinstance(value, bool):
            normalized.append(0)
        elif value in (1, "1") and not isinstance(value, bool):
            normalized.append(1)
        else:
            raise ValueError("partial sequence element at index {} is not binary or None".format(index))
    return tuple(normalized)
