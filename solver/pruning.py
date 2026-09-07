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


def prune_partial_assignment(a: PartialSequence, b: PartialSequence) -> PruningResult:
    """Safely reject only partial pairs whose completions cannot meet the target.

    For each nonzero shift, its conservative interval must contain at least
    one allowed final value in {-4, 0, +4}.  Also, if zero lies outside an
    interval, that actual shift is forced nonzero; more than two forced
    nonzero indices cannot be repaired by any completion.  These tests are
    necessary conditions, not objective-based heuristics.
    """
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
