"""Exact integer objective for Project 2 (L, 4)-PQCP profiles.

Project 2 counts actual nonzero shift indices, not symmetry orbits.  Thus a
nonzero value at u and its required partner L-u counts as two nonzero sums.
The target is consequently two nonzero entries among S[1:] (each +4 or -4).
"""

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple


TARGET_NONZERO_MAGNITUDE = 4
REQUIRED_NONZERO_SHIFTS = 2
ALLOWED_NONZERO_SHIFT_VALUES = (-TARGET_NONZERO_MAGNITUDE, 0, TARGET_NONZERO_MAGNITUDE)


@dataclass(frozen=True)
class ObjectiveBreakdown:
    """Integer components whose sum is the exact deterministic objective."""

    total: int
    zero_shift_deviation: int
    nonzero_count_deviation: int
    origin_deviation: int
    symmetry_deviation: int
    nonzero_shifts: Tuple[int, ...]


def evaluate_objective(profile: Sequence[int]) -> ObjectiveBreakdown:
    """Measure a profile's exact integer distance from the Project 2 target.

    The score is the unweighted sum of four quantities: distance of every
    nonzero-shift value to {0, -4, +4}; absolute deviation from two nonzero
    shift indices; deviation of S[0] from 2L; and periodic-symmetry
    deviation.  No heuristic weights are used.  The score is zero if and
    only if the supplied profile obeys all stated target conditions.
    """
    values = _validate_profile(profile)
    length = len(values)
    nonzero_shifts = tuple(shift for shift in range(1, length) if values[shift] != 0)
    zero_shift_deviation = sum(
        min(abs(values[shift] - target) for target in ALLOWED_NONZERO_SHIFT_VALUES)
        for shift in range(1, length)
    )
    nonzero_count_deviation = abs(len(nonzero_shifts) - REQUIRED_NONZERO_SHIFTS)
    origin_deviation = abs(values[0] - 2 * length)
    symmetry_deviation = sum(
        abs(values[shift] - values[(-shift) % length])
        for shift in range(1, length)
    )
    total = (zero_shift_deviation + nonzero_count_deviation + origin_deviation
             + symmetry_deviation)
    return ObjectiveBreakdown(
        total=total,
        zero_shift_deviation=zero_shift_deviation,
        nonzero_count_deviation=nonzero_count_deviation,
        origin_deviation=origin_deviation,
        symmetry_deviation=symmetry_deviation,
        nonzero_shifts=nonzero_shifts,
    )


def pqcp_objective(profile: Sequence[int]) -> int:
    """Return only the exact Project 2 objective score for ``profile``."""
    return evaluate_objective(profile).total


def target_pair_squared_energy(profile: Sequence[int]) -> int:
    """Return exact squared error to the closest allowed nonzero shift pair.

    For a periodic-symmetric pair profile, choose one representative ``u``
    other than the even-length half shift and one target sign.  The full
    squared error is minimized at the sign of ``S[u]`` and simplifies to

    ``sum(S[v]**2 for v != 0) + 32 - 16*max(abs(S[u]))``.

    This deterministic integer energy is zero exactly at the Project 2 target
    and supplies a smoother SA acceptance landscape.  It does not replace
    :func:`pqcp_objective` for best-score reporting or verification.
    """
    values = _validate_profile(profile)
    representatives = range(1, (len(values) + 1) // 2)
    squared_sidelobes = sum(value * value for value in values[1:])
    maximum = max((abs(values[shift]) for shift in representatives), default=0)
    return squared_sidelobes + 32 - 16 * maximum


def target_profile_l1_distance(profile: Sequence[int]) -> int:
    """Return L1 distance to the closest *complete* legal target profile.

    Unlike :func:`pqcp_objective`, this metric does not independently allow
    every sidelobe to choose from ``{-4, 0, 4}``.  It first chooses exactly
    one periodic-symmetry orbit and one common sign, then measures all other
    shifts against zero.  Official even-length binary pair profiles have the
    relevant congruence in units of four; for a generic integer profile the
    result is conservatively rounded upward to the next such unit.

    This is a search/difficulty heuristic, not a safe Hamming bound or pruning
    rule.  It is zero exactly for a Project 2 target profile.
    """
    values = _validate_profile(profile)
    length = len(values)
    if length < 3:
        raise ValueError("a Project 2 target profile requires length at least 3")
    candidates = []
    for target_shift in range(1, (length + 1) // 2):
        if target_shift == length - target_shift:
            continue
        for target_value in (-TARGET_NONZERO_MAGNITUDE, TARGET_NONZERO_MAGNITUDE):
            distance = abs(values[0] - 2 * length)
            distance += sum(
                abs(values[shift] - (
                    target_value if shift in (target_shift, length - target_shift) else 0
                ))
                for shift in range(1, length)
            )
            candidates.append(distance)
    if not candidates:
        raise ValueError("profile has no eligible two-shift target orbit")
    minimum = min(candidates)
    return (minimum + 3) // 4


def pqcp_objective_breakdown(profile: Sequence[int]) -> Dict[str, int]:
    """Return named, JSON-safe components of :func:`pqcp_objective`.

    This is an observation helper: it uses exactly the same calculation as
    ``pqcp_objective`` and does not change its value or the search objective.
    ``target_value_distance`` is the sum of each nonzero-shift entry's
    distance to ``{-4, 0, 4}``; the remaining fields are the corresponding
    target-count, origin, and periodic-symmetry penalties.
    """
    result = evaluate_objective(profile)
    return {
        "target_value_distance": result.zero_shift_deviation,
        "nonzero_count_penalty": result.nonzero_count_deviation,
        "s0_penalty": result.origin_deviation,
        "symmetry_penalty": result.symmetry_deviation,
        "total": result.total,
    }


def _validate_profile(profile: Sequence[int]) -> Tuple[int, ...]:
    """Require a non-empty profile of ordinary integer values."""
    values = tuple(profile)
    if not values:
        raise ValueError("a correlation profile must be non-empty")
    if any(not isinstance(value, int) or isinstance(value, bool) for value in values):
        raise ValueError("a correlation profile must contain only integers")
    return values
