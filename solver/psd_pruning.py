"""Certified all-frequency PSD necessary-condition pruning.

For a fixed Project 2 target ``T`` and sign sequences ``a,b``, the periodic
Wiener--Khinchin identity gives, at every DFT frequency ``f``,

``PSD_a(f) + PSD_b(f) = DFT(T)(f)``.

Both individual power spectral densities are non-negative.  Consequently a
complete sequence ``x`` cannot be either member of a solution for this target
when ``PSD_x(f) > DFT(T)(f)`` at even one frequency.

The profiles supplied here contain exact integers.  Frequencies whose cosine
characters are rational (0, L/2 and L/4 when present) are evaluated with
integer arithmetic.  Other characters are evaluated with outward-rounded
``mpmath.iv`` interval arithmetic at increasing precision.  A candidate is
pruned *only* when the whole certified interval is strictly negative for
``DFT(T)-PSD_x``.  Equality or an interval that still overlaps zero is
reported as unresolved and is retained.  Ordinary floating-point values are
therefore never used to make a hard-pruning decision.

This is an individual-candidate necessary condition, not a complete PQCP
test.  Failure to prune does not imply that a compatible partner exists.
"""

from dataclasses import dataclass
from functools import lru_cache
from threading import RLock
from typing import Optional, Sequence, Tuple

from .correlation import normalize_binary_sequence
from .structured_energy import integer_periodic_autocorrelation_profile, target_profile
from .target_profiles import TargetContentProfile


_DEFAULT_PRECISIONS = (30, 60, 120)
_INTERVAL_LOCK = RLock()


@dataclass(frozen=True)
class PSDPruningResult:
    """Auditable result of one certified individual PSD-cap test.

    ``should_prune`` is true exactly when ``violating_frequencies`` is
    nonempty.  ``unresolved_frequencies`` were too close to zero for the
    configured interval precisions and were conservatively retained.
    """

    should_prune: bool
    violating_frequencies: Tuple[int, ...]
    unresolved_frequencies: Tuple[int, ...]
    certified_frequencies: Tuple[int, ...]
    max_precision_dps: int


def certified_psd_pruning_from_profiles(
    individual_pacf: Sequence[int],
    target_pair_pacf: Sequence[int],
    *,
    frequencies: Optional[Sequence[int]] = None,
    precisions: Sequence[int] = _DEFAULT_PRECISIONS,
    stop_at_first_violation: bool = False,
) -> PSDPruningResult:
    """Prove whether an individual PACF exceeds a pair target PSD.

    Both inputs must be complete, symmetric, integer periodic-correlation
    profiles of the same length.  By default only the independent real DFT
    frequencies ``0..floor(L/2)`` are tested.  Supplying ``frequencies`` is
    useful for diagnostics; conjugate frequencies carry the same information.

    The returned hard-pruning verdict is safe: ``True`` proves that this fixed
    individual sequence cannot occur on either side of a pair realizing the
    supplied target profile.  ``False`` may include unresolved boundary cases.
    ``stop_at_first_violation`` reduces production filtering cost without
    changing the verdict; its diagnostic frequency lists are then prefixes.
    """
    individual = _validate_symmetric_integer_profile(individual_pacf, "individual_pacf")
    target = _validate_symmetric_integer_profile(target_pair_pacf, "target_pair_pacf")
    if len(individual) != len(target):
        raise ValueError("individual and target profiles must have equal lengths")
    length = len(individual)
    selected = _normalize_frequencies(length, frequencies)
    precision_steps = _normalize_precisions(precisions)
    if not isinstance(stop_at_first_violation, bool):
        raise ValueError("stop_at_first_violation must be boolean")
    difference = tuple(wanted - observed for wanted, observed in zip(target, individual))

    violating = []
    unresolved = []
    certified = []
    for frequency in selected:
        exact = _exact_rational_character_value(difference, frequency)
        if exact is not None:
            (violating if exact < 0 else certified).append(frequency)
            if exact < 0 and stop_at_first_violation:
                break
            continue

        verdict = None
        for dps in precision_steps:
            sign = _certified_interval_character_sign(
                difference, frequency, dps
            )
            if sign < 0:
                verdict = False
                break
            if sign > 0:
                verdict = True
                break
        if verdict is False:
            violating.append(frequency)
            if stop_at_first_violation:
                break
        elif verdict is True:
            certified.append(frequency)
        else:
            # An exact zero often remains an interval containing zero because
            # separate irrational terms lose algebraic cancellation.  Keeping
            # it is required for safe pruning.
            unresolved.append(frequency)

    return PSDPruningResult(
        should_prune=bool(violating),
        violating_frequencies=tuple(violating),
        unresolved_frequencies=tuple(unresolved),
        certified_frequencies=tuple(certified),
        max_precision_dps=precision_steps[-1],
    )


def certified_binary_psd_pruning(
    sequence: Sequence[int],
    target: TargetContentProfile,
    *,
    precisions: Sequence[int] = _DEFAULT_PRECISIONS,
    stop_at_first_violation: bool = False,
) -> PSDPruningResult:
    """Apply the certified all-frequency PSD cap to one Project binary word."""
    bits = normalize_binary_sequence(sequence)
    if not isinstance(target, TargetContentProfile) or target.L != len(bits):
        raise ValueError("target must be a matching TargetContentProfile")
    if not isinstance(stop_at_first_violation, bool):
        raise ValueError("stop_at_first_violation must be boolean")
    desired = target_profile(target.L, target.k, target.eta)
    precision_values = tuple(precisions)
    if precision_values == _DEFAULT_PRECISIONS and not stop_at_first_violation:
        return _cached_binary_psd_pruning(bits, target)
    signs = tuple(1 if bit == 0 else -1 for bit in bits)
    individual = integer_periodic_autocorrelation_profile(signs)
    return certified_psd_pruning_from_profiles(
        individual, desired, precisions=precision_values,
        stop_at_first_violation=stop_at_first_violation,
    )


@lru_cache(maxsize=8192)
def _cached_binary_psd_pruning(
    bits: Tuple[int, ...], target: TargetContentProfile,
) -> PSDPruningResult:
    signs = tuple(1 if bit == 0 else -1 for bit in bits)
    return certified_psd_pruning_from_profiles(
        integer_periodic_autocorrelation_profile(signs),
        target_profile(target.L, target.k, target.eta),
    )


def _validate_symmetric_integer_profile(
    profile: Sequence[int], name: str,
) -> Tuple[int, ...]:
    values = tuple(profile)
    if not values:
        raise ValueError(name + " must be nonempty")
    if any(not isinstance(value, int) or isinstance(value, bool) for value in values):
        raise ValueError(name + " must contain only integers")
    if any(values[shift] != values[-shift] for shift in range(1, len(values))):
        raise ValueError(name + " must satisfy periodic symmetry")
    return values


def _normalize_frequencies(
    length: int, frequencies: Optional[Sequence[int]],
) -> Tuple[int, ...]:
    if frequencies is None:
        return tuple(range(length // 2 + 1))
    try:
        selected = tuple(sorted(set(frequencies)))
    except TypeError as error:
        raise ValueError("frequencies must be an iterable of integers") from error
    if any(
        not isinstance(value, int) or isinstance(value, bool)
        or not 0 <= value <= length // 2
        for value in selected
    ):
        raise ValueError("frequencies must lie in 0..floor(L/2)")
    return selected


def _normalize_precisions(precisions: Sequence[int]) -> Tuple[int, ...]:
    try:
        values = tuple(sorted(set(precisions)))
    except TypeError as error:
        raise ValueError("precisions must be an iterable of integers") from error
    if not values or any(
        not isinstance(value, int) or isinstance(value, bool) or value < 15
        for value in values
    ):
        raise ValueError("precisions must be integers of at least 15 decimal digits")
    return values


def _exact_rational_character_value(
    profile: Tuple[int, ...], frequency: int,
) -> Optional[int]:
    """Evaluate the rational DFT characters exactly, else return ``None``."""
    length = len(profile)
    if frequency == 0:
        return sum(profile)
    if length % 2 == 0 and frequency == length // 2:
        return sum(value if shift % 2 == 0 else -value
                   for shift, value in enumerate(profile))
    if length % 4 == 0 and frequency == length // 4:
        # cos(pi*u/2) cycles 1,0,-1,0.  Symmetry guarantees the sine
        # contribution is zero.
        return sum(
            value * (1 if shift % 4 == 0 else -1 if shift % 4 == 2 else 0)
            for shift, value in enumerate(profile)
        )
    return None


def _certified_interval_character_sign(
    profile: Tuple[int, ...], frequency: int, dps: int,
) -> int:
    """Return -1/+1 only for a certified sign, or zero when unresolved."""
    try:
        from mpmath import iv
    except ImportError as error:  # pragma: no cover - torch installs mpmath here
        raise RuntimeError(
            "general-frequency certified PSD pruning requires mpmath"
        ) from error

    length = len(profile)
    with _INTERVAL_LOCK:
        previous = iv.dps
        try:
            iv.dps = dps
            cosines = _cosine_row(length, frequency, dps)
            value = iv.mpf(profile[0])
            for shift in range(1, (length - 1) // 2 + 1):
                value += 2 * profile[shift] * cosines[shift]
            if length % 2 == 0:
                value += profile[length // 2] * (1 if frequency % 2 == 0 else -1)
            if value < 0:
                return -1
            if value >= 0:
                return 1
            return 0
        finally:
            iv.dps = previous


@lru_cache(maxsize=4096)
def _cosine_row(length: int, frequency: int, dps: int):
    """Cache outward-rounded cosine intervals at the requested precision."""
    from mpmath import iv

    # Caller holds _INTERVAL_LOCK and has already selected iv.dps.
    return tuple(
        iv.cos(2 * iv.pi * frequency * shift / length)
        for shift in range((length - 1) // 2 + 1)
    )


__all__ = (
    "PSDPruningResult",
    "certified_binary_psd_pruning",
    "certified_psd_pruning_from_profiles",
)
