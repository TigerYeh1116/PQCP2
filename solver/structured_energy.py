"""Exact sequence compression and multiscale residual energies for Project 2.

For a selected target case ``(k, eta)``, the target profile has value
``4*eta`` at shifts ``k`` and ``L-k`` and zero at every other sidelobe.
``full_error_energy`` sums squared residuals over the independent periodic
shifts ``1..L/2``.  It is zero exactly for that complete target.

Compressed residual energies group shifts modulo ``L/factor``.  They are
navigation heuristics only: cancellation inside a group means a compressed
energy can vanish away from a solution.  Consequently they are only added to
the full energy and are never used by the verifier or as hard pruning.
"""

from dataclasses import dataclass
from typing import Mapping, Sequence, Tuple

from .correlation import normalize_binary_sequence


@dataclass(frozen=True)
class StructuredEnergyBreakdown:
    """Exact fixed-target energy components, all scaled by ``1/16``.

    ``full`` is sufficient for the selected Project 2 target.  Entries in
    ``compressed`` are exact residual energies after genuine sequence
    compression, but are only necessary conditions on their own because
    residuals in one compression class may cancel.
    """

    full: int
    compressed: Tuple[Tuple[int, int], ...]

    def component(self, factor: int) -> int:
        """Return one compressed component, raising for an absent factor."""
        for recorded_factor, energy in self.compressed:
            if recorded_factor == factor:
                return energy
        raise KeyError(factor)

    def weighted_total(self, weights: Mapping[int, int]) -> int:
        """Return ``full + sum(weights[m] * E_m)`` exactly."""
        available = dict(self.compressed)
        total = self.full
        for factor, weight in weights.items():
            if not isinstance(weight, int) or isinstance(weight, bool) or weight < 0:
                raise ValueError("compression weights must be non-negative integers")
            if factor not in available:
                raise ValueError("missing compressed component for factor {}".format(factor))
            total += weight * available[factor]
        return total


def compress_binary_sequence(sequence: Sequence[int], factor: int) -> Tuple[int, ...]:
    """Return the genuine ``factor``-compression of a binary sign sequence.

    Project bits are converted by ``0 -> +1`` and ``1 -> -1``.  For
    ``L = factor * d``, the result is

    ``c[r] = sum(x[r + q*d] for q=0..factor-1)``.

    Its entries are integers in ``{-factor, -factor+2, ..., factor}``; this
    function performs no lossy thresholding or block approximation.
    """
    bits = normalize_binary_sequence(sequence)
    _validate_factor(len(bits), factor)
    compressed_length = len(bits) // factor
    signs = tuple(1 if bit == 0 else -1 for bit in bits)
    return tuple(
        sum(signs[r + q * compressed_length] for q in range(factor))
        for r in range(compressed_length)
    )


def integer_periodic_autocorrelation_profile(sequence: Sequence[int]) -> Tuple[int, ...]:
    """Return the periodic autocorrelation profile of an integer sequence."""
    values = tuple(sequence)
    if not values:
        raise ValueError("an integer sequence must be non-empty")
    if any(not isinstance(value, int) or isinstance(value, bool) for value in values):
        raise ValueError("an integer sequence must contain only integers")
    length = len(values)
    return tuple(
        sum(values[index] * values[(index + shift) % length] for index in range(length))
        for shift in range(length)
    )


def compressed_pair_profile(
    a: Sequence[int], b: Sequence[int], factor: int,
) -> Tuple[int, ...]:
    """Recompute the exact pair PACF after compressing A and B directly."""
    compressed_a = compress_binary_sequence(a, factor)
    compressed_b = compress_binary_sequence(b, factor)
    if len(compressed_a) != len(compressed_b):
        raise ValueError("a and b must have equal lengths")
    profile_a = integer_periodic_autocorrelation_profile(compressed_a)
    profile_b = integer_periodic_autocorrelation_profile(compressed_b)
    return tuple(left + right for left, right in zip(profile_a, profile_b))


def fold_profile(profile: Sequence[int], factor: int) -> Tuple[int, ...]:
    """Fold a length-L PACF profile exactly as sequence compression requires.

    The compression identity is

    ``PAF(compress(x,m))[r] = sum_q PAF(x)[r + q*(L/m)]``.

    It applies to the observed pair profile, the Project target, and their
    residual.  Folding the residual first automatically handles target shifts
    that collide with one another or with the compressed origin.
    """
    values = tuple(profile)
    if not values:
        raise ValueError("a profile must be non-empty")
    if any(not isinstance(value, int) or isinstance(value, bool) for value in values):
        raise ValueError("profile values must be integers")
    _validate_factor(len(values), factor)
    compressed_length = len(values) // factor
    return tuple(
        sum(values[r + q * compressed_length] for q in range(factor))
        for r in range(compressed_length)
    )


def target_profile(length: int, shift: int, eta: int) -> Tuple[int, ...]:
    """Return the exact Project 2 profile for one target position and sign."""
    _validate_target(length, shift, eta)
    values = [0] * length
    values[0] = 2 * length
    values[shift] = values[length - shift] = 4 * eta
    return tuple(values)


def residual_profile(profile: Sequence[int], shift: int, eta: int) -> Tuple[int, ...]:
    """Return exact integer residuals ``observed-target`` at every shift."""
    values = _validate_profile(profile)
    target = target_profile(len(values), shift, eta)
    return tuple(value - desired for value, desired in zip(values, target))


def full_error_energy(profile: Sequence[int], shift: int, eta: int) -> int:
    """Return squared residual energy over independent nonzero shifts."""
    residuals = residual_profile(profile, shift, eta)
    return sum(residuals[u] * residuals[u] for u in range(1, len(residuals) // 2 + 1))


def compressed_error_energy(
    profile: Sequence[int], shift: int, eta: int, factor: int,
) -> int:
    """Return exact residual energy after grouping shifts by a compression factor."""
    residuals = residual_profile(profile, shift, eta)
    return sum(value * value for value in fold_profile(residuals, factor))


def structured_energy_breakdown(
    profile: Sequence[int],
    shift: int,
    eta: int,
    factors: Sequence[int] = (2,),
) -> StructuredEnergyBreakdown:
    """Return full and compressed exact energies in Project correlation units.

    Binary-pair residuals are multiples of four, so division by 16 remains
    exact.  Duplicate factors are collapsed and the stored order is sorted to
    keep checkpoint/benchmark comparisons deterministic.
    """
    values = _validate_profile(profile)
    unique = tuple(sorted(set(factors)))
    for factor in unique:
        _validate_factor(len(values), factor)
    raw_full = full_error_energy(values, shift, eta)
    raw_compressed = tuple(
        (factor, compressed_error_energy(values, shift, eta, factor))
        for factor in unique
    )
    if raw_full % 16 or any(energy % 16 for _, energy in raw_compressed):
        raise ValueError(
            "Project binary-pair residual energies must be divisible by 16"
        )
    return StructuredEnergyBreakdown(
        full=raw_full // 16,
        compressed=tuple(
            (factor, energy // 16) for factor, energy in raw_compressed
        ),
    )


def multiscale_error_energy(
    profile: Sequence[int],
    shift: int,
    eta: int,
    compression_weights: Mapping[int, int],
) -> int:
    """Add weighted compressed energies to the full fixed-target energy.

    Weights are explicit experimental controls, not mathematical constants.
    The full component always has coefficient one, so total energy zero still
    implies the exact selected Project 2 target regardless of compression.
    """
    energy = full_error_energy(profile, shift, eta)
    for factor, weight in sorted(compression_weights.items()):
        if not isinstance(weight, int) or isinstance(weight, bool) or weight < 0:
            raise ValueError("compression weights must be non-negative integers")
        energy += weight * compressed_error_energy(profile, shift, eta, factor)
    return energy


def _validate_profile(profile: Sequence[int]) -> Tuple[int, ...]:
    values = tuple(profile)
    if len(values) < 4 or len(values) % 2:
        raise ValueError("structured target energy requires an even length at least four")
    if any(not isinstance(value, int) or isinstance(value, bool) for value in values):
        raise ValueError("profile values must be integers")
    return values


def _validate_target(length: int, shift: int, eta: int) -> None:
    if not isinstance(length, int) or isinstance(length, bool) or length < 4 or length % 2:
        raise ValueError("length must be an even integer at least four")
    if not isinstance(shift, int) or isinstance(shift, bool) or not 1 <= shift < length // 2:
        raise ValueError("shift must represent a two-element orbit below L/2")
    if eta not in (-1, 1) or isinstance(eta, bool):
        raise ValueError("eta must be -1 or +1")


def _validate_factor(length: int, factor: int) -> None:
    if not isinstance(factor, int) or isinstance(factor, bool) or factor < 2:
        raise ValueError("factor must be an integer at least two")
    if length % factor:
        raise ValueError("compression factor must divide the sequence/profile length")
