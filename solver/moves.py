"""Exact weight-preserving moves shared by PQCP search implementations."""

from dataclasses import dataclass
import random
from typing import Mapping, Optional, Sequence, Tuple

from .compression import CorrelationState


@dataclass(frozen=True)
class WeightPreservingSwap:
    """Flip one zero and one one in the same sequence.

    The two flips exchange their values, so ``wt(A)`` and ``wt(B)`` are both
    invariant.  ``sequence`` is lowercase ``"a"`` or ``"b"``.
    """

    sequence: str
    zero_position: int
    one_position: int

    def __post_init__(self) -> None:
        if self.sequence not in ("a", "b"):
            raise ValueError("sequence must be 'a' or 'b'")
        if self.zero_position == self.one_position:
            raise ValueError("weight-preserving swap positions must be distinct")


@dataclass(frozen=True)
class SwapEvaluation:
    """Exact objective values after a swap, computed without mutating state."""

    score: int
    target_pair_energy: int
    profile: Tuple[int, ...]


def choose_weight_preserving_swap(
    a: Sequence[int], b: Sequence[int], rng: random.Random
) -> Optional[WeightPreservingSwap]:
    """Choose a deterministic RNG-driven legal swap, or ``None`` if immobile."""
    moves = sample_weight_preserving_swaps(a, b, rng, 1)
    return moves[0] if moves else None


def sample_weight_preserving_swaps(
    a: Sequence[int],
    b: Sequence[int],
    rng: random.Random,
    count: int,
    same_parity: bool = False,
) -> Tuple[WeightPreservingSwap, ...]:
    """Sample legal swaps while scanning the current bits only once.

    Sampling is with replacement.  For ``count == 1`` the random draws and
    returned move exactly match :func:`choose_weight_preserving_swap`'s
    historical behavior.
    """
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        raise ValueError("count must be a positive integer")
    movable = []
    for name, values in (("a", tuple(a)), ("b", tuple(b))):
        parities = (0, 1) if same_parity else (None,)
        for parity in parities:
            zeros = tuple(
                index for index, bit in enumerate(values)
                if bit == 0 and (parity is None or index % 2 == parity)
            )
            ones = tuple(
                index for index, bit in enumerate(values)
                if bit == 1 and (parity is None or index % 2 == parity)
            )
            if zeros and ones:
                movable.append((name, zeros, ones))
    if not movable:
        return ()
    return tuple(
        WeightPreservingSwap(
            name,
            zeros[rng.randrange(len(zeros))],
            ones[rng.randrange(len(ones))],
        )
        for name, zeros, ones in (
            movable[rng.randrange(len(movable))] for _ in range(count)
        )
    )


def apply_weight_preserving_swap(
    correlation: CorrelationState, move: WeightPreservingSwap
) -> None:
    """Apply a legal swap through two exact incremental correlation updates."""
    values = correlation.a if move.sequence == "a" else correlation.b
    if values[move.zero_position] != 0 or values[move.one_position] != 1:
        raise ValueError("move positions are not currently a zero/one pair")
    flip = correlation._flip_a_unchecked if move.sequence == "a" else correlation._flip_b_unchecked
    flip(move.zero_position)
    flip(move.one_position)


def rollback_weight_preserving_swap(
    correlation: CorrelationState, move: WeightPreservingSwap
) -> None:
    """Reverse an applied swap while restoring the exact incremental profile."""
    values = correlation.a if move.sequence == "a" else correlation.b
    if values[move.zero_position] != 1 or values[move.one_position] != 0:
        raise ValueError("move is not currently in its applied state")
    flip = correlation._flip_a_unchecked if move.sequence == "a" else correlation._flip_b_unchecked
    flip(move.one_position)
    flip(move.zero_position)


def trial_weight_preserving_swap(
    correlation: CorrelationState, move: WeightPreservingSwap
) -> SwapEvaluation:
    """Evaluate one swap exactly in O(L), without apply/rollback mutation.

    For each representative shift, add the two ordinary one-bit deltas.  If
    the exchanged positions are adjacent at that shift, their shared directed
    correlation term was counted twice even though flipping both endpoints
    leaves it unchanged; ``4*x[p]*x[q]`` is therefore added per connection.
    Periodic symmetry supplies the representative multiplicity.
    """
    bits = correlation.a if move.sequence == "a" else correlation.b
    if bits[move.zero_position] != 0 or bits[move.one_position] != 1:
        raise ValueError("move positions are not currently a zero/one pair")
    signs = correlation._a_signs if move.sequence == "a" else correlation._b_signs
    position_zero, position_one = move.zero_position, move.one_position
    sign_zero, sign_one = signs[position_zero], signs[position_one]
    length = correlation.L
    target_distance = 0
    nonzero_count = 0
    squared_sidelobes = 0
    maximum_magnitude = 0
    proposed_profile = list(correlation._profile)
    for shift in range(1, length // 2 + 1):
        old_value = correlation._profile[shift]
        delta = -2 * sign_zero * (
            signs[(position_zero + shift) % length]
            + signs[(position_zero - shift) % length]
        )
        delta -= 2 * sign_one * (
            signs[(position_one + shift) % length]
            + signs[(position_one - shift) % length]
        )
        connections = (
            int((position_zero + shift) % length == position_one)
            + int((position_zero - shift) % length == position_one)
        )
        delta += 4 * sign_zero * sign_one * connections
        new_value = old_value + delta
        proposed_profile[shift] = new_value
        proposed_profile[-shift] = new_value
        multiplicity = 1 if shift == length - shift else 2
        target_distance += multiplicity * _target_distance(new_value)
        nonzero_count += multiplicity * int(new_value != 0)
        squared_sidelobes += multiplicity * new_value * new_value
        if shift < (length + 1) // 2:
            maximum_magnitude = max(maximum_magnitude, abs(new_value))
    return SwapEvaluation(
        score=target_distance + abs(nonzero_count - 2),
        target_pair_energy=squared_sidelobes + 32 - 16 * maximum_magnitude,
        profile=tuple(proposed_profile),
    )


def trial_fixed_target_swap_energy(
    correlation: CorrelationState,
    move: WeightPreservingSwap,
    target_shift: int,
    target_value: int,
) -> int:
    """Return exact fixed-target squared energy after ``move`` without mutation.

    This is the production fast path for ``fixed_target_full`` SA.  It uses
    the same algebraic two-flip autocorrelation delta as
    :func:`trial_weight_preserving_swap`, but accumulates only the quantity

    ``sum((S_new[u] - T[u])**2 for u=1..L/2) // 16``.

    Candidate ranking therefore needs one representative-shift pass instead
    of first constructing a full profile and then scanning it again.  The
    calculation is exact integer arithmetic and does not alter the target,
    move set, acceptance probability, or verifier.
    """
    length = correlation.L
    if (
        not isinstance(target_shift, int)
        or isinstance(target_shift, bool)
        or not 1 <= target_shift < length // 2
    ):
        raise ValueError("target_shift must be below the even-length half shift")
    if target_value not in (-4, 4) or isinstance(target_value, bool):
        raise ValueError("target_value must be -4 or +4")
    bits = correlation.a if move.sequence == "a" else correlation.b
    if bits[move.zero_position] != 0 or bits[move.one_position] != 1:
        raise ValueError("move positions are not currently a zero/one pair")
    signs = correlation._a_signs if move.sequence == "a" else correlation._b_signs
    position_zero, position_one = move.zero_position, move.one_position
    sign_zero, sign_one = signs[position_zero], signs[position_one]
    forward_distance = position_one - position_zero
    if forward_distance < 0:
        forward_distance += length
    backward_distance = length - forward_distance
    energy = 0
    for shift in range(1, length // 2 + 1):
        zero_forward = position_zero + shift
        if zero_forward >= length:
            zero_forward -= length
        zero_backward = position_zero - shift
        if zero_backward < 0:
            zero_backward += length
        one_forward = position_one + shift
        if one_forward >= length:
            one_forward -= length
        one_backward = position_one - shift
        if one_backward < 0:
            one_backward += length
        delta = -2 * sign_zero * (
            signs[zero_forward] + signs[zero_backward]
        )
        delta -= 2 * sign_one * (
            signs[one_forward] + signs[one_backward]
        )
        connections = ((shift == forward_distance) + (shift == backward_distance))
        delta += 4 * sign_zero * sign_one * connections
        desired = target_value if shift == target_shift else 0
        residual = correlation._profile[shift] + delta - desired
        energy += residual * residual
    return energy // 16


def trial_multiscale_fixed_target_swap_energy(
    correlation: CorrelationState,
    move: WeightPreservingSwap,
    target_shift: int,
    target_value: int,
    compression_weights: Mapping[int, int],
) -> int:
    """Return exact full-plus-compressed energy after ``move`` in one O(L) pass.

    For compression factor ``m`` and ``d=L/m``, the exact compressed residual
    bin is ``B_m[r] = sum_q residual[r+q*d]``.  This routine accumulates those
    bins while computing the ordinary two-flip correlation deltas, avoiding
    the old temporary full-profile construction and subsequent scans.

    The returned integer is

    ``E_full/16 + sum(weight[m] * E_m/16)``.

    Compression weights remain explicit navigation heuristics; keeping the
    full component guarantees that energy zero still means the exact selected
    Project target.
    """
    factors = []
    length = correlation.L
    for factor, weight in sorted(compression_weights.items()):
        if not isinstance(factor, int) or isinstance(factor, bool) or factor < 2:
            raise ValueError("compression factors must be integers at least two")
        if length % factor:
            raise ValueError("compression factor must divide the sequence length")
        if not isinstance(weight, int) or isinstance(weight, bool) or weight < 0:
            raise ValueError("compression weights must be non-negative integers")
        if weight:
            factors.append(factor)
    key = trial_fixed_target_swap_energy_key(
        correlation, move, target_shift, target_value, factors,
    )
    return key[0] + sum(
        compression_weights[factor] * energy
        for factor, energy in zip(factors, key[1:])
    )


def trial_fixed_target_swap_energy_key(
    correlation: CorrelationState,
    move: WeightPreservingSwap,
    target_shift: int,
    target_value: int,
    compression_factors: Sequence[int],
) -> Tuple[int, ...]:
    """Return ``(E_full, E_factor...)`` after a swap in one exact O(L) pass.

    The first component is always the complete fixed-target squared residual
    divided by 16.  Subsequent components follow the caller's sorted unique
    factor list.  This exact tuple lets seed/proposal ranking use compression
    as a lexicographic precision tie-break without assigning an arbitrary
    numerical weight.
    """
    length = correlation.L
    if (
        not isinstance(target_shift, int)
        or isinstance(target_shift, bool)
        or not 1 <= target_shift < length // 2
    ):
        raise ValueError("target_shift must be below the even-length half shift")
    if target_value not in (-4, 4) or isinstance(target_value, bool):
        raise ValueError("target_value must be -4 or +4")
    factors = tuple(compression_factors)
    if factors != tuple(sorted(set(factors))):
        raise ValueError("compression factors must be sorted and unique")
    factor_data = []
    for factor in factors:
        if not isinstance(factor, int) or isinstance(factor, bool) or factor < 2:
            raise ValueError("compression factors must be integers at least two")
        if length % factor:
            raise ValueError("compression factor must divide the sequence length")
        compressed_length = length // factor
        factor_data.append((compressed_length, [0] * compressed_length))

    bits = correlation.a if move.sequence == "a" else correlation.b
    if bits[move.zero_position] != 0 or bits[move.one_position] != 1:
        raise ValueError("move positions are not currently a zero/one pair")
    signs = correlation._a_signs if move.sequence == "a" else correlation._b_signs
    position_zero, position_one = move.zero_position, move.one_position
    sign_zero, sign_one = signs[position_zero], signs[position_one]
    forward_distance = (position_one - position_zero) % length
    backward_distance = length - forward_distance
    full_energy = 0
    for shift in range(1, length // 2 + 1):
        zero_forward = position_zero + shift
        if zero_forward >= length:
            zero_forward -= length
        zero_backward = position_zero - shift
        if zero_backward < 0:
            zero_backward += length
        one_forward = position_one + shift
        if one_forward >= length:
            one_forward -= length
        one_backward = position_one - shift
        if one_backward < 0:
            one_backward += length
        delta = -2 * sign_zero * (signs[zero_forward] + signs[zero_backward])
        delta -= 2 * sign_one * (signs[one_forward] + signs[one_backward])
        connections = ((shift == forward_distance) + (shift == backward_distance))
        delta += 4 * sign_zero * sign_one * connections
        desired = target_value if shift == target_shift else 0
        residual = correlation._profile[shift] + delta - desired
        full_energy += residual * residual
        mirror = length - shift
        for compressed_length, bins in factor_data:
            bins[shift % compressed_length] += residual
            if mirror != shift:
                bins[mirror % compressed_length] += residual

    return (full_energy // 16,) + tuple(
        sum(value * value for value in bins) // 16
        for _, bins in factor_data
    )


def hamming_weights(a: Sequence[int], b: Sequence[int]) -> Tuple[int, int]:
    """Return the pair of binary Hamming weights used by invariant checks."""
    return sum(a), sum(b)


def _target_distance(value: int) -> int:
    magnitude = abs(value)
    return min(magnitude, abs(magnitude - 4))
