"""Pure-Python periodic autocorrelation functions for binary sequences."""

from typing import Iterable, List, Sequence, Tuple, Union


BinaryElement = Union[int, str]
BinarySequence = Sequence[BinaryElement]


def normalize_binary_sequence(sequence: Iterable[BinaryElement]) -> Tuple[int, ...]:
    """Return ``sequence`` as a tuple of bits, rejecting non-binary values.

    Integers ``0`` and ``1`` and string bits ``"0"`` and ``"1"`` are
    accepted.  Empty sequences are rejected because periodic correlation has
    no period for length zero.
    """
    try:
        values = tuple(sequence)
    except TypeError as error:
        raise ValueError("a binary sequence must be iterable") from error

    if not values:
        raise ValueError("a binary sequence must be non-empty")

    bits = []
    for index, value in enumerate(values):
        if value in (0, "0") and not isinstance(value, bool):
            bits.append(0)
        elif value in (1, "1") and not isinstance(value, bool):
            bits.append(1)
        else:
            raise ValueError("sequence element at index {} is not binary".format(index))
    return tuple(bits)


def periodic_autocorrelation(sequence: BinarySequence, shift: int) -> int:
    """Compute ``rho(sequence; shift)`` under the Project 2 periodic definition.

    For binary bits this is
    ``sum_i (-1) ** (a[i] + a[(i + shift) mod L])``.  Valid shifts are the
    canonical representatives ``0`` through ``L - 1``.
    """
    bits = normalize_binary_sequence(sequence)
    length = len(bits)
    if not isinstance(shift, int) or isinstance(shift, bool) or not 0 <= shift < length:
        raise ValueError("shift must be an integer in the range 0 through L - 1")

    return sum(1 if bits[index] == bits[(index + shift) % length] else -1
               for index in range(length))


def pair_autocorrelation_sum(a: BinarySequence, b: BinarySequence, shift: int) -> int:
    """Return ``rho(a; shift) + rho(b; shift)`` for equal-length binary inputs."""
    a_bits = normalize_binary_sequence(a)
    b_bits = normalize_binary_sequence(b)
    if len(a_bits) != len(b_bits):
        raise ValueError("a and b must have equal lengths")
    return periodic_autocorrelation(a_bits, shift) + periodic_autocorrelation(b_bits, shift)


def full_correlation_profile(a: BinarySequence, b: BinarySequence) -> List[int]:
    """Return pair autocorrelation sums ``S[0]`` through ``S[L - 1]``."""
    a_bits = normalize_binary_sequence(a)
    b_bits = normalize_binary_sequence(b)
    if len(a_bits) != len(b_bits):
        raise ValueError("a and b must have equal lengths")
    return [pair_autocorrelation_sum(a_bits, b_bits, shift)
            for shift in range(len(a_bits))]
