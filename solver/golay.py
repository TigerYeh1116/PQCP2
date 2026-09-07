"""Binary Golay, periodic-Golay, and length-adapted Golay seed helpers.

The Rudin--Shapiro recursion starts from ``(+1),(+1)`` and maps a Golay
pair ``(A,B)`` to ``(A||B, A||(-B))``.  Binary bit 0 represents sign +1 and
bit 1 represents sign -1.  The resulting lengths are powers of two.

The Project 2 lengths are not ordinary binary Golay lengths.  For 58, 68,
and 90 this module uses published *periodic* Golay representatives, which
obey exactly the same periodic-correlation definition used by Project 2.
For 44, 46, 86, and 94 an exact periodic Golay pair is impossible because
``2L`` is not a sum of two integer squares.  Those lengths instead use the
low-cost Turyn construction from a short binary GCP and a Legendre perfect
sequence pair.  The result is explicitly labelled ``turyn-near-periodic``;
it is never reported as an exact GCP.

Sources:

* T. Lumsden, I. Kotsireas, C. Bright, "New Results on Periodic Golay
  Pairs", DOI 10.1090/mcom/4096 (the length-90 representatives).
* T. Lumsden, "Enumerations for periodic Golay pairs up to lengths 72",
  DOI 10.5281/zenodo.12792345 (the length-58 and length-68 representatives).
* A. R. Adhikary et al., "Optimal Binary Periodic Almost-Complementary
  Pairs", DOI 10.1109/LSP.2016.2600586 (Turyn Construction 3).

An aperiodic Golay pair has zero periodic pair sidelobes, hence Project
objective score 2 rather than 0.  ``complete_one_flip_each`` tests the exact
neighborhood formed by flipping one A bit and one B bit.  Its correlation
deltas are computed algebraically and every returned candidate is checked by
the independent verifier.
"""

from dataclasses import dataclass
import random
from typing import Optional, Sequence, Tuple

from .correlation import normalize_binary_sequence
from .verifier import verify_pqcp


BinaryPair = Tuple[Tuple[int, ...], Tuple[int, ...]]


PROJECT_GOLAY_LENGTHS = frozenset((44, 46, 58, 68, 86, 90, 94))

# One published representative is sufficient: inexpensive autocorrelation-
# preserving equivalences generate deterministic seed-dependent variants.
# The strings use this project's convention 0 -> +1 and 1 -> -1.
_PUBLISHED_PERIODIC_PAIRS = {
    58: (
        "1111111111100010001011000100011110000010100100101000110000",
        "1110000101100110010000010100110010100010011000101101001010",
    ),
    68: (
        "11111111111111110010001011010110001100010111010010010111010101001000",
        "11110111001001100110110000111010100111001111010001111010011010110000",
    ),
    90: (
        "111100101011001001011000001000100011100100011001111100001100010110011010001100101011001011",
        "111011000000001111011001111000000001100101010010100001010010000110000101000101111101101010",
    ),
}

# target length -> (ordinary GCP length, Legendre perfect-sequence length)
_TURYN_PROJECT_FACTORS = {
    44: (4, 11),
    46: (2, 23),
    86: (2, 43),
    94: (2, 47),
}


@dataclass(frozen=True)
class GolaySeed:
    """A length-specific binary seed with an honest construction label.

    ``kind == "periodic"`` means all nonzero periodic pair sidelobes are
    exactly zero.  ``kind == "turyn-near-periodic"`` means a short GCP was
    applied by Turyn's product, but the result is not falsely advertised as
    a periodic Golay pair.
    """

    a: Tuple[int, ...]
    b: Tuple[int, ...]
    kind: str
    source: str


def rudin_shapiro_golay_pair(length: int) -> BinaryPair:
    """Construct the standard binary Golay pair for a power-of-two length."""
    if not isinstance(length, int) or isinstance(length, bool) or length <= 0:
        raise ValueError("length must be a positive integer")
    if length & (length - 1):
        raise ValueError("Rudin-Shapiro construction requires a power-of-two length")
    a_signs = b_signs = (1,)
    while len(a_signs) < length:
        a_signs, b_signs = (
            a_signs + b_signs,
            a_signs + tuple(-value for value in b_signs),
        )
    return (_signs_to_bits(a_signs), _signs_to_bits(b_signs))


def is_golay_complementary_pair(a: Sequence[int], b: Sequence[int]) -> bool:
    """Check the defining zero aperiodic-correlation sum at every nonzero lag."""
    a_bits = normalize_binary_sequence(a)
    b_bits = normalize_binary_sequence(b)
    if len(a_bits) != len(b_bits):
        return False
    a_signs = tuple(1 if bit == 0 else -1 for bit in a_bits)
    b_signs = tuple(1 if bit == 0 else -1 for bit in b_bits)
    length = len(a_signs)
    return all(
        sum(
            a_signs[index] * a_signs[index + shift]
            + b_signs[index] * b_signs[index + shift]
            for index in range(length - shift)
        ) == 0
        for shift in range(1, length)
    )


def is_periodic_golay_pair(a: Sequence[int], b: Sequence[int]) -> bool:
    """Check exact zero periodic-autocorrelation sums at every nonzero shift."""
    a_bits = normalize_binary_sequence(a)
    b_bits = normalize_binary_sequence(b)
    if len(a_bits) != len(b_bits):
        return False
    a_signs = tuple(1 if bit == 0 else -1 for bit in a_bits)
    b_signs = tuple(1 if bit == 0 else -1 for bit in b_bits)
    length = len(a_signs)
    return all(
        sum(
            a_signs[index] * a_signs[(index + shift) % length]
            + b_signs[index] * b_signs[(index + shift) % length]
            for index in range(length)
        ) == 0
        for shift in range(1, length)
    )


def project_length_golay_seed(length: int, seed: int = 0) -> GolaySeed:
    """Construct a cheap Golay-derived seed for every requested Project 2 L.

    For lengths 58, 68, and 90, the returned pair is a published exact
    periodic Golay pair.  For 44, 46, 86, and 94, the sum-of-two-squares
    identity rules out an exact periodic pair, so a deterministic Turyn
    GCP/Legendre near-periodic construction is returned instead.

    ``seed`` selects only autocorrelation-preserving equivalences (independent
    cyclic shifts, reversal, complementation, and optional A/B swap).  It
    therefore gives multiple reproducible SA centres without changing the
    exact correlation profile or doing an expensive search.
    """
    if not isinstance(length, int) or isinstance(length, bool) or length <= 0:
        raise ValueError("length must be a positive integer")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    if length in _PUBLISHED_PERIODIC_PAIRS:
        encoded_a, encoded_b = _PUBLISHED_PERIODIC_PAIRS[length]
        pair = (_bit_string(encoded_a), _bit_string(encoded_b))
        if not is_periodic_golay_pair(*pair):
            raise RuntimeError("embedded published periodic Golay pair failed verification")
        pair = _equivalent_pair(*pair, seed=seed)
        return GolaySeed(
            pair[0], pair[1], "periodic",
            "published periodic Golay representative",
        )
    if length in _TURYN_PROJECT_FACTORS:
        gcp_length, prime = _TURYN_PROJECT_FACTORS[length]
        gcp_a, gcp_b = rudin_shapiro_golay_pair(gcp_length)
        base = _legendre_perfect_sequence(prime)
        pair = _turyn_product(base, base, gcp_a, gcp_b)
        if len(pair[0]) != length:
            raise RuntimeError("Turyn construction produced the wrong length")
        pair = _equivalent_pair(*pair, seed=seed)
        return GolaySeed(
            pair[0], pair[1], "turyn-near-periodic",
            "Turyn product of a short binary GCP and a Legendre pair",
        )
    if length & (length - 1) == 0:
        pair = _equivalent_pair(*rudin_shapiro_golay_pair(length), seed=seed)
        return GolaySeed(pair[0], pair[1], "aperiodic", "Rudin-Shapiro binary GCP")
    raise ValueError(
        "no verified low-cost Golay construction is registered for length {}".format(length)
    )


def complete_one_flip_each(length: int, seed: int = 0) -> Optional[BinaryPair]:
    """Return a verified PQCP one A-flip and one B-flip from the standard GCP.

    ``seed`` changes only the deterministic cyclic order in which positions
    are examined.  It does not change the Golay construction or mathematical
    constraints.  ``None`` means this exact finite neighborhood contained no
    solution; it is not a global nonexistence result.
    """
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    a, b = rudin_shapiro_golay_pair(length)
    deltas_a = _one_flip_periodic_deltas(a)
    deltas_b = _one_flip_periodic_deltas(b)
    for a_offset in range(length):
        position_a = (seed + a_offset) % length
        for b_offset in range(length):
            position_b = (seed // length + b_offset) % length
            nonzero_count = 0
            admissible = True
            for shift in range(1, length):
                value = deltas_a[position_a][shift] + deltas_b[position_b][shift]
                if value != 0:
                    nonzero_count += 1
                    if abs(value) != 4 or nonzero_count > 2:
                        admissible = False
                        break
            if not admissible or nonzero_count != 2:
                continue
            candidate_a = _flipped(a, position_a)
            candidate_b = _flipped(b, position_b)
            if not verify_pqcp(candidate_a, candidate_b).is_valid:
                raise RuntimeError("Golay delta completion disagrees with independent verifier")
            return candidate_a, candidate_b
    return None


def _one_flip_periodic_deltas(bits: Sequence[int]) -> Tuple[Tuple[int, ...], ...]:
    """Return exact periodic-autocorrelation deltas for flipping each position."""
    signs = tuple(1 if bit == 0 else -1 for bit in bits)
    length = len(signs)
    rows = []
    for position in range(length):
        delta = [0] * length
        for shift in range(1, length // 2 + 1):
            value = -2 * signs[position] * (
                signs[(position + shift) % length]
                + signs[(position - shift) % length]
            )
            delta[shift] = value
            delta[length - shift] = value
        rows.append(tuple(delta))
    return tuple(rows)


def _flipped(bits: Sequence[int], position: int) -> Tuple[int, ...]:
    result = list(bits)
    result[position] ^= 1
    return tuple(result)


def _signs_to_bits(signs: Sequence[int]) -> Tuple[int, ...]:
    return tuple(0 if value == 1 else 1 for value in signs)


def _bits_to_signs(bits: Sequence[int]) -> Tuple[int, ...]:
    return tuple(1 if bit == 0 else -1 for bit in normalize_binary_sequence(bits))


def _bit_string(value: str) -> Tuple[int, ...]:
    if not value or any(character not in "01" for character in value):
        raise RuntimeError("embedded Golay data must be a nonempty binary string")
    return tuple(int(character) for character in value)


def _legendre_perfect_sequence(prime: int) -> Tuple[int, ...]:
    """Return the +/-1 Legendre sequence for a prime ``p == 3 (mod 4)``.

    Its periodic autocorrelation is -1 at every nonzero shift.  All primes
    used here are tiny fixed construction parameters, so trial division is
    deliberately clearer and cheaper than adding a number-theory dependency.
    """
    if prime % 4 != 3 or prime < 3 or any(prime % divisor == 0 for divisor in range(2, int(prime ** 0.5) + 1)):
        raise ValueError("Legendre perfect construction requires a prime congruent to 3 modulo 4")
    residues = {value * value % prime for value in range(1, prime)}
    return tuple(1 if index == 0 or index in residues else -1 for index in range(prime))


def _turyn_product(
    a_signs: Sequence[int],
    b_signs: Sequence[int],
    c_bits: Sequence[int],
    d_bits: Sequence[int],
) -> BinaryPair:
    """Apply the binary Turyn product in Adhikary et al., Construction 3."""
    a, b = tuple(a_signs), tuple(b_signs)
    c, d = _bits_to_signs(c_bits), _bits_to_signs(d_bits)
    if len(a) != len(b) or len(c) != len(d):
        raise ValueError("both input pairs must have equal constituent lengths")
    if any(value not in (-1, 1) for value in a + b):
        raise ValueError("Turyn base sequences must contain only -1/+1")
    plus = tuple((left + right) // 2 for left, right in zip(c, d))
    minus = tuple((left - right) // 2 for left, right in zip(c, d))
    e_signs = tuple(
        a_value * plus_value + b_value * minus_value
        for a_value, b_value in zip(a, b)
        for plus_value, minus_value in zip(plus, minus)
    )
    f_signs = tuple(
        b_value * plus_value - a_value * minus_value
        for a_value, b_value in zip(a, b)
        for plus_value, minus_value in zip(plus, minus)
    )
    if any(value not in (-1, 1) for value in e_signs + f_signs):
        raise RuntimeError("Turyn product did not remain binary")
    return _signs_to_bits(e_signs), _signs_to_bits(f_signs)


def _equivalent_pair(a: Sequence[int], b: Sequence[int], seed: int) -> BinaryPair:
    """Choose a deterministic periodic-autocorrelation-preserving equivalent pair."""
    rng = random.Random(seed)

    def transform(bits: Sequence[int]) -> Tuple[int, ...]:
        values = tuple(bits)
        if rng.randrange(2):
            values = tuple(reversed(values))
        if rng.randrange(2):
            values = tuple(bit ^ 1 for bit in values)
        shift = rng.randrange(len(values))
        return values[shift:] + values[:shift]

    transformed_a, transformed_b = transform(a), transform(b)
    if rng.randrange(2):
        transformed_a, transformed_b = transformed_b, transformed_a
    return transformed_a, transformed_b
