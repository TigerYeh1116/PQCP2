"""FKM generation of binary cyclic representatives (binary necklaces).

This module intentionally knows nothing about PQCP correlation or validity.
It supplies one representative from each cyclic orbit for later candidate
evaluation stages.
"""

from typing import Iterator, Optional, Sequence, Tuple


BinaryWord = Tuple[int, ...]


def is_cyclic_representative(sequence: Sequence[int]) -> bool:
    """Return whether ``sequence`` is its cyclic orbit's lexicographic minimum.

    This helper is intended for small-scale validation and tests.  The FKM
    generator itself does not construct all rotations.
    """
    word = tuple(sequence)
    if not word:
        raise ValueError("a cyclic representative must have positive length")
    if any(bit not in (0, 1) or isinstance(bit, bool) for bit in word):
        raise ValueError("a cyclic representative must contain only integer bits")
    return word == min(word[offset:] + word[:offset] for offset in range(len(word)))


def generate_fkm_sequences(
    L: int,
    weight: Optional[int] = None,
    limit: Optional[int] = None,
) -> Iterator[BinaryWord]:
    """Yield binary cyclic representatives of length ``L`` using the FKM rule.

    Each yielded tuple is the lexicographically smallest rotation in exactly
    one binary cyclic orbit (a binary necklace).  If ``weight`` is supplied,
    only representatives with that number of ones are yielded.  ``limit``
    caps the number of yielded candidates without materialising a candidate
    list.

    The implementation is the Fredricksen--Kessler--Maiorana recursive
    necklace-generation recurrence.  It performs no correlation calculation,
    PQCP verification, or search ranking.
    """
    _validate_parameters(L, weight, limit)
    return _generate_necklaces(L, weight, limit)


def _validate_parameters(L: int, weight: Optional[int], limit: Optional[int]) -> None:
    """Validate public FKM generator parameters."""
    if not isinstance(L, int) or isinstance(L, bool) or L <= 0:
        raise ValueError("L must be a positive integer")
    if weight is not None:
        if not isinstance(weight, int) or isinstance(weight, bool) or not 0 <= weight <= L:
            raise ValueError("weight must be an integer in the range 0 through L")
    if limit is not None:
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0:
            raise ValueError("limit must be a non-negative integer or None")


def _generate_necklaces(
    length: int,
    weight: Optional[int],
    limit: Optional[int],
) -> Iterator[BinaryWord]:
    """Implement the FKM recurrence while retaining only one working word."""
    word = [0] * (length + 1)  # FKM is conventionally indexed from 1.
    yielded = 0

    def visit(position: int, period: int) -> Iterator[BinaryWord]:
        nonlocal yielded
        if limit is not None and yielded >= limit:
            return
        if position > length:
            if length % period == 0:
                candidate = tuple(word[1:])
                if weight is None or sum(candidate) == weight:
                    yielded += 1
                    yield candidate
            return

        word[position] = word[position - period]
        yield from visit(position + 1, period)
        if limit is not None and yielded >= limit:
            return

        # The FKM alphabet loop is ``a[position - period] + 1 .. k - 1``.
        # For a binary alphabet, the second branch exists only after a zero.
        if word[position - period] == 0:
            word[position] = 1
            yield from visit(position + 1, position)

    yield from visit(1, 1)
