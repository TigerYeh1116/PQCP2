"""Exact PQCP orbit keys under a specified PACF-preserving symmetry group.

Each row can independently rotate, reverse, or change all signs. Both rows
can undergo the SAME unit decimation i->q*i mod L, gcd(q,L)=1, and rows may
be exchanged. Independent row operations preserve each PACF; common
decimation permutes shifts and preserves the Project target's actual count
and magnitudes. Separate non-unit or separate-row decimations are NOT used.

The key classifies precisely this generated group, not all imaginable
equivalences. Equal PACFs/compressed patterns alone do not imply equivalence.
Canonicalization is only for result deduplication, never mathematical pruning.
"""

from functools import lru_cache
from math import gcd
from typing import Sequence, Tuple, Union

from .correlation import normalize_binary_sequence


@lru_cache(maxsize=8192)
def _row_key(word: str) -> int:
    L = len(word)
    mask = (1 << L) - 1
    best = mask
    for orientation in (word, word[::-1]):
        value = int(orientation, 2)
        for _ in range(L):
            best = min(best, value, mask ^ value)
            value = ((value << 1) & mask) | (value >> (L - 1))
    return best


@lru_cache(maxsize=8192)
def _pair_key(a: str, b: str) -> Tuple[int, int, int]:
    L = len(a)
    units = (1,) if L == 1 else (q for q in range(1, L) if gcd(q, L) == 1)
    best = None
    for q in units:
        aa = "".join(a[(q * i) % L] for i in range(L))
        bb = "".join(b[(q * i) % L] for i in range(L))
        candidate = tuple(sorted((_row_key(aa), _row_key(bb))))
        if best is None or candidate < best:
            best = candidate
    return L, best[0], best[1]


def pqcp_equivalence_key(
    a: Sequence[Union[int, str]], b: Sequence[Union[int, str]],
) -> Tuple[int, int, int]:
    """Return an exact canonical orbit key; input/output use Project bits.

    Validity as a PQCP is the independent verifier's responsibility. The
    length is part of the key and the shared q is enumerated exhaustively.
    Minimizing each row over dihedral/complement operations is valid because
    unit decimation normalizes these independent row transformations.
    """
    a, b = normalize_binary_sequence(a), normalize_binary_sequence(b)
    if len(a) != len(b):
        raise ValueError("equivalence requires equal sequence lengths")
    aa, bb = "".join(map(str, a)), "".join(map(str, b))
    return _pair_key(*sorted((aa, bb)))
