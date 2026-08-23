"""Data structures describing the Project 2 PQCP requirement.

This module deliberately contains no candidate-generation or search logic.
"""

from dataclasses import dataclass
from typing import Optional, Sequence


BinarySequence = Sequence[int]


@dataclass(frozen=True)
class PQCPInstance:
    """Parameters and optional candidate sequences for an ``(L, 4)``-PQCP.

    The Project 2, question 3 requirement is represented by the defaults:
    exactly two nonzero periodic autocorrelation sums, both of magnitude 4.
    ``a`` and ``b`` are optional so that the same object can describe a target
    length before a candidate has been produced.
    """

    L: int
    a: Optional[BinarySequence] = None
    b: Optional[BinarySequence] = None
    target_nonzero_magnitude: int = 4
    required_nonzero_sums: int = 2

    def __post_init__(self) -> None:
        """Validate problem parameters and any supplied sequence lengths."""
        if not isinstance(self.L, int) or isinstance(self.L, bool) or self.L <= 0:
            raise ValueError("L must be a positive integer")
        if self.target_nonzero_magnitude != 4:
            raise ValueError("Project 2 PQCP target nonzero magnitude must be 4")
        if self.required_nonzero_sums != 2:
            raise ValueError("Project 2 requires exactly two nonzero sums")
        if (self.a is None) != (self.b is None):
            raise ValueError("a and b must either both be supplied or both be omitted")
        if self.a is not None and (len(self.a) != self.L or len(self.b) != self.L):
            raise ValueError("supplied sequences must both have length L")


# ``Problem`` is a concise name for callers that only need the specification.
Problem = PQCPInstance
