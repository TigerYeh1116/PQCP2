"""Exact incremental periodic-correlation evaluation for bit-flip searches.

The state stores binary sequences as signs (+1 for bit 0 and -1 for bit 1)
and retains their pair autocorrelation profile.  This is a compact reusable
representation, not a mathematical block-compression transformation.
"""

from typing import List, Sequence, Tuple, Union

from .correlation import normalize_binary_sequence


BinaryInput = Sequence[Union[int, str]]


class CorrelationState:
    """Maintain an exact pair-correlation profile under individual bit flips.

    For a sign sequence ``x[i] = (-1) ** a[i]``, a nonzero-shift term is
    ``x[i] * x[(i + u) mod L]``.  Flipping bit ``p`` negates exactly the
    terms at ``i = p`` and ``i = p - u (mod L)``.  Each old term ``c`` thus
    changes by ``-2c``.  Applying that delta for every nonzero shift updates
    a full profile in O(L), with integer arithmetic only.
    """

    def __init__(self, a: BinaryInput, b: BinaryInput) -> None:
        """Create state from equal-length non-empty binary sequences."""
        a_bits = normalize_binary_sequence(a)
        b_bits = normalize_binary_sequence(b)
        if len(a_bits) != len(b_bits):
            raise ValueError("a and b must have equal lengths")

        self._a_signs = [1 if bit == 0 else -1 for bit in a_bits]
        self._b_signs = [1 if bit == 0 else -1 for bit in b_bits]
        self._profile = self._initial_profile()

    @property
    def L(self) -> int:
        """Return the common sequence length."""
        return len(self._a_signs)

    @property
    def a(self) -> Tuple[int, ...]:
        """Return the current A sequence as immutable binary bits."""
        return tuple(0 if sign == 1 else 1 for sign in self._a_signs)

    @property
    def b(self) -> Tuple[int, ...]:
        """Return the current B sequence as immutable binary bits."""
        return tuple(0 if sign == 1 else 1 for sign in self._b_signs)

    @property
    def profile(self) -> Tuple[int, ...]:
        """Return the current exact pair profile ``S[0]`` through ``S[L-1]``."""
        return tuple(self._profile)

    def flip_a(self, position: int) -> None:
        """Flip ``A[position]`` and incrementally update the pair profile."""
        self._flip(self._a_signs, position)

    def flip_b(self, position: int) -> None:
        """Flip ``B[position]`` and incrementally update the pair profile."""
        self._flip(self._b_signs, position)

    def _initial_profile(self) -> List[int]:
        """Compute the initial profile once from the compact sign arrays."""
        length = self.L
        return [
            sum(
                self._a_signs[index] * self._a_signs[(index + shift) % length]
                + self._b_signs[index] * self._b_signs[(index + shift) % length]
                for index in range(length)
            )
            for shift in range(length)
        ]

    def _flip(self, signs: List[int], position: int) -> None:
        """Apply an O(L) profile update after one sign's corresponding bit flips."""
        self._validate_position(position)
        length = self.L
        old_sign = signs[position]

        # At shift zero every term is x[i]**2, so no term changes.  For u > 0,
        # the two affected ordered terms are indexed by position and position-u.
        for shift in range(1, length):
            forward = signs[(position + shift) % length]
            backward = signs[(position - shift) % length]
            self._profile[shift] -= 2 * old_sign * (forward + backward)

        signs[position] = -old_sign

    def _validate_position(self, position: int) -> None:
        """Require a canonical sequence index for a public flip operation."""
        if (not isinstance(position, int) or isinstance(position, bool)
                or not 0 <= position < self.L):
            raise ValueError("position must be an integer in the range 0 through L - 1")
