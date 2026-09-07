"""Exact incremental periodic-correlation evaluation for bit-flip searches.

The state stores binary sequences as signs (+1 for bit 0 and -1 for bit 1)
and retains their pair autocorrelation profile.  This is a compact reusable
representation, not a mathematical block-compression transformation.
"""

from typing import List, Optional, Sequence, Tuple, Union

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

        self._a_bits = list(a_bits)
        self._b_bits = list(b_bits)
        self._a_cache: Optional[Tuple[int, ...]] = tuple(a_bits)
        self._b_cache: Optional[Tuple[int, ...]] = tuple(b_bits)
        self._a_signs = [1 if bit == 0 else -1 for bit in a_bits]
        self._b_signs = [1 if bit == 0 else -1 for bit in b_bits]
        self._profile = self._initial_profile()
        self._distance_offset = 2 * self.L
        self._distance_table = tuple(
            _target_distance(value) for value in range(-2 * self.L, 2 * self.L + 1)
        )
        self._target_value_distance = sum(
            self._distance_table[value + self._distance_offset] for value in self._profile[1:]
        )
        self._nonzero_count = sum(value != 0 for value in self._profile[1:])
        self._squared_sidelobes = sum(value * value for value in self._profile[1:])
        self._representative_magnitude_counts = [0] * (2 * self.L + 1)
        for shift in range(1, (self.L + 1) // 2):
            self._representative_magnitude_counts[abs(self._profile[shift])] += 1
        self._max_representative_magnitude = max(
            (magnitude for magnitude, count in enumerate(self._representative_magnitude_counts) if count),
            default=0,
        )

    @property
    def L(self) -> int:
        """Return the common sequence length."""
        return len(self._a_signs)

    @property
    def a(self) -> Tuple[int, ...]:
        """Return the current A sequence as immutable binary bits."""
        if self._a_cache is None:
            self._a_cache = tuple(self._a_bits)
        return self._a_cache

    @property
    def b(self) -> Tuple[int, ...]:
        """Return the current B sequence as immutable binary bits."""
        if self._b_cache is None:
            self._b_cache = tuple(self._b_bits)
        return self._b_cache

    @property
    def profile(self) -> Tuple[int, ...]:
        """Return the current exact pair profile ``S[0]`` through ``S[L-1]``."""
        return tuple(self._profile)

    @property
    def score(self) -> int:
        """Return the exact Project 2 objective for the maintained profile.

        Profiles produced from binary pairs always have ``S[0]=2L`` and
        exact periodic symmetry.  The state therefore maintains the two
        remaining objective components incrementally during the same O(L)
        loop that updates correlation, avoiding a second full profile scan.
        """
        return self._target_value_distance + abs(self._nonzero_count - 2)

    @property
    def target_pair_energy(self) -> int:
        """Return exact squared error to the closest allowed nonzero shift pair."""
        return self._squared_sidelobes + 32 - 16 * self._max_representative_magnitude

    def flip_a(self, position: int) -> None:
        """Flip ``A[position]`` and incrementally update the pair profile."""
        self._validate_position(position)
        self._flip_a_unchecked(position)

    def flip_b(self, position: int) -> None:
        """Flip ``B[position]`` and incrementally update the pair profile."""
        self._validate_position(position)
        self._flip_b_unchecked(position)

    def _flip_a_unchecked(self, position: int) -> None:
        """Flip a solver-validated A index without repeating public validation."""
        self._flip(self._a_signs, position)
        self._a_bits[position] ^= 1
        self._a_cache = None

    def _flip_b_unchecked(self, position: int) -> None:
        """Flip a solver-validated B index without repeating public validation."""
        self._flip(self._b_signs, position)
        self._b_bits[position] ^= 1
        self._b_cache = None

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
        length = self.L
        old_sign = signs[position]

        # At shift zero every term is x[i]**2, so no term changes.  Periodic
        # symmetry makes shifts u and L-u identical; compute one representative
        # and mirror it, including the self-symmetric L/2 shift exactly once.
        profile = self._profile
        distance = self._distance_table
        offset = self._distance_offset
        target_delta = 0
        count_delta = 0
        squared_delta = 0
        for shift in range(1, length // 2 + 1):
            forward_index = position + shift
            if forward_index >= length:
                forward_index -= length
            backward_index = position - shift
            if backward_index < 0:
                backward_index += length
            forward = signs[forward_index]
            backward = signs[backward_index]
            old_value = profile[shift]
            new_value = old_value - 2 * old_sign * (forward + backward)
            mirror = length - shift
            multiplicity = 1 if mirror == shift else 2
            profile[shift] = new_value
            profile[mirror] = new_value
            target_delta += multiplicity * (distance[new_value + offset] - distance[old_value + offset])
            count_delta += multiplicity * ((new_value != 0) - (old_value != 0))
            squared_delta += multiplicity * (new_value * new_value - old_value * old_value)
            if mirror != shift:
                magnitudes = self._representative_magnitude_counts
                magnitudes[abs(old_value)] -= 1
                magnitudes[abs(new_value)] += 1
                if abs(new_value) > self._max_representative_magnitude:
                    self._max_representative_magnitude = abs(new_value)

        self._target_value_distance += target_delta
        self._nonzero_count += count_delta
        self._squared_sidelobes += squared_delta
        while (self._max_representative_magnitude > 0
               and self._representative_magnitude_counts[self._max_representative_magnitude] == 0):
            self._max_representative_magnitude -= 1

        signs[position] = -old_sign

    def _validate_position(self, position: int) -> None:
        """Require a canonical sequence index for a public flip operation."""
        if (not isinstance(position, int) or isinstance(position, bool)
                or not 0 <= position < self.L):
            raise ValueError("position must be an integer in the range 0 through L - 1")


def _target_distance(value: int) -> int:
    """Return exact distance from an integer to ``{-4, 0, +4}``."""
    magnitude = abs(value)
    return min(magnitude, abs(magnitude - 4))
