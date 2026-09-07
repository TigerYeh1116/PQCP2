"""Correlation-guided bounded completion before exact Z3 confirmation.

For signs ``x[i] = (-1)**bit[i]``, flipping a set ``F`` changes periodic
autocorrelation at shift ``u`` only on directed edges crossing the cut
``F``/``F^c``::

    delta rho(u) = -2 * sum(x[i] * x[i+u] for crossing directed edges)

At most ``2r`` directed edges cross for ``r`` flipped bits, hence every pair
profile coordinate changes by at most ``4r``.  The cut size is even on each
cycle, so the change is a multiple of four.  This supplies a safe Hamming
lower bound, not a heuristic rejection of a possible completion.

Production SA preserves an admissible Hamming-weight pair.  We enumerate all
one- or two-bit moves whose resulting pair is still mathematically admissible,
rank the first layer by the unchanged Project objective, and inspect a bounded
number of second layers.  A returned candidate is always recomputed and passed
through the independent verifier; ranking affects coverage, never correctness.
"""

from dataclasses import dataclass
from itertools import combinations
from typing import Optional, Sequence, Tuple

from .compression import CorrelationState
from .correlation import normalize_binary_sequence
from .verifier import verify_pqcp
from .weight_constraints import admissible_weight_pairs


Flip = Tuple[str, int]
FlipMove = Tuple[Flip, ...]
BinaryPair = Tuple[Tuple[int, ...], Tuple[int, ...]]


@dataclass(frozen=True)
class GuidedCompletion:
    """Exact result and diagnostics from the bounded guided neighborhood."""

    a: Optional[Tuple[int, ...]]
    b: Optional[Tuple[int, ...]]
    lower_bound: int
    first_moves_examined: int
    second_moves_examined: int
    top_k: int

    @property
    def solved(self) -> bool:
        """Return whether an independently verified candidate was found."""
        return self.a is not None and self.b is not None


def correlation_hamming_lower_bound(profile: Sequence[int]) -> int:
    """Return a safe lower bound on bits needed to reach any target profile.

    Project 2 has exactly one nonzero periodic-symmetry representative with
    value ``+4`` or ``-4``; every other representative, including the even
    half shift, is zero.  For every such possible target profile, take its
    maximum coordinate discrepancy, divide by the per-bit bound four, and
    minimize over target position/sign.
    """
    values = tuple(profile)
    length = len(values)
    if length < 3 or any(not isinstance(value, int) or isinstance(value, bool) for value in values):
        raise ValueError("profile must be an integer Project profile of length at least 3")
    representatives = tuple(range(1, (length + 1) // 2))
    if not representatives:
        raise ValueError("profile has no eligible nonzero symmetry representative")
    best_maximum = None
    for target_shift in representatives:
        for target_value in (-4, 4):
            maximum = abs(values[length // 2]) if length % 2 == 0 else 0
            for shift in representatives:
                expected = target_value if shift == target_shift else 0
                maximum = max(maximum, abs(values[shift] - expected))
            if best_maximum is None or maximum < best_maximum:
                best_maximum = maximum
    if best_maximum is None:  # pragma: no cover - guarded by representatives
        raise AssertionError("target profile enumeration unexpectedly empty")
    return (best_maximum + 3) // 4


def admissible_short_moves(a: Sequence[int], b: Sequence[int]) -> Tuple[FlipMove, ...]:
    """Enumerate all one/two-bit moves ending at an admissible weight pair."""
    a_bits = normalize_binary_sequence(a)
    b_bits = normalize_binary_sequence(b)
    if len(a_bits) != len(b_bits):
        raise ValueError("a and b must have equal lengths")
    length = len(a_bits)
    allowed = frozenset(admissible_weight_pairs(length))
    if not allowed:
        return ()
    weight_a, weight_b = sum(a_bits), sum(b_bits)
    indexed = tuple(("a", index) for index in range(length)) + tuple(
        ("b", index) for index in range(length)
    )
    moves = []
    for size in (1, 2):
        for move in combinations(indexed, size):
            delta_a = sum(1 if a_bits[index] == 0 else -1 for name, index in move if name == "a")
            delta_b = sum(1 if b_bits[index] == 0 else -1 for name, index in move if name == "b")
            if (weight_a + delta_a, weight_b + delta_b) in allowed:
                moves.append(move)
    return tuple(moves)


def guided_completion(
    a: Sequence[int],
    b: Sequence[int],
    top_k: int = 10,
    max_profile_lower_bound: int = 4,
) -> GuidedCompletion:
    """Search a correlation-ranked subset through four bit flips exactly.

    First, the complete admissible radius-two neighborhood is checked.  If it
    has no solution, at most ``top_k`` first moves are applied and each one's
    complete admissible radius-two neighborhood is checked.  Thus every
    reported solution is within four flips of the center.  Not finding one is
    only a miss in the ranked radius-four subset, never a global UNSAT claim.
    """
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    if (
        not isinstance(max_profile_lower_bound, int)
        or isinstance(max_profile_lower_bound, bool)
        or max_profile_lower_bound < 0
    ):
        raise ValueError("max_profile_lower_bound must be a non-negative integer")
    state = CorrelationState(a, b)
    lower_bound = correlation_hamming_lower_bound(state.profile)
    if state.score == 0:
        verification = verify_pqcp(state.a, state.b)
        if not verification.is_valid:
            raise RuntimeError("score-zero guidance center failed independent verification")
        return GuidedCompletion(state.a, state.b, lower_bound, 0, 0, top_k)
    if lower_bound > max_profile_lower_bound:
        return GuidedCompletion(None, None, lower_bound, 0, 0, top_k)

    first_moves = admissible_short_moves(state.a, state.b)
    ranked = []
    for move in first_moves:
        _apply_move(state, move)
        ranked.append((state.score, state.target_pair_energy, move))
        if state.score == 0:
            solution = _verified_pair(state)
            _rollback_move(state, move)
            return GuidedCompletion(solution[0], solution[1], lower_bound, len(ranked), 0, top_k)
        _rollback_move(state, move)
    ranked.sort(key=lambda item: (item[0], item[1], item[2]))

    second_examined = 0
    for _, _, first_move in ranked[:top_k]:
        _apply_move(state, first_move)
        for second_move in admissible_short_moves(state.a, state.b):
            _apply_move(state, second_move)
            second_examined += 1
            if state.score == 0:
                solution = _verified_pair(state)
                _rollback_move(state, second_move)
                _rollback_move(state, first_move)
                return GuidedCompletion(
                    solution[0], solution[1], lower_bound,
                    len(first_moves), second_examined, top_k,
                )
            _rollback_move(state, second_move)
        _rollback_move(state, first_move)
    return GuidedCompletion(None, None, lower_bound, len(first_moves), second_examined, top_k)


def _apply_move(state: CorrelationState, move: FlipMove) -> None:
    for sequence, position in move:
        state.flip_a(position) if sequence == "a" else state.flip_b(position)


def _rollback_move(state: CorrelationState, move: FlipMove) -> None:
    for sequence, position in reversed(move):
        state.flip_a(position) if sequence == "a" else state.flip_b(position)


def _verified_pair(state: CorrelationState) -> BinaryPair:
    verification = verify_pqcp(state.a, state.b)
    if not verification.is_valid:
        raise RuntimeError("guided score-zero candidate failed independent verification")
    return state.a, state.b
