"""Empirical candidate-difficulty metrics kept separate from exact validity.

``repair_difficulty`` combines distance to one complete legal correlation
target with the local barrier seen by the production fixed-weight single-swap
neighborhood.  It was designed to answer whether a low profile score is also
easy for the current search to repair.  It must never be used as SAFE pruning:
larger values do not prove that a PQCP is unreachable.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

from .compression import CorrelationState
from .moves import (
    WeightPreservingSwap,
    apply_weight_preserving_swap,
    rollback_weight_preserving_swap,
)
from .objective import target_profile_l1_distance


@dataclass(frozen=True)
class RepairDifficulty:
    """Breakdown of one exact, deterministic local difficulty observation."""

    total: int
    target_l1: int
    best_neighbor_l1: Optional[int]
    local_barrier: int
    legal_moves: int
    improving_moves: int


@dataclass(frozen=True)
class LookaheadDifficulty:
    """Beam lookahead estimate of remaining fixed-weight swap cost."""

    total: int
    initial_target_l1: int
    best_depth: int
    residual_target_l1: int
    states_examined: int
    beam_width: Optional[int]
    solution_depth: Optional[int]


def repair_difficulty(a: Sequence[int], b: Sequence[int]) -> RepairDifficulty:
    """Score target distance and the complete fixed-weight one-swap basin.

    Let ``d`` be L1 distance (in correlation units of four) to the closest
    complete target profile, and ``m`` the best ``d`` among all legal
    fixed-weight one-swap neighbors.  The local barrier is zero if a neighbor
    improves, otherwise ``m-d+1``.  The final score is

    ``d * (1 + local_barrier)``.

    Hence every exact target scores zero.  A non-solution local minimum is
    ranked as harder than an equal-distance state with a downhill move.  The
    full neighborhood makes the result deterministic and independent of RNG.
    It is O(L^3) in this analysis implementation and is intended for elite
    evaluation/triggering, not every SA iteration.
    """
    state = CorrelationState(a, b)
    current = target_profile_l1_distance(state.profile)
    best_neighbor = None
    legal_moves = 0
    improving_moves = 0
    for name, values in (("a", state.a), ("b", state.b)):
        zeros = tuple(index for index, bit in enumerate(values) if bit == 0)
        ones = tuple(index for index, bit in enumerate(values) if bit == 1)
        for zero in zeros:
            for one in ones:
                move = WeightPreservingSwap(name, zero, one)
                apply_weight_preserving_swap(state, move)
                neighbor = target_profile_l1_distance(state.profile)
                rollback_weight_preserving_swap(state, move)
                legal_moves += 1
                improving_moves += int(neighbor < current)
                if best_neighbor is None or neighbor < best_neighbor:
                    best_neighbor = neighbor
    if current == 0:
        barrier = 0
    elif best_neighbor is None:
        barrier = 1
    else:
        barrier = max(0, best_neighbor - current + 1)
    return RepairDifficulty(
        total=current * (1 + barrier),
        target_l1=current,
        best_neighbor_l1=best_neighbor,
        local_barrier=barrier,
        legal_moves=legal_moves,
        improving_moves=improving_moves,
    )


def lookahead_repair_difficulty(
    a: Sequence[int],
    b: Sequence[int],
    max_depth: int = 2,
    beam_width: Optional[int] = 10,
) -> LookaheadDifficulty:
    """Estimate cost-to-go using exact fixed-weight swap neighborhoods.

    For every state retained through ``max_depth``, evaluate

    ``moves already used + target_profile_l1_distance(state)``

    and retain the minimum residual estimate.  If an exact target is reached,
    ``total`` is its first depth.  Otherwise ``total`` is the residual estimate
    plus ``max_depth + 1``, recording that the bounded repair attempt was
    exhausted without success.  The complete first layer is always evaluated;
    subsequent layers expand the ``beam_width`` lowest-residual distinct
    states.  ``beam_width=None`` performs exhaustive breadth expansion and is
    intended only for small-L validation.

    The value is zero exactly when the input itself is a target.  A candidate
    with an exact solution one or two legal swaps away scores at most one or
    two respectively, making the metric directly reflect the neighborhood
    used by the existing search.  Beam truncation makes this a heuristic,
    never a SAFE lower bound or pruning condition.
    """
    if not isinstance(max_depth, int) or isinstance(max_depth, bool) or max_depth < 0:
        raise ValueError("max_depth must be a non-negative integer")
    if beam_width is not None and (
        not isinstance(beam_width, int) or isinstance(beam_width, bool) or beam_width <= 0
    ):
        raise ValueError("beam_width must be a positive integer or None")
    initial = CorrelationState(a, b)
    initial_l1 = target_profile_l1_distance(initial.profile)
    best = (initial_l1, 0, initial_l1)
    solution_depth = 0 if initial_l1 == 0 else None
    examined = 1
    frontier: Tuple[Tuple[Tuple[int, ...], Tuple[int, ...]], ...] = ((initial.a, initial.b),)
    seen = {(initial.a, initial.b)}
    for depth in range(1, max_depth + 1):
        candidates: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], Tuple[int, int]] = {}
        for frontier_a, frontier_b in frontier:
            state = CorrelationState(frontier_a, frontier_b)
            for name, values in (("a", state.a), ("b", state.b)):
                zeros = tuple(index for index, bit in enumerate(values) if bit == 0)
                ones = tuple(index for index, bit in enumerate(values) if bit == 1)
                for zero in zeros:
                    for one in ones:
                        move = WeightPreservingSwap(name, zero, one)
                        apply_weight_preserving_swap(state, move)
                        pair = (state.a, state.b)
                        if pair not in seen and pair not in candidates:
                            residual = target_profile_l1_distance(state.profile)
                            candidates[pair] = (residual, state.score)
                            examined += 1
                            estimate = depth + residual
                            if residual == 0 and solution_depth is None:
                                solution_depth = depth
                            if (estimate, depth, residual) < best:
                                best = (estimate, depth, residual)
                        rollback_weight_preserving_swap(state, move)
        if not candidates:
            break
        ordered = sorted(
            candidates,
            key=lambda pair: (candidates[pair][0], candidates[pair][1], pair),
        )
        if beam_width is not None:
            ordered = ordered[:beam_width]
        frontier = tuple(ordered)
        seen.update(frontier)
    total = solution_depth if solution_depth is not None else max_depth + 1 + best[0]
    return LookaheadDifficulty(
        total=total, initial_target_l1=initial_l1,
        best_depth=best[1], residual_target_l1=best[2],
        states_examined=examined, beam_width=beam_width,
        solution_depth=solution_depth,
    )
