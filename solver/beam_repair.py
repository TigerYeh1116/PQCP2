"""Deterministic fixed-weight beam repair for complete PQCP elites.

This is a bounded final-completion algorithm, not a replacement for global
candidate generation.  It expands the same legal zero/one swaps used by the
production SA, ranks exact incremental outcomes by the established navigation
energy, and independently verifies every score-zero state.  A miss means only
that the configured beam/depth did not find a solution; it is never UNSAT.
"""

from dataclasses import dataclass
from time import perf_counter
from typing import Dict, Optional, Sequence, Tuple

from .compression import CorrelationState
from .moves import (
    WeightPreservingSwap,
    apply_weight_preserving_swap,
    rollback_weight_preserving_swap,
    trial_weight_preserving_swap,
)
from .objective import target_profile_l1_distance
from .verifier import verify_pqcp


BinaryPair = Tuple[Tuple[int, ...], Tuple[int, ...]]


@dataclass(frozen=True)
class BeamRepairResult:
    """Exact candidate and bounded-search diagnostics."""

    solved: bool
    a: Tuple[int, ...]
    b: Tuple[int, ...]
    profile: Tuple[int, ...]
    depth: Optional[int]
    initial_score: int
    best_score: int
    states_examined: int
    beam_width: int
    max_depth: int
    elapsed_time: float


def beam_repair(
    a: Sequence[int],
    b: Sequence[int],
    max_depth: int = 4,
    beam_width: int = 30,
    objective_energy_weight: int = 3,
    ranking_mode: str = "navigation",
) -> BeamRepairResult:
    """Search a bounded beam of exact fixed-weight swap trajectories.

    The complete first layer is evaluated.  At later layers, the best
    ``beam_width`` distinct states are retained according to

    ``objective_energy_weight * pqcp_objective + target_pair_energy // 16``.

    This matches the current production SA navigation scale but removes
    temperature and random proposal sampling for the final repair phase.
    All arithmetic used for ranking is exact integer arithmetic.
    """
    if not isinstance(max_depth, int) or isinstance(max_depth, bool) or max_depth < 0:
        raise ValueError("max_depth must be a non-negative integer")
    if not isinstance(beam_width, int) or isinstance(beam_width, bool) or beam_width <= 0:
        raise ValueError("beam_width must be a positive integer")
    if (
        not isinstance(objective_energy_weight, int)
        or isinstance(objective_energy_weight, bool)
        or objective_energy_weight <= 0
    ):
        raise ValueError("objective_energy_weight must be a positive integer")
    if ranking_mode not in ("navigation", "target_l1", "objective"):
        raise ValueError("ranking_mode must be navigation, target_l1, or objective")
    started = perf_counter()
    initial = CorrelationState(a, b)
    initial_score = initial.score
    if initial_score == 0:
        verification = verify_pqcp(initial.a, initial.b)
        if not verification.is_valid:
            raise RuntimeError("score-zero beam center failed independent verification")
        return BeamRepairResult(
            True, initial.a, initial.b, verification.profile, 0,
            0, 0, 1, beam_width, max_depth, perf_counter() - started,
        )

    frontier: Tuple[BinaryPair, ...] = ((initial.a, initial.b),)
    seen = set(frontier)
    best_score = initial_score
    best_pair = (initial.a, initial.b)
    best_profile = initial.profile
    best_rank = _rank(
        initial.score, initial.target_pair_energy, initial.profile, best_pair,
        objective_energy_weight, ranking_mode,
    )
    states_examined = 1
    for depth in range(1, max_depth + 1):
        candidates: Dict[BinaryPair, Tuple[object, ...]] = {}
        for frontier_a, frontier_b in frontier:
            state = CorrelationState(frontier_a, frontier_b)
            for sequence_name, values in (("a", state.a), ("b", state.b)):
                zeros = tuple(index for index, bit in enumerate(values) if bit == 0)
                ones = tuple(index for index, bit in enumerate(values) if bit == 1)
                for zero in zeros:
                    for one in ones:
                        move = WeightPreservingSwap(sequence_name, zero, one)
                        evaluation = trial_weight_preserving_swap(state, move)
                        apply_weight_preserving_swap(state, move)
                        pair = (state.a, state.b)
                        if pair not in seen and pair not in candidates:
                            states_examined += 1
                            candidate_rank = _rank(
                                evaluation.score, evaluation.target_pair_energy,
                                state.profile, pair, objective_energy_weight,
                                ranking_mode,
                            )
                            if candidate_rank < best_rank:
                                best_rank = candidate_rank
                                best_pair = pair
                                best_profile = state.profile
                                best_score = evaluation.score
                            if evaluation.score == 0:
                                verification = verify_pqcp(*pair)
                                rollback_weight_preserving_swap(state, move)
                                if not verification.is_valid:
                                    raise RuntimeError("score-zero beam state failed independent verification")
                                return BeamRepairResult(
                                    True, pair[0], pair[1], verification.profile, depth,
                                    initial_score, 0, states_examined, beam_width,
                                    max_depth, perf_counter() - started,
                                )
                            candidates[pair] = candidate_rank
                        rollback_weight_preserving_swap(state, move)
        if not candidates:
            break
        ordered = sorted(candidates, key=candidates.__getitem__)
        frontier = tuple(ordered[:beam_width])
        seen.update(frontier)
    return BeamRepairResult(
        False, best_pair[0], best_pair[1], best_profile, None, initial_score, best_score,
        states_examined, beam_width, max_depth, perf_counter() - started,
    )


def _rank(
    score: int,
    target_pair_energy: int,
    profile: Sequence[int],
    pair: BinaryPair,
    objective_energy_weight: int,
    ranking_mode: str,
) -> Tuple[object, ...]:
    """Return one deterministic rank; alternate modes support experiments."""
    navigation = objective_energy_weight * score + target_pair_energy // 16
    if ranking_mode == "target_l1":
        target_l1 = target_profile_l1_distance(profile)
        return target_l1, navigation, score, target_pair_energy, pair
    if ranking_mode == "objective":
        return score, target_pair_energy, pair
    return navigation, score, target_pair_energy, pair
