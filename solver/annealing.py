"""Reproducible fixed-weight swap annealing for Project 2 PQCP search.

SA states are complete binary pairs, maintained by ``CorrelationState``.  A
complete-profile pruning failure means only that the *current* pair is not a
solution; later flips can still reach one.  It is therefore used as a safe
terminal-solution gate, not as an unsafe hard rejection of ordinary SA moves.
"""

from dataclasses import dataclass
from itertools import islice
import math
import random
from time import perf_counter
from typing import List, Optional, Sequence, Tuple, Union

from .compression import CorrelationState
from .fkm import generate_fkm_sequences
from .moves import (
    apply_weight_preserving_swap,
    choose_weight_preserving_swap,
    hamming_weights,
    rollback_weight_preserving_swap,
)
from .pruning import prune_complete_profile
from .verifier import verify_pqcp
from .target_profiles import TargetContentProfile
from .seed_selection import select_fkm_content_seed


BinaryInput = Sequence[Union[int, str]]
BinaryPair = Tuple[Tuple[int, ...], Tuple[int, ...]]


@dataclass(frozen=True)
class AnnealingParameters:
    """Explicit, reproducible controls for the baseline SA implementation."""

    max_iterations: int = 10_000
    initial_temperature: float = 8.0
    cooling_rate: float = 0.999
    min_temperature: float = 0.05
    seed: Optional[int] = None
    restart_count: int = 1
    weight: Optional[int] = None
    fkm_pool_size: int = 128
    verbose: bool = False

    def __post_init__(self) -> None:
        """Validate SA controls without imposing a Project 2 Hamming weight."""
        if not isinstance(self.max_iterations, int) or isinstance(self.max_iterations, bool) or self.max_iterations < 0:
            raise ValueError("max_iterations must be a non-negative integer")
        if not isinstance(self.restart_count, int) or isinstance(self.restart_count, bool) or self.restart_count <= 0:
            raise ValueError("restart_count must be a positive integer")
        if not isinstance(self.fkm_pool_size, int) or isinstance(self.fkm_pool_size, bool) or self.fkm_pool_size <= 0:
            raise ValueError("fkm_pool_size must be a positive integer")
        if self.seed is not None and (not isinstance(self.seed, int) or isinstance(self.seed, bool)):
            raise ValueError("seed must be an integer or None")
        if self.weight is not None and (not isinstance(self.weight, int) or isinstance(self.weight, bool) or self.weight < 0):
            raise ValueError("weight must be a non-negative integer or None")
        if self.initial_temperature <= 0 or self.min_temperature <= 0:
            raise ValueError("temperatures must be positive")
        if self.min_temperature > self.initial_temperature:
            raise ValueError("min_temperature must not exceed initial_temperature")
        if not 0 < self.cooling_rate <= 1:
            raise ValueError("cooling_rate must be in the interval (0, 1]")


@dataclass(frozen=True)
class AnnealingResult:
    """Best state and reproducibility data from an annealing execution."""

    best_a: Tuple[int, ...]
    best_b: Tuple[int, ...]
    best_profile: Tuple[int, ...]
    best_score: int
    initial_scores: Tuple[int, ...]
    iterations: int
    restarts: int
    elapsed_time: float
    seed: Optional[int]
    solved: bool
    independently_verified: bool
    accepted_moves: int
    best_score_history: Tuple[int, ...]


def initialize_from_fkm(
    L: int,
    weight: Optional[int] = None,
    seed: Optional[int] = None,
    pool_size: int = 128,
) -> BinaryPair:
    """Choose an A/B initial pair from a bounded deterministic FKM candidate pool.

    No weight is assumed when ``weight`` is None.  The bounded pool preserves
    generator-based FKM operation while making initialization reproducible;
    the two representatives are selected independently using ``seed``.
    """
    if not isinstance(pool_size, int) or isinstance(pool_size, bool) or pool_size <= 0:
        raise ValueError("pool_size must be a positive integer")
    rng = random.Random(seed)
    return _initialize_from_fkm_rng(L, weight, pool_size, rng)


def initialize_from_fkm_weights(
    L: int,
    weight_a: int,
    weight_b: int,
    seed: Optional[int] = None,
    pool_size: int = 128,
) -> BinaryPair:
    """Initialize A/B independently from existing FKM representatives by weight."""
    rng = random.Random(seed)
    return _initialize_from_fkm_weights_rng(L, weight_a, weight_b, pool_size, rng)


def initialize_from_fkm_content_profile(
    profile: TargetContentProfile,
    seed: Optional[int] = None,
    pool_size: int = 128,
    policy: str = "legacy",
    candidate_count: int = 16,
    elite_count: int = 4,
) -> BinaryPair:
    """Build A/B from four bounded FKM parity-subsequence pools.

    Even and odd coordinates are generated as independent binary necklaces of
    length ``L/2`` with the exact counts derived for ``profile`` and then
    interleaved.  This is a structured initializer, not a claim that rotating
    each parity subsequence is a symmetry of the original PQCP problem.

    ``legacy`` exactly preserves the historical four random pool choices.
    Alternative policies delegate only seed observation/ranking to
    :mod:`solver.seed_selection`; they never change FKM generation, target
    content, the later SA RNG, or the verifier.
    """
    if seed is None:
        # Retain the public API's nondeterministic ``None`` behavior while the
        # selector itself keeps an auditable integer seed contract.
        seed = random.SystemRandom().randrange(-(2 ** 63), 2 ** 63)
    return select_fkm_content_seed(
        profile,
        seed=seed,
        pool_size=pool_size,
        policy=policy,
        candidate_count=candidate_count,
        elite_count=elite_count,
    ).pair


def simulated_annealing(
    L: int,
    parameters: AnnealingParameters,
    initial_pair: Optional[Tuple[BinaryInput, BinaryInput]] = None,
) -> AnnealingResult:
    """Search complete binary A/B pairs with fixed-weight swap SA.

    For an uphill move of integer score difference ``delta``, the standard
    acceptance probability is ``exp(-delta / temperature)``.  A rejected
    move is rolled back by the same two O(L) incremental bit flips; complete
    correlation recomputation is never used in the move loop.
    """
    if not isinstance(L, int) or isinstance(L, bool) or L <= 0:
        raise ValueError("L must be a positive integer")
    master_rng = random.Random(parameters.seed)
    started = perf_counter()
    initial_scores: List[int] = []
    history: List[int] = []
    best_a: Optional[Tuple[int, ...]] = None
    best_b: Optional[Tuple[int, ...]] = None
    best_profile: Optional[Tuple[int, ...]] = None
    best_score: Optional[int] = None
    accepted_moves = 0
    total_iterations = 0
    completed_restarts = 0
    independently_verified = False

    for restart_index in range(parameters.restart_count):
        restart_rng = random.Random(master_rng.randrange(2 ** 63))
        if initial_pair is None:
            state = _state_from_fkm(L, parameters, restart_rng)
        else:
            state = CorrelationState(initial_pair[0], initial_pair[1])
            if state.L != L:
                raise ValueError("initial_pair sequences must both have length L")

        current_score = state.score
        initial_scores.append(current_score)
        if best_score is None or current_score < best_score:
            best_a, best_b, best_profile, best_score = state.a, state.b, state.profile, current_score
        history.append(best_score)

        if best_score == 0:
            independently_verified = _verify_terminal_candidate(best_a, best_b, best_profile)
            completed_restarts = restart_index + 1
            break

        temperature = parameters.initial_temperature
        restart_weights = hamming_weights(state.a, state.b)
        for _ in range(parameters.max_iterations):
            move = choose_weight_preserving_swap(state.a, state.b, restart_rng)
            if move is not None:
                apply_weight_preserving_swap(state, move)
                proposed_score = state.score
                delta = proposed_score - current_score
                accepted = delta <= 0 or restart_rng.random() < math.exp(-delta / temperature)
                if accepted:
                    current_score = proposed_score
                    accepted_moves += 1
                    if current_score < best_score:
                        best_a, best_b, best_profile, best_score = state.a, state.b, state.profile, current_score
                else:
                    rollback_weight_preserving_swap(state, move)
            if hamming_weights(state.a, state.b) != restart_weights:
                raise RuntimeError("weight-preserving annealing move changed a Hamming weight")

            total_iterations += 1
            history.append(best_score)
            if best_score == 0:
                independently_verified = _verify_terminal_candidate(best_a, best_b, best_profile)
                break
            temperature = max(parameters.min_temperature, temperature * parameters.cooling_rate)

        completed_restarts = restart_index + 1
        if parameters.verbose:
            print("L={} restart={} initial_score={} best_score={} iterations={}".format(
                L, completed_restarts, initial_scores[-1], best_score, total_iterations
            ))
        if best_score == 0:
            break

    assert best_a is not None and best_b is not None and best_profile is not None and best_score is not None
    solved = best_score == 0 and independently_verified
    return AnnealingResult(
        best_a=best_a,
        best_b=best_b,
        best_profile=best_profile,
        best_score=best_score,
        initial_scores=tuple(initial_scores),
        iterations=total_iterations,
        restarts=completed_restarts,
        elapsed_time=perf_counter() - started,
        seed=parameters.seed,
        solved=solved,
        independently_verified=independently_verified,
        accepted_moves=accepted_moves,
        best_score_history=tuple(history),
    )


def _initialize_from_fkm_rng(
    L: int,
    weight: Optional[int],
    pool_size: int,
    rng: random.Random,
) -> BinaryPair:
    """Build a bounded FKM pool and select an independent pair from it."""
    candidates = tuple(islice(generate_fkm_sequences(L, weight=weight), pool_size))
    if not candidates:
        raise ValueError("FKM produced no candidates for the requested L and weight")
    return rng.choice(candidates), rng.choice(candidates)


def _initialize_from_fkm_weights_rng(
    L: int, weight_a: int, weight_b: int, pool_size: int, rng: random.Random,
) -> BinaryPair:
    """Choose an ordered independently weighted FKM pair from bounded pools."""
    candidates_a = tuple(islice(generate_fkm_sequences(L, weight=weight_a), pool_size))
    candidates_b = tuple(islice(generate_fkm_sequences(L, weight=weight_b), pool_size))
    if not candidates_a or not candidates_b:
        raise ValueError("FKM produced no candidates for the requested A/B weights")
    return rng.choice(candidates_a), rng.choice(candidates_b)


def _state_from_fkm(L: int, parameters: AnnealingParameters, rng: random.Random) -> CorrelationState:
    """Create one restart state from FKM representatives using its restart RNG."""
    a, b = _initialize_from_fkm_rng(L, parameters.weight, parameters.fkm_pool_size, rng)
    return CorrelationState(a, b)


def _verify_terminal_candidate(a: Tuple[int, ...], b: Tuple[int, ...], profile: Tuple[int, ...]) -> bool:
    """Use only safe target checks and the independent verifier for score-zero output."""
    if prune_complete_profile(profile).should_prune:
        return False
    return verify_pqcp(a, b).is_valid
