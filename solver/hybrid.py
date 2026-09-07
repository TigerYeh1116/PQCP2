"""Deterministic multiple-SA-elite, small-radius Z3 portfolio orchestration.

This module coordinates existing solvers without changing their mathematics.
Every Z3 task is exact only for its selected Hamming neighborhood; exhausting
the portfolio never constitutes a global nonexistence claim.
"""

from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .annealing import AnnealingParameters, simulated_annealing
from .objective import pqcp_objective
from .verifier import verify_pqcp
from .z3_solver import Z3CompletionResult, save_verified_solution, solve_with_z3


BinaryPair = Tuple[Tuple[int, ...], Tuple[int, ...]]
Z3SolveFunction = Callable[[int, Sequence[int], Sequence[int], int, Optional[int]], Z3CompletionResult]

# Rank aggregation avoids mixing objective units with Hamming-distance units.
SCORE_RANK_WEIGHT = 1
# After anchoring the lowest-score elite, favor basin coverage over a one-rank
# quality difference.  Rank units, rather than raw score/distance units, keep
# this deterministic comparison scale-free.
DIVERSITY_RANK_WEIGHT = 2


@dataclass(frozen=True)
class SAElite:
    """One complete SA best state and the deterministic run that produced it."""

    a: Tuple[int, ...]
    b: Tuple[int, ...]
    score: int
    source_seed: int
    run_id: int


@dataclass(frozen=True)
class PortfolioConfig:
    """Bounded sequential Z3 schedule for selected SA elites."""

    timeout_radius_0_ms: int = 300
    timeout_radius_1_ms: int = 400
    timeout_radius_2_ms: int = 600
    timeout_radius_3_ms: int = 600
    radius_0_elite_count: Optional[int] = None
    radius_1_elite_count: Optional[int] = None
    radius_2_elite_count: int = 2
    radius_3_elite_count: int = 1
    total_timeout_seconds: float = 4.0

    def __post_init__(self) -> None:
        """Validate time caps and the bounded late-stage portfolio sizes."""
        for timeout in (
            self.timeout_radius_0_ms,
            self.timeout_radius_1_ms,
            self.timeout_radius_2_ms,
            self.timeout_radius_3_ms,
        ):
            if not isinstance(timeout, int) or isinstance(timeout, bool) or timeout < 0:
                raise ValueError("per-task timeouts must be non-negative integers")
        for count in (
            self.radius_0_elite_count, self.radius_1_elite_count,
            self.radius_2_elite_count, self.radius_3_elite_count,
        ):
            if count is not None and (not isinstance(count, int) or isinstance(count, bool) or count < 0):
                raise ValueError("late-stage elite counts must be non-negative integers")
        if self.total_timeout_seconds < 0:
            raise ValueError("total_timeout_seconds must be non-negative")


@dataclass(frozen=True)
class PortfolioTaskResult:
    """One neighborhood query, including its independent verification outcome."""

    elite: SAElite
    radius: int
    timeout_ms: int
    status: str
    elapsed_time: float
    verified: bool
    reason: Optional[str]


@dataclass(frozen=True)
class HybridResult:
    """All selected elites and bounded Z3 task outcomes for one length."""

    L: int
    collected_elites: Tuple[SAElite, ...]
    selected_elites: Tuple[SAElite, ...]
    tasks: Tuple[PortfolioTaskResult, ...]
    elapsed_time: float
    sa_elapsed_time: float
    portfolio_elapsed_time: float
    solved: bool
    solution: Optional[Z3CompletionResult]
    saved_path: Optional[Path]
    implementation_error: bool

    @property
    def status_counts(self) -> Dict[str, int]:
        """Return exact task-status counts without conflating UNKNOWN and UNSAT."""
        return {status: sum(task.status == status for task in self.tasks)
                for status in ("SAT", "UNSAT", "UNKNOWN")}


def collect_sa_elites(
    L: int,
    num_runs: int,
    parameters: AnnealingParameters,
    base_seed: int,
) -> Tuple[SAElite, ...]:
    """Collect one best pair from each independently seeded deterministic SA run."""
    if not isinstance(num_runs, int) or isinstance(num_runs, bool) or num_runs <= 0:
        raise ValueError("num_runs must be a positive integer")
    if not isinstance(base_seed, int) or isinstance(base_seed, bool):
        raise ValueError("base_seed must be an integer")
    elites = []
    for run_id in range(num_runs):
        source_seed = base_seed + run_id
        result = simulated_annealing(L, replace(parameters, seed=source_seed))
        elites.append(SAElite(result.best_a, result.best_b, result.best_score, source_seed, run_id))
    return tuple(elites)


def cyclic_aligned_pair_distance(left: SAElite, right: SAElite) -> int:
    """Minimize combined pair distance over independent rotations of one elite.

    Rotating A and B independently preserves their individual periodic
    autocorrelations, while the pair remains paired (A is never swapped with
    B).  This is a diversity metric, not a transformation of solver states.
    """
    rotated_a = [right.a[shift:] + right.a[:shift] for shift in range(len(right.a))]
    rotated_b = [right.b[shift:] + right.b[:shift] for shift in range(len(right.b))]
    return min(
        _pair_hamming(left.a, left.b, candidate_a, candidate_b)
        for candidate_a in rotated_a
        for candidate_b in rotated_b
    )


def deduplicate_elites(elites: Iterable[SAElite]) -> Tuple[SAElite, ...]:
    """Keep the best deterministic representative of each cyclic-equivalent pair."""
    chosen: Dict[BinaryPair, SAElite] = {}
    for elite in elites:
        key = (_cyclic_canonical(elite.a), _cyclic_canonical(elite.b))
        previous = chosen.get(key)
        if previous is None or _elite_order(elite) < _elite_order(previous):
            chosen[key] = elite
    return tuple(sorted(chosen.values(), key=_elite_order))


def select_diverse_elites(elites: Iterable[SAElite], elite_count: int) -> Tuple[SAElite, ...]:
    """Select low-score, cyclically diverse elites by deterministic rank aggregation.

    The first elite is the lowest-score deduplicated state.  Subsequently each
    candidate receives one rank by objective score and one rank by its minimum
    cyclic-aligned distance to selected elites.  The centralized diversity rank
    weight deliberately favors coverage after the first quality anchor, while
    avoiding arbitrary conversion between score and Hamming units.
    """
    if not isinstance(elite_count, int) or isinstance(elite_count, bool) or elite_count <= 0:
        raise ValueError("elite_count must be a positive integer")
    remaining = list(deduplicate_elites(elites))
    if not remaining:
        return ()
    selected = [remaining.pop(0)]
    while remaining and len(selected) < elite_count:
        score_rank = {elite: rank for rank, elite in enumerate(sorted(remaining, key=_elite_order))}
        distances = {elite: min(cyclic_aligned_pair_distance(elite, prior) for prior in selected)
                     for elite in remaining}
        distance_rank = {
            elite: rank for rank, elite in enumerate(
                sorted(remaining, key=lambda item: (-distances[item], _elite_order(item)))
            )
        }
        next_elite = min(
            remaining,
            key=lambda elite: (
                SCORE_RANK_WEIGHT * score_rank[elite] + DIVERSITY_RANK_WEIGHT * distance_rank[elite],
                _elite_order(elite),
            ),
        )
        selected.append(next_elite)
        remaining.remove(next_elite)
    return tuple(selected)


def build_portfolio_schedule(
    selected_elites: Sequence[SAElite],
    config: PortfolioConfig,
) -> Tuple[Tuple[SAElite, int, int], ...]:
    """Create radius 0 -> 1 -> 2 -> 3 tasks without repeated neighborhoods."""
    stage_specs = (
        (0, config.timeout_radius_0_ms,
         len(selected_elites) if config.radius_0_elite_count is None else config.radius_0_elite_count),
        (1, config.timeout_radius_1_ms,
         len(selected_elites) if config.radius_1_elite_count is None else config.radius_1_elite_count),
        (2, config.timeout_radius_2_ms, config.radius_2_elite_count),
        (3, config.timeout_radius_3_ms, config.radius_3_elite_count),
    )
    return tuple(
        (elite, radius, timeout)
        for radius, timeout, count in stage_specs
        for elite in selected_elites[:count]
    )


def run_hybrid_portfolio(
    L: int,
    collected_elites: Sequence[SAElite],
    elite_count: int,
    config: PortfolioConfig,
    solve_function: Z3SolveFunction = solve_with_z3,
    result_directory: Optional[Path] = None,
) -> HybridResult:
    """Run the bounded exact portfolio and stop at the first independently verified SAT.

    Time budget is checked before every task and used to cap its Z3 timeout.
    Solver construction may add a small wall-time overhead, but no task is
    launched once budget is exhausted.  UNSAT is neighborhood-local and
    UNKNOWN remains a distinct task status.
    """
    selected = select_diverse_elites(collected_elites, elite_count)
    started = perf_counter()
    tasks: List[PortfolioTaskResult] = []
    solution = None
    saved_path = None
    implementation_error = False
    for elite, radius, configured_timeout in build_portfolio_schedule(selected, config):
        remaining_seconds = config.total_timeout_seconds - (perf_counter() - started)
        if remaining_seconds <= 0:
            break
        task_timeout = min(configured_timeout, max(0, int(remaining_seconds * 1000)))
        if task_timeout == 0 and configured_timeout > 0:
            break
        result = solve_function(L, elite.a, elite.b, radius, task_timeout)
        verified = False
        if result.status == "SAT":
            verified = bool(result.a is not None and result.b is not None and verify_pqcp(result.a, result.b).is_valid)
            if not verified:
                implementation_error = True
        tasks.append(PortfolioTaskResult(
            elite=elite,
            radius=radius,
            timeout_ms=task_timeout,
            status=result.status,
            elapsed_time=result.elapsed_time,
            verified=verified,
            reason=result.reason,
        ))
        if verified:
            # The portfolio's independent verifier is authoritative even when
            # an injected/test result did not pre-populate its verified flag.
            solution = replace(result, verified=True)
            if result_directory is not None:
                saved_path = save_verified_solution(solution, result_directory)
            break
        if implementation_error:
            break
    return HybridResult(
        L=L,
        collected_elites=tuple(collected_elites),
        selected_elites=selected,
        tasks=tuple(tasks),
        elapsed_time=perf_counter() - started,
        sa_elapsed_time=0.0,
        portfolio_elapsed_time=perf_counter() - started,
        solved=solution is not None,
        solution=solution,
        saved_path=saved_path,
        implementation_error=implementation_error,
    )


def run_hybrid(
    L: int,
    num_sa_runs: int,
    elite_count: int,
    parameters: AnnealingParameters,
    base_seed: int,
    config: PortfolioConfig,
    solve_function: Z3SolveFunction = solve_with_z3,
    result_directory: Optional[Path] = None,
) -> HybridResult:
    """Collect deterministic SA elites then execute their bounded Z3 portfolio."""
    started = perf_counter()
    collected = collect_sa_elites(L, num_sa_runs, parameters, base_seed)
    sa_elapsed = perf_counter() - started
    portfolio_result = run_hybrid_portfolio(
        L, collected, elite_count, config, solve_function=solve_function, result_directory=result_directory
    )
    return replace(
        portfolio_result,
        elapsed_time=perf_counter() - started,
        sa_elapsed_time=sa_elapsed,
        portfolio_elapsed_time=portfolio_result.elapsed_time,
    )


def _pair_hamming(a: Sequence[int], b: Sequence[int], c: Sequence[int], d: Sequence[int]) -> int:
    """Return Hamming distance while retaining the A/B pairing relation."""
    return sum(left != right for left, right in zip(a, c)) + sum(left != right for left, right in zip(b, d))


def _cyclic_canonical(sequence: Sequence[int]) -> Tuple[int, ...]:
    """Return the lexicographically least cyclic rotation of one binary sequence."""
    word = tuple(sequence)
    return min(word[shift:] + word[:shift] for shift in range(len(word)))


def _elite_order(elite: SAElite) -> Tuple[int, int, int, Tuple[int, ...], Tuple[int, ...]]:
    """Provide deterministic quality-first ordering for equivalent choices."""
    return elite.score, elite.source_seed, elite.run_id, elite.a, elite.b
