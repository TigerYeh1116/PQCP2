"""Diagnostic analysis of the SA-to-Z3 interface; this is not a solver.

It studies whether small Hamming neighborhoods are empirically justified,
how independent SA elites are distributed, bit-flip objective sensitivity, and
the size/timing of the existing exact Z3 neighborhood encoding.
"""

from collections import Counter, deque
from itertools import combinations
from pathlib import Path
import statistics
import sys
from time import perf_counter
from typing import Dict, Iterable, List, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import z3

from solver.annealing import AnnealingParameters, AnnealingResult, simulated_annealing
from solver.compression import CorrelationState
from solver.correlation import full_correlation_profile
from solver.objective import pqcp_objective
from solver.z3_solver import _pair_correlation


PROJECT_LENGTHS = (44, 46, 58, 68)
SMALL_LENGTHS = (4, 6, 8)
ELITE_RUNS = 20
# Keep 20 independent elites while allowing the full four-length diagnostic to
# finish as one bounded CPU-only experiment.
ELITE_ITERATIONS = 750
CONSTRAINT_TIMEOUT_MS = 1_000


def combined_hamming(a: Sequence[int], b: Sequence[int], c: Sequence[int], d: Sequence[int]) -> int:
    """Return Hamming distance over the concatenated A/B pair representation."""
    return sum(left != right for left, right in zip(a, c)) + sum(left != right for left, right in zip(b, d))


def cyclic_aligned_pair_distance(a: Sequence[int], b: Sequence[int], c: Sequence[int], d: Sequence[int]) -> int:
    """Minimize pair Hamming distance over independent cyclic rotations of C and D.

    Independent rotations preserve each sequence's periodic autocorrelation, so
    this removes a representational effect before interpreting elite distance.
    It is descriptive only; no SA state is canonicalized here.
    """
    a_word, b_word, c_word, d_word = tuple(a), tuple(b), tuple(c), tuple(d)
    c_rotations = [c_word[shift:] + c_word[:shift] for shift in range(len(c_word))]
    d_rotations = [d_word[shift:] + d_word[:shift] for shift in range(len(d_word))]
    return min(
        combined_hamming(a_word, b_word, rotated_c, rotated_d)
        for rotated_c in c_rotations
        for rotated_d in d_rotations
    )


def exhaustive_distance_analysis(length: int) -> Dict[str, object]:
    """Exhaustively measure nearest-target distance for all pairs at a small L.

    A multi-source BFS on the 2L-dimensional Hamming cube gives the exact
    nearest legal-target distance for every pair, avoiding any sampled or
    assumed solution-to-near-state correspondence.
    """
    state_count = 1 << (2 * length)
    scores: List[int] = []
    solutions: List[int] = []
    for code in range(state_count):
        a = tuple((code >> index) & 1 for index in range(length))
        b = tuple((code >> (length + index)) & 1 for index in range(length))
        score = pqcp_objective(full_correlation_profile(a, b))
        scores.append(score)
        if score == 0:
            solutions.append(code)

    distances = _nearest_solution_distances(state_count, 2 * length, solutions)
    positive_scores = [score for score in scores if score > 0]
    minimum_positive = min(positive_scores)
    near_codes = [code for code, score in enumerate(scores) if score == minimum_positive]
    histogram = Counter(distances[code] for code in near_codes)
    return {
        "L": length,
        "states": state_count,
        "solutions": len(solutions),
        "minimum_positive_score": minimum_positive,
        "near_state_count": len(near_codes),
        "nearest_distance_histogram": dict(sorted(histogram.items())),
    }


def _nearest_solution_distances(state_count: int, dimension: int, sources: Iterable[int]) -> List[int]:
    """Compute exact nearest-source Hamming distance by multi-source BFS."""
    distances = [-1] * state_count
    frontier = deque()
    for source in sources:
        distances[source] = 0
        frontier.append(source)
    while frontier:
        current = frontier.popleft()
        for bit in range(dimension):
            neighbor = current ^ (1 << bit)
            if distances[neighbor] == -1:
                distances[neighbor] = distances[current] + 1
                frontier.append(neighbor)
    return distances


def standard_sa_center(length: int) -> AnnealingResult:
    """Reproduce the fixed Checkpoint 5 center used for hybrid diagnostics."""
    return simulated_annealing(
        length,
        AnnealingParameters(
            max_iterations=25_000,
            initial_temperature=8.0,
            cooling_rate=0.9995,
            min_temperature=0.05,
            seed=20_260_825,
            restart_count=2,
            fkm_pool_size=128,
        ),
    )


def collect_elites(length: int) -> List[AnnealingResult]:
    """Collect deterministic independent SA best states without altering SA code."""
    results = []
    for run_index in range(ELITE_RUNS):
        results.append(simulated_annealing(
            length,
            AnnealingParameters(
                max_iterations=ELITE_ITERATIONS,
                initial_temperature=8.0,
                cooling_rate=0.999,
                min_temperature=0.05,
                seed=20_260_900 + length * 100 + run_index,
                restart_count=1,
                fkm_pool_size=128,
            ),
        ))
    return sorted(results, key=lambda result: (result.best_score, result.best_a, result.best_b))


def elite_statistics(elites: Sequence[AnnealingResult], count: int) -> Dict[str, object]:
    """Summarize score and raw/cyclic-aligned pairwise elite dispersion."""
    selected = list(elites[:min(count, len(elites))])
    raw = []
    aligned = []
    for left, right in combinations(selected, 2):
        raw.append(combined_hamming(left.best_a, left.best_b, right.best_a, right.best_b))
        aligned.append(cyclic_aligned_pair_distance(left.best_a, left.best_b, right.best_a, right.best_b))
    unique_states = len({(result.best_a, result.best_b) for result in selected})
    return {
        "K": len(selected),
        "scores": tuple(result.best_score for result in selected),
        "unique_states": unique_states,
        "raw": _distance_summary(raw),
        "cyclic_aligned": _distance_summary(aligned),
    }


def _distance_summary(distances: Sequence[int]) -> Dict[str, float]:
    """Return compact descriptive statistics for a nonempty pairwise sample."""
    if not distances:
        return {"minimum": 0, "median": 0, "mean": 0, "maximum": 0}
    return {
        "minimum": min(distances),
        "median": statistics.median(distances),
        "mean": statistics.mean(distances),
        "maximum": max(distances),
    }


def sensitivity_analysis(a: Sequence[int], b: Sequence[int]) -> Dict[str, object]:
    """Measure exact one-flip objective deltas around one complete SA center.

    These deltas are heuristic diagnostics only.  They do not prove a bit may
    be fixed, so they are never fed into safe pruning.
    """
    state = CorrelationState(a, b)
    current_score = pqcp_objective(state.profile)
    deltas = []
    for sequence_name in ("A", "B"):
        for position in range(state.L):
            if sequence_name == "A":
                state.flip_a(position)
            else:
                state.flip_b(position)
            delta = pqcp_objective(state.profile) - current_score
            if sequence_name == "A":
                state.flip_a(position)
            else:
                state.flip_b(position)
            deltas.append((sequence_name, position, delta))
    by_absolute = sorted(deltas, key=lambda item: (-abs(item[2]), item[0], item[1]))
    low = [item for item in deltas if item[2] == 0]
    return {
        "score": current_score,
        "zero_delta_count": len(low),
        "total_bits": len(deltas),
        "most_sensitive": tuple(by_absolute[:5]),
        "least_sensitive": tuple(sorted(deltas, key=lambda item: (abs(item[2]), item[0], item[1]))[:5]),
        "delta_minimum": min(item[2] for item in deltas),
        "delta_maximum": max(item[2] for item in deltas),
    }


def z3_constraint_statistics(length: int, a: Sequence[int], b: Sequence[int], radius: int = 3) -> Dict[str, object]:
    """Measure construction and solve time for the current exact neighborhood model.

    This mirrors the existing ``solve_with_z3`` constraints solely to report
    encoding size and timing; it does not expose another solver API.
    """
    started = perf_counter()
    solver = z3.Solver()
    solver.set(timeout=CONSTRAINT_TIMEOUT_MS)
    a_vars = [z3.Int("analysis_a_{}_{}".format(length, index)) for index in range(length)]
    b_vars = [z3.Int("analysis_b_{}_{}".format(length, index)) for index in range(length)]
    for variable in a_vars + b_vars:
        solver.add(z3.Or(variable == 0, variable == 1))
    solver.add(z3.Sum(
        [z3.If(variable != bit, 1, 0) for variable, bit in zip(a_vars, a)]
        + [z3.If(variable != bit, 1, 0) for variable, bit in zip(b_vars, b)]
    ) <= radius)
    solver.add(_pair_correlation(a_vars, b_vars, 0, length) == 2 * length)
    indicators = []
    for shift in range(1, length // 2 + 1):
        correlation_sum = _pair_correlation(a_vars, b_vars, shift, length)
        solver.add(z3.Or(correlation_sum == -4, correlation_sum == 0, correlation_sum == 4))
        indicator = z3.Bool("analysis_nonzero_{}_{}".format(length, shift))
        solver.add(indicator == (correlation_sum != 0))
        actual_weight = 1 if length % 2 == 0 and shift == length // 2 else 2
        indicators.append(z3.If(indicator, actual_weight, 0))
    solver.add(z3.Sum(indicators) == 2)
    source_assertions = len(solver.assertions())
    construction_time = perf_counter() - started
    solve_started = perf_counter()
    outcome = solver.check()
    solve_time = perf_counter() - solve_started
    return {
        "bit_variables": 2 * length,
        "auxiliary_variables": length // 2,
        "correlation_expressions": 1 + length // 2,
        "correlation_membership_constraints": length // 2,
        "indicator_constraints": length // 2,
        "hamming_constraints": 1,
        "source_assertions": source_assertions,
        "construction_time": construction_time,
        "solve_time": solve_time,
        "status": str(outcome).upper(),
        "reason": solver.reason_unknown() if outcome == z3.unknown else None,
    }


def main() -> None:
    """Print concise, reproducible Hybrid SA -> Z3 diagnostic results."""
    print("Hybrid SA -> Z3 interface analysis")
    print("=" * 72)
    print("Small-L exhaustive distance analysis")
    for length in SMALL_LENGTHS:
        report = exhaustive_distance_analysis(length)
        print("L={L} states={states} solutions={solutions} min_positive_score={minimum_positive_score} "
              "near_states={near_state_count} nearest_distance_histogram={nearest_distance_histogram}".format(**report))

    print("\nProject-length SA centers, elites, sensitivity, and Z3 size")
    for length in PROJECT_LENGTHS:
        center = standard_sa_center(length)
        print("\nL={} center_score={} center_time={:.3f}s".format(
            length, center.best_score, center.elapsed_time
        ))
        elites = collect_elites(length)
        for count in (5, 10, 20):
            report = elite_statistics(elites, count)
            print("elites K={K} scores={scores} unique={unique_states} raw={raw} aligned={cyclic_aligned}".format(
                **report
            ))
        sensitivity = sensitivity_analysis(center.best_a, center.best_b)
        print("sensitivity score={score} zero_delta={zero_delta_count}/{total_bits} min_delta={delta_minimum} "
              "max_delta={delta_maximum} high={most_sensitive} low={least_sensitive}".format(**sensitivity))
        constraints = z3_constraint_statistics(length, center.best_a, center.best_b)
        print("z3 bits={bit_variables} auxiliary={auxiliary_variables} corr_expr={correlation_expressions} "
              "membership={correlation_membership_constraints} indicators={indicator_constraints} "
              "hamming={hamming_constraints} source_assertions={source_assertions} "
              "build={construction_time:.4f}s "
              "solve={solve_time:.4f}s status={status} reason={reason}".format(**constraints))


if __name__ == "__main__":
    main()
