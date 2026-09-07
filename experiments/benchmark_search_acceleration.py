"""Paired time-to-first-PQCP benchmark for the exact search hot-path optimization.

The reference path reproduces Checkpoint 3's original all-shift correlation
update followed by a full objective scan.  The optimized path is the current
``CorrelationState``.  Both are installed into otherwise identical
``SearchRunner`` instances, so RNG draws, accepted moves, solution iteration,
and independently verified A/B must match exactly.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
import statistics
import sys
from time import perf_counter
from typing import List, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from solver.checkpoint import SearchParameters
from solver.compression import CorrelationState
from solver.objective import pqcp_objective, target_pair_squared_energy
from solver.search_runner import SearchRunner
from solver.verifier import verify_pqcp
from solver.weight_constraints import canonical_weight_pairs


DEFAULT_SEEDS = (6, 24, 43, 57, 98, 100, 143, 183, 190, 196)


class ReferenceCorrelationState(CorrelationState):
    """Original exact all-shift update plus full objective recomputation."""

    @property
    def a(self) -> Tuple[int, ...]:
        """Reproduce the original full sequence reconstruction."""
        return tuple(0 if sign == 1 else 1 for sign in self._a_signs)

    @property
    def b(self) -> Tuple[int, ...]:
        """Reproduce the original full sequence reconstruction."""
        return tuple(0 if sign == 1 else 1 for sign in self._b_signs)

    @property
    def score(self) -> int:
        return pqcp_objective(self.profile)

    @property
    def target_pair_energy(self) -> int:
        """Keep the reference facade exact for trial-evaluation assertions."""
        return target_pair_squared_energy(self.profile)

    def _flip_a_unchecked(self, position: int) -> None:
        """Reproduce the original runner-to-state update path."""
        self._flip(self._a_signs, position)

    def _flip_b_unchecked(self, position: int) -> None:
        """Reproduce the original runner-to-state update path."""
        self._flip(self._b_signs, position)

    def _flip(self, signs: List[int], position: int) -> None:
        self._validate_position(position)
        length = self.L
        old_sign = signs[position]
        for shift in range(1, length):
            self._profile[shift] -= 2 * old_sign * (
                signs[(position + shift) % length]
                + signs[(position - shift) % length]
            )
        signs[position] = -old_sign


@dataclass(frozen=True)
class FirstSolution:
    """Deterministic first independently verified solution on one trajectory."""

    seed: int
    iteration: int
    a: Tuple[int, ...]
    b: Tuple[int, ...]


def run_to_first_solution(length: int, seed: int, max_steps: int, reference: bool) -> FirstSolution:
    """Return the first verifier-passing pair, or fail if the bound is insufficient."""
    weight_pairs = canonical_weight_pairs(length) if length % 2 == 0 else None
    runner = SearchRunner.new(
        length, seed,
        SearchParameters(stagnation_iterations=None, weight_pairs=weight_pairs or None),
    )
    if reference:
        # Reuse the already constructed exact state so both paths pay the same
        # FKM/correlation initialization cost; only the per-step implementation
        # differs during the timed search.
        runner._correlation.__class__ = ReferenceCorrelationState
    for _ in range(max_steps):
        outcome = runner.step()
        if reference:
            # Reproduce the pre-optimization runner's per-step MT-state copy.
            runner.state.rng_state = runner._rng.getstate()
        if outcome.verified_solution is not None:
            a, b = outcome.verified_solution
            if not verify_pqcp(a, b).is_valid:
                raise AssertionError("search reported a verifier-failing solution")
            return FirstSolution(seed, runner.state.iteration, a, b)
    raise RuntimeError("seed {} did not find a PQCP within {} steps".format(seed, max_steps))


def benchmark(length: int, seeds: Sequence[int], repeats: int, max_steps: int):
    """Measure arithmetic-mean time per solution with paired, interleaved runs.

    Reference/optimized order alternates by repeat and seed, preventing a
    consistent warm-cache or thermal-order advantage.  Every timed sample is
    one complete run to its first independently verified PQCP.
    """
    if not seeds:
        raise ValueError("at least one seed is required")
    if repeats <= 0 or max_steps <= 0:
        raise ValueError("repeats and max_steps must be positive")
    reference_times = []
    optimized_times = []
    solutions = []
    # Warm both code paths before timing to reduce one-time interpreter noise.
    run_to_first_solution(length, seeds[0], max_steps, True)
    run_to_first_solution(length, seeds[0], max_steps, False)
    for repeat_index in range(repeats):
        repeat_solutions = []
        for seed_index, seed in enumerate(seeds):
            timed = {}
            for reference in ((True, False) if (repeat_index + seed_index) % 2 == 0 else (False, True)):
                started = perf_counter()
                timed[reference] = run_to_first_solution(length, seed, max_steps, reference)
                elapsed = perf_counter() - started
                (reference_times if reference else optimized_times).append(elapsed)
            if timed[True] != timed[False]:
                raise AssertionError("optimized and reference trajectories found different first solutions")
            repeat_solutions.append(timed[False])
        solutions = repeat_solutions
    reference_mean = statistics.mean(reference_times)
    optimized_mean = statistics.mean(optimized_times)
    reduction = 100.0 * (reference_mean - optimized_mean) / reference_mean
    return {
        "reference_times": reference_times,
        "optimized_times": optimized_times,
        "reference_mean": reference_mean,
        "optimized_mean": optimized_mean,
        "reference_median": statistics.median(reference_times),
        "optimized_median": statistics.median(optimized_times),
        "mean_time_reduction_percent": reduction,
        "solutions": solutions,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--L", type=int, default=20)
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--max-steps", type=int, default=5000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    result = benchmark(args.L, seeds, args.repeats, args.max_steps)
    print("Exact paired time-to-first-PQCP benchmark")
    print("L={} seeds={} repeats={}".format(args.L, len(seeds), args.repeats))
    print("timed runs per method = {}".format(len(result["reference_times"])))
    print("reference mean = {:.6f} sec/solution".format(result["reference_mean"]))
    print("optimized mean = {:.6f} sec/solution".format(result["optimized_mean"]))
    print("mean time reduction = {:.2f}%".format(result["mean_time_reduction_percent"]))
    print("reference median = {:.6f} sec/solution".format(result["reference_median"]))
    print("optimized median = {:.6f} sec/solution".format(result["optimized_median"]))
    print("trajectory equality = PASS")
    print("verified first solutions = {}/{} per repeat".format(len(result["solutions"]), len(seeds)))
    unique = {min((solution.a, solution.b), (solution.b, solution.a)) for solution in result["solutions"]}
    print("unique verified pairs modulo A/B swap = {}".format(len(unique)))


if __name__ == "__main__":
    main()
