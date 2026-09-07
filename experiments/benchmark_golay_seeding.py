"""Paired time-to-first-new-PQCP benchmark for GCP lift versus FKM+SA.

The selected L=8 seeds yield eight different GCP-lifted pairs modulo A/B
exchange in every repeat.  Repeats stabilize arithmetic-mean timing; repeated
campaigns are not misreported as additional distinct solutions.
"""

import argparse
from pathlib import Path
import statistics
import sys
from time import perf_counter
from typing import Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from solver.checkpoint import SearchParameters
from solver.golay import complete_one_flip_each
from solver.search_runner import SearchRunner
from solver.verifier import verify_pqcp


DEFAULT_SEEDS = (0, 3, 5, 6, 11, 13, 24, 30)
BASELINE_PARAMETERS = SearchParameters(stagnation_iterations=None)


def run_fkm_to_solution(length: int, seed: int, max_steps: int):
    """Return elapsed time and the first independently verified FKM+SA pair."""
    started = perf_counter()
    runner = SearchRunner.new(length, seed, BASELINE_PARAMETERS)
    if runner.state.best_score == 0:
        pair = (runner.state.best_a, runner.state.best_b)
        if not verify_pqcp(*pair).is_valid:
            raise AssertionError("score-zero FKM initial state failed verifier")
        return perf_counter() - started, pair
    for _ in range(max_steps):
        outcome = runner.step()
        if outcome.verified_solution is not None:
            return perf_counter() - started, outcome.verified_solution
    raise RuntimeError("FKM+SA did not find a solution within max_steps")


def run_golay_to_solution(length: int, seed: int):
    """Return elapsed time and one internally verifier-checked GCP lift."""
    started = perf_counter()
    pair = complete_one_flip_each(length, seed)
    elapsed = perf_counter() - started
    if pair is None:
        raise RuntimeError("the one-flip-each GCP neighborhood contained no solution")
    return elapsed, pair


def benchmark(length: int, seeds: Sequence[int], repeats: int, max_steps: int):
    """Measure paired arithmetic-mean time and enforce per-repeat novelty."""
    if not seeds or repeats <= 0 or max_steps <= 0:
        raise ValueError("seeds, repeats, and max_steps must be positive")
    # Warm both paths outside timed samples.
    run_fkm_to_solution(length, seeds[0], max_steps)
    run_golay_to_solution(length, seeds[0])
    fkm_times = []
    golay_times = []
    golay_unique_per_repeat = []
    for repeat in range(repeats):
        golay_pairs = set()
        for seed_index, seed in enumerate(seeds):
            timed = {}
            order = (False, True) if (repeat + seed_index) % 2 == 0 else (True, False)
            for golay in order:
                elapsed, pair = (
                    run_golay_to_solution(length, seed)
                    if golay else run_fkm_to_solution(length, seed, max_steps)
                )
                if not verify_pqcp(*pair).is_valid:
                    raise AssertionError("timed method returned a verifier-failing pair")
                (golay_times if golay else fkm_times).append(elapsed)
                timed[golay] = pair
            golay_pair = timed[True]
            golay_pairs.add(min(golay_pair, (golay_pair[1], golay_pair[0])))
        if len(golay_pairs) != len(seeds):
            raise AssertionError("GCP benchmark seeds did not produce distinct new pairs")
        golay_unique_per_repeat.append(len(golay_pairs))
    fkm_mean = statistics.mean(fkm_times)
    golay_mean = statistics.mean(golay_times)
    return {
        "fkm_times": fkm_times,
        "golay_times": golay_times,
        "fkm_mean": fkm_mean,
        "golay_mean": golay_mean,
        "time_reduction_percent": 100.0 * (fkm_mean - golay_mean) / fkm_mean,
        "unique_per_repeat": tuple(golay_unique_per_repeat),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--L", type=int, default=8)
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--repeats", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=1000)
    args = parser.parse_args()
    seeds = tuple(int(seed) for seed in args.seeds.split(",") if seed.strip())
    result = benchmark(args.L, seeds, args.repeats, args.max_steps)
    print("GCP seeding time-to-first-new-PQCP benchmark")
    print("L={} seeds={} repeats={} timed runs/method={}".format(
        args.L, len(seeds), args.repeats, len(result["fkm_times"])
    ))
    print("FKM+SA mean = {:.9f} sec/solution".format(result["fkm_mean"]))
    print("GCP lift mean = {:.9f} sec/solution".format(result["golay_mean"]))
    print("mean time reduction = {:.2f}%".format(result["time_reduction_percent"]))
    print("distinct GCP pairs per repeat = {}".format(min(result["unique_per_repeat"])))


if __name__ == "__main__":
    main()
