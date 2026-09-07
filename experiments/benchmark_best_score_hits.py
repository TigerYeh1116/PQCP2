"""Paired benchmark for the frequency of reaching Project objective score < 4.

One hit is counted per independent deterministic seed if its run reaches
``best_score < 4`` within the identical wall-clock budget.  Initialization is
inside the budget.  The baseline and guided methods differ only in the SA
acceptance energy and its matching initial temperature; reported best scores
remain the unchanged Project objective.
"""

import argparse
from pathlib import Path
import statistics
import sys
from time import perf_counter
from typing import Dict, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from solver.checkpoint import SearchParameters
from solver.search_runner import SearchRunner
from solver.verifier import verify_pqcp
from solver.weight_constraints import canonical_weight_pairs


def run_one(length: int, seed: int, seconds: float, guided: bool) -> Dict[str, object]:
    """Run one method to its first score<4 hit or the wall-clock deadline."""
    parameters = SearchParameters(
        weight_pairs=canonical_weight_pairs(length),
        stagnation_iterations=None,
        acceptance_mode="objective_plus_target_pair" if guided else "objective",
        initial_temperature=8.0,
    )
    started = perf_counter()
    runner = SearchRunner.new(length, seed, parameters)
    while perf_counter() - started < seconds and runner.state.best_score >= 4:
        runner.step()
    elapsed = perf_counter() - started
    verified = False
    if runner.state.best_score == 0:
        verified = verify_pqcp(runner.state.best_a, runner.state.best_b).is_valid
        if not verified:
            raise AssertionError("score-zero benchmark candidate failed independent verifier")
    return {
        "seed": seed,
        "guided": guided,
        "hit": runner.state.best_score < 4,
        "best_score": runner.state.best_score,
        "iterations": runner.state.iteration,
        "elapsed": elapsed,
        "verified": verified,
    }


def benchmark(length: int, seeds: Sequence[int], repeats: int, seconds: float):
    """Interleave paired methods and return exact hit-count improvement."""
    if not seeds or repeats <= 0 or seconds <= 0:
        raise ValueError("seeds, repeats, and seconds must be positive")
    records = []
    # Warm imports and both code paths outside timed samples.
    run_one(length, seeds[0], seconds, False)
    run_one(length, seeds[0], seconds, True)
    for repeat in range(repeats):
        for seed_index, seed in enumerate(seeds):
            order = (False, True) if (repeat + seed_index) % 2 == 0 else (True, False)
            for guided in order:
                record = run_one(length, seed, seconds, guided)
                record["repeat"] = repeat
                records.append(record)
    baseline = [record for record in records if not record["guided"]]
    guided = [record for record in records if record["guided"]]
    baseline_hits = sum(bool(record["hit"]) for record in baseline)
    guided_hits = sum(bool(record["hit"]) for record in guided)
    if baseline_hits == 0:
        raise RuntimeError("baseline produced zero hits; percentage improvement is undefined")
    return {
        "records": records,
        "baseline_hits": baseline_hits,
        "guided_hits": guided_hits,
        "hit_factor": guided_hits / baseline_hits,
        "hit_increase_percent": 100.0 * (guided_hits - baseline_hits) / baseline_hits,
        "baseline_mean_iterations": statistics.mean(record["iterations"] for record in baseline),
        "guided_mean_iterations": statistics.mean(record["iterations"] for record in guided),
        "baseline_verified": sum(bool(record["verified"]) for record in baseline),
        "guided_verified": sum(bool(record["verified"]) for record in guided),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--L", type=int, default=24)
    parser.add_argument("--seed-count", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seconds-per-run", type=float, default=0.018)
    args = parser.parse_args()
    result = benchmark(args.L, tuple(range(args.seed_count)), args.repeats, args.seconds_per_run)
    print("Paired best_score < 4 hit benchmark")
    print("L={} seeds={} repeats={} seconds/run={}".format(
        args.L, args.seed_count, args.repeats, args.seconds_per_run
    ))
    print("baseline hits = {}".format(result["baseline_hits"]))
    print("guided hits = {}".format(result["guided_hits"]))
    print("hit factor = {:.3f}x".format(result["hit_factor"]))
    print("hit increase = {:.2f}%".format(result["hit_increase_percent"]))
    print("mean iterations baseline/guided = {:.1f}/{:.1f}".format(
        result["baseline_mean_iterations"], result["guided_mean_iterations"]
    ))
    print("verified score-zero baseline/guided = {}/{}".format(
        result["baseline_verified"], result["guided_verified"]
    ))


if __name__ == "__main__":
    main()
