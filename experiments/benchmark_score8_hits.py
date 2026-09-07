"""Paired wall-clock benchmark for reaching Project objective score <= 8.

The two methods use identical L, seed, FKM pool, legal weight pair, objective,
temperature schedule, and restart controls.  The optimized method changes
only two SA navigation controls while leaving the reported Project objective
unchanged:

* baseline: one randomly selected swap;
* optimized: choose the lowest-energy swap among a fixed-size random sample
  and give the exact Project objective weight 3 instead of 2 in the existing
  auxiliary navigation energy.

FKM initialization is performed before the timed interval because the change
targets the repeated SA step used by multi-hour runs.  Initial A/B equality is
recorded and checked for every paired seed.  One hit means that an independent
run first reaches ``best_score <= 8`` within the same search-phase wall budget.
"""

import argparse
import json
from pathlib import Path
import statistics
import sys
from time import perf_counter
from typing import Dict, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from solver.checkpoint import SearchParameters
from solver.search_runner import (
    DEFAULT_GUIDED_OBJECTIVE_ENERGY_WEIGHT,
    DEFAULT_GUIDED_PROPOSAL_SAMPLES,
    SearchRunner,
)
from solver.verifier import verify_pqcp
from solver.weight_constraints import canonical_weight_pairs


TARGET_SCORE = 8


def run_one(
    length: int,
    seed: int,
    seconds: float,
    proposal_samples: int,
) -> Dict[str, object]:
    """Run one deterministic initialization under a strict search-time budget."""
    parameters = SearchParameters(
        weight_pairs=canonical_weight_pairs(length),
        stagnation_iterations=None,
        acceptance_mode="objective_plus_target_pair",
        initial_temperature=8.0,
        proposal_samples=proposal_samples,
        # The one-proposal side reproduces the pre-change navigation energy;
        # the optimized side uses the measured production setting.
        objective_energy_weight=(
            2 if proposal_samples == 1 else DEFAULT_GUIDED_OBJECTIVE_ENERGY_WEIGHT
        ),
    )
    runner = SearchRunner.new(length, seed, parameters)
    initial_a = runner.state.current_a
    initial_b = runner.state.current_b
    initial_score = runner.state.best_score
    started = perf_counter()
    reached_at = 0.0 if initial_score <= TARGET_SCORE else None
    while perf_counter() - started < seconds and runner.state.best_score > TARGET_SCORE:
        runner.step()
        if runner.state.best_score <= TARGET_SCORE:
            reached_at = perf_counter() - started
    elapsed = perf_counter() - started
    verified = False
    if runner.state.best_score == 0:
        verified = verify_pqcp(runner.state.best_a, runner.state.best_b).is_valid
        if not verified:
            raise AssertionError("score-zero benchmark candidate failed independent verifier")
    return {
        "seed": seed,
        "proposal_samples": proposal_samples,
        "objective_energy_weight": parameters.objective_energy_weight,
        "initial_a": list(initial_a),
        "initial_b": list(initial_b),
        "initial_score": initial_score,
        "hit": runner.state.best_score <= TARGET_SCORE,
        "time_to_hit": reached_at,
        "best_score": runner.state.best_score,
        "iterations": runner.state.iteration,
        "elapsed": elapsed,
        "verified": verified,
    }


def aggregate_records(records: Sequence[Dict[str, object]], optimized_samples: int) -> Dict[str, object]:
    """Aggregate paired records and calculate the literal percentage increase."""
    baseline = [record for record in records if record["proposal_samples"] == 1]
    optimized = [record for record in records if record["proposal_samples"] == optimized_samples]
    baseline_by_key = {(record["repeat"], record["seed"]): record for record in baseline}
    optimized_by_key = {(record["repeat"], record["seed"]): record for record in optimized}
    if not baseline or set(baseline_by_key) != set(optimized_by_key):
        raise ValueError("records must contain one complete baseline/optimized pair per repeat and seed")
    baseline_hits = sum(bool(record["hit"]) for record in baseline)
    optimized_hits = sum(bool(record["hit"]) for record in optimized)
    if baseline_hits == 0:
        raise RuntimeError("baseline produced zero hits; a percentage increase cannot be established")
    initial_mismatches = sum(
        (
            baseline_by_key[key]["initial_a"] != optimized_by_key[key]["initial_a"]
            or baseline_by_key[key]["initial_b"] != optimized_by_key[key]["initial_b"]
            or baseline_by_key[key]["initial_score"] != optimized_by_key[key]["initial_score"]
        )
        for key in baseline_by_key
    )
    if initial_mismatches:
        raise AssertionError("paired methods did not receive identical FKM initial states")
    wins = ties = losses = 0
    for key, baseline_record in baseline_by_key.items():
        optimized_score = int(optimized_by_key[key]["best_score"])
        baseline_score = int(baseline_record["best_score"])
        if optimized_score < baseline_score:
            wins += 1
        elif optimized_score == baseline_score:
            ties += 1
        else:
            losses += 1
    return {
        "baseline_hits": baseline_hits,
        "optimized_hits": optimized_hits,
        "hit_factor": optimized_hits / baseline_hits,
        "hit_increase_percent": 100.0 * (optimized_hits - baseline_hits) / baseline_hits,
        "paired_score_wins": wins,
        "paired_score_ties": ties,
        "paired_score_losses": losses,
        "baseline_mean_best_score": statistics.mean(int(record["best_score"]) for record in baseline),
        "optimized_mean_best_score": statistics.mean(int(record["best_score"]) for record in optimized),
        "baseline_mean_iterations": statistics.mean(int(record["iterations"]) for record in baseline),
        "optimized_mean_iterations": statistics.mean(int(record["iterations"]) for record in optimized),
        "baseline_verified": sum(bool(record["verified"]) for record in baseline),
        "optimized_verified": sum(bool(record["verified"]) for record in optimized),
        "initial_mismatches": initial_mismatches,
    }


def benchmark(
    length: int,
    seeds: Sequence[int],
    repeats: int,
    seconds: float,
    optimized_samples: int = DEFAULT_GUIDED_PROPOSAL_SAMPLES,
) -> Dict[str, object]:
    """Interleave paired methods to reduce order-dependent timing bias."""
    if not seeds or repeats <= 0 or seconds <= 0:
        raise ValueError("seeds, repeats, and seconds must be positive")
    if optimized_samples <= 1:
        raise ValueError("optimized_samples must exceed the one-sample baseline")
    records = []
    for repeat in range(repeats):
        for seed_index, seed in enumerate(seeds):
            order = (1, optimized_samples) if (repeat + seed_index) % 2 == 0 else (optimized_samples, 1)
            for proposal_samples in order:
                record = run_one(length, seed, seconds, proposal_samples)
                record["repeat"] = repeat
                record["method"] = "baseline" if proposal_samples == 1 else "optimized"
                records.append(record)
    return {
        "L": length,
        "target_score": TARGET_SCORE,
        "seconds_per_run": seconds,
        "seed_count": len(seeds),
        "repeats": repeats,
        "optimized_proposal_samples": optimized_samples,
        "records": records,
        **aggregate_records(records, optimized_samples),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--L", type=int, default=32)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-count", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seconds-per-run", type=float, default=0.01)
    parser.add_argument("--optimized-samples", type=int, default=DEFAULT_GUIDED_PROPOSAL_SAMPLES)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    seeds = tuple(range(args.seed_start, args.seed_start + args.seed_count))
    result = benchmark(args.L, seeds, args.repeats, args.seconds_per_run, args.optimized_samples)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("Paired best_score <= 8 benchmark")
    print("L={} unique seeds={} repeats={} seconds/run={}".format(
        args.L, args.seed_count, args.repeats, args.seconds_per_run
    ))
    print("proposal samples baseline/optimized = 1/{}".format(args.optimized_samples))
    print("hits baseline/optimized = {}/{}".format(result["baseline_hits"], result["optimized_hits"]))
    print("hit factor = {:.3f}x".format(result["hit_factor"]))
    print("hit increase = {:.2f}%".format(result["hit_increase_percent"]))
    print("paired score wins/ties/losses = {}/{}/{}".format(
        result["paired_score_wins"], result["paired_score_ties"], result["paired_score_losses"]
    ))
    print("mean best score baseline/optimized = {:.3f}/{:.3f}".format(
        result["baseline_mean_best_score"], result["optimized_mean_best_score"]
    ))
    print("mean iterations baseline/optimized = {:.1f}/{:.1f}".format(
        result["baseline_mean_iterations"], result["optimized_mean_iterations"]
    ))


if __name__ == "__main__":
    main()
