"""Fair wall-clock ablation for methods ported from the reference project.

Examples::

    python experiments/benchmark_reference_port.py --lengths 8 44 --seconds 5

Initialization is charged against each run's wall-clock budget.  The script
never writes a discovered pair to ``L.txt``; it is a measurement tool only.
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path
from statistics import mean, median
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from solver.correlation import full_correlation_profile
from solver.objective import pqcp_objective
from solver.reference_search import (
    legacy_search_parameters,
    reference_search_parameters,
)
from solver.search_runner import SearchRunner
from solver.verifier import verify_pqcp


METHODS = (
    "legacy", "compressed_seed", "reference_seed", "multiscale",
    "reference_full", "reference",
)


def method_parameters(length, method):
    """Return one controlled ablation configuration."""
    legacy = legacy_search_parameters(length)
    if method == "legacy":
        return legacy
    if method == "compressed_seed":
        return replace(legacy, fkm_seed_policy="compressed_top_q")
    if method == "reference_seed":
        return replace(
            legacy, fkm_seed_policy="compressed_a_random_b_top_q",
            randomize_target_profiles=True,
        )
    if method == "multiscale":
        return replace(legacy, acceptance_mode="fixed_target_multiscale")
    if method == "reference_full":
        return replace(
            reference_search_parameters(length),
            acceptance_mode="fixed_target_full",
        )
    if method == "reference":
        return reference_search_parameters(length)
    raise ValueError("unknown method: {}".format(method))


def run_one(length, seed, seconds, method):
    """Run one method while charging initialization to the shared budget."""
    started = perf_counter()
    runner = SearchRunner.new(length, seed, method_parameters(length, method))
    initialized = perf_counter()
    initial_score = runner.state.current_score
    remaining = max(0.0, seconds - (initialized - started))
    first_solution_seconds = None

    def record_solution(_a, _b, _state):
        nonlocal first_solution_seconds
        if first_solution_seconds is None:
            first_solution_seconds = perf_counter() - started

    summary = runner.run(seconds=remaining, on_verified_solution=record_solution)
    elapsed = perf_counter() - started
    profile = tuple(full_correlation_profile(
        summary.state.best_a, summary.state.best_b
    ))
    verification = verify_pqcp(summary.state.best_a, summary.state.best_b)
    if pqcp_objective(profile) != summary.state.best_score:
        raise RuntimeError("benchmark best score failed exact recomputation")
    if verification.profile != profile:
        raise RuntimeError("benchmark verifier profile mismatch")
    return {
        "L": length,
        "method": method,
        "seed": seed,
        "budget_seconds": seconds,
        "elapsed": elapsed,
        "initialization_seconds": initialized - started,
        "iterations": summary.state.iteration,
        "restarts": summary.state.restart_index,
        "initial_score": initial_score,
        "best_score": summary.state.best_score,
        "verified": verification.is_valid,
        "found_verified_solution": first_solution_seconds is not None,
        "first_solution_seconds": first_solution_seconds,
        "iterations_per_second": summary.state.iteration / elapsed if elapsed else 0.0,
    }


def aggregate(records):
    """Summarize score, throughput, initialization, and paired outcomes."""
    groups = {}
    for record in records:
        groups.setdefault((record["L"], record["method"]), []).append(record)
    summaries = []
    for (length, method), group in sorted(groups.items()):
        scores = [record["best_score"] for record in group]
        summaries.append({
            "L": length,
            "method": method,
            "runs": len(group),
            "minimum": min(scores),
            "median": median(scores),
            "mean": mean(scores),
            "maximum": max(scores),
            "verified": sum(record["verified"] for record in group),
            "runs_finding_solution": sum(
                record.get("found_verified_solution", record["verified"])
                for record in group
            ),
            "mean_iterations_per_second": mean(
                record["iterations_per_second"] for record in group
            ),
            "mean_initialization_seconds": mean(
                record["initialization_seconds"] for record in group
            ),
        })
    paired = []
    baseline = {
        (record["L"], record["seed"]): record["best_score"]
        for record in records if record["method"] == "legacy"
    }
    for method in METHODS[1:]:
        values = [
            (baseline[(record["L"], record["seed"])], record["best_score"])
            for record in records if record["method"] == method
            and (record["L"], record["seed"]) in baseline
        ]
        paired.append({
            "method": method,
            "wins": sum(candidate < old for old, candidate in values),
            "ties": sum(candidate == old for old, candidate in values),
            "losses": sum(candidate > old for old, candidate in values),
        })
    return {"summaries": summaries, "paired_vs_legacy": paired}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", type=int, nargs="+", default=(8, 44, 46))
    parser.add_argument("--seeds", type=int, nargs="+", default=(123, 456, 789, 1024, 2026))
    parser.add_argument("--methods", choices=METHODS, nargs="+", default=METHODS)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--output", type=Path, default=Path("results/reference_port_benchmark.json"))
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.seconds <= 0:
        raise ValueError("--seconds must be positive")
    records = []
    for length in args.lengths:
        for seed in args.seeds:
            for method in args.methods:
                record = run_one(length, seed, args.seconds, method)
                records.append(record)
                print(
                    "L={} seed={} {:15s} initial={} best={} iter/s={:.0f} verified={}".format(
                        length, seed, method, record["initial_score"],
                        record["best_score"], record["iterations_per_second"],
                        record["verified"],
                    ),
                    flush=True,
                )
    payload = {"records": records, **aggregate(records)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("\nSummary")
    for row in payload["summaries"]:
        print(
            "L={L} {method:15s} min={minimum} median={median} mean={mean:.2f} "
            "max={maximum} verified={verified}/{runs} iter/s={mean_iterations_per_second:.0f}".format(**row)
        )
    print("Paired vs legacy: {}".format(payload["paired_vs_legacy"]))
    print("Saved: {}".format(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
