"""Paired time-to-first-verified-PQCP benchmark for independent SA workers.

The default length 28 is deliberately smaller than the official Project 2
instances: it uses the identical correlation target, exact content identities,
fixed-target energy, same-parity swaps, incremental evaluator, and independent
verifier, while producing enough real solutions for a practical development
benchmark.  No planted or known solution is supplied to any worker.
"""

import argparse
import json
import multiprocessing as mp
from pathlib import Path
import random
from statistics import mean, median
import sys
from time import perf_counter
import traceback
from typing import Dict, List, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from solver.checkpoint import SearchParameters
from solver.search_runner import SearchRunner
from solver.target_profiles import canonical_target_content_profiles
from solver.verifier import verify_pqcp
from experiments.benchmark_parallel_search import (
    counterbalanced_counts,
    worker_seed,
)


def encoded_profiles(length: int) -> Tuple[Tuple[int, int, int, int, int, int], ...]:
    """Return the same canonical exact target/content profiles as production."""
    return tuple(
        (
            profile.k, profile.eta,
            profile.a_even_ones, profile.a_odd_ones,
            profile.b_even_ones, profile.b_odd_ones,
        )
        for profile in canonical_target_content_profiles(length)
    )


def _solution_worker(
    length: int,
    seed: int,
    seconds: float,
    worker_index: int,
    start_event: object,
    stop_event: object,
    ready_queue: object,
    result_queue: object,
) -> None:
    """Search an ordinary FKM/content start; no known answer is available."""
    try:
        profiles = encoded_profiles(length)
        if not profiles:
            raise ValueError("length has no exact target/content profiles")
        ready_queue.put(worker_index)
        start_event.wait()
        started = perf_counter()
        runner = SearchRunner.new(
            length,
            seed,
            SearchParameters(
                stagnation_iterations=100_000,
                target_content_profiles=profiles,
                preserve_alternating_content=True,
                acceptance_mode="fixed_target_full",
                proposal_samples=10,
            ),
        )
        solved = False
        solution = None
        if runner.state.current_score == 0:
            verification = verify_pqcp(runner.state.current_a, runner.state.current_b)
            if not verification.is_valid:
                raise RuntimeError("initial score-zero state failed verifier")
            solved = True
            solution = (runner.state.current_a, runner.state.current_b)
        while not solved and perf_counter() - started < seconds:
            # Poll only once per block so coordination does not become a hot-
            # path operation that changes the measured SA throughput.
            if stop_event.is_set():
                break
            for _ in range(32):
                outcome = runner.step()
                if outcome.verified_solution is not None:
                    solution = outcome.verified_solution
                    if not verify_pqcp(*solution).is_valid:
                        raise RuntimeError("reported solution failed independent verifier")
                    solved = True
                    break
                if perf_counter() - started >= seconds:
                    break
        elapsed = min(perf_counter() - started, seconds)
        if solved:
            stop_event.set()
        result_queue.put({
            "ok": True,
            "worker": worker_index,
            "seed": seed,
            "solved": solved,
            "elapsed": elapsed,
            "iterations": runner.state.iteration,
            "score": runner.state.best_score,
            "A": list(solution[0]) if solution is not None else None,
            "B": list(solution[1]) if solution is not None else None,
        })
    except BaseException as error:  # pragma: no cover - child error plumbing
        stop_event.set()
        result_queue.put({
            "ok": False,
            "worker": worker_index,
            "seed": seed,
            "error": "{}: {}".format(type(error).__name__, error),
            "traceback": traceback.format_exc(),
        })


def run_trial(length: int, base_seed: int, seconds: float, workers: int) -> Dict[str, object]:
    """Return first verified discovery under one synchronized worker portfolio."""
    if seconds <= 0:
        raise ValueError("seconds must be positive")
    if not isinstance(workers, int) or isinstance(workers, bool) or workers <= 0:
        raise ValueError("workers must be a positive integer")
    context = mp.get_context("spawn")
    start_event = context.Event()
    stop_event = context.Event()
    ready_queue = context.Queue()
    result_queue = context.Queue()
    processes = []
    for worker_index in range(workers):
        process = context.Process(
            target=_solution_worker,
            args=(
                length, worker_seed(base_seed, worker_index), seconds,
                worker_index, start_event, stop_event, ready_queue, result_queue,
            ),
            name="pqcp-solution-w{:02d}".format(worker_index),
        )
        process.start()
        processes.append(process)
    for _ in processes:
        ready_queue.get()
    parent_started = perf_counter()
    start_event.set()
    rows = [result_queue.get() for _ in processes]
    for process in processes:
        process.join()
    parent_wall = perf_counter() - parent_started
    failures = [row for row in rows if not row.get("ok")]
    if failures:
        raise RuntimeError("solution worker failed: {}".format(failures[0]))
    rows.sort(key=lambda row: int(row["worker"]))
    solved_rows = [row for row in rows if row["solved"]]
    discovery = min(solved_rows, key=lambda row: float(row["elapsed"])) if solved_rows else None
    if discovery is not None:
        if not verify_pqcp(tuple(discovery["A"]), tuple(discovery["B"])).is_valid:
            raise RuntimeError("parent verification failed")
    return {
        "L": length,
        "base_seed": base_seed,
        "seconds": seconds,
        "workers": workers,
        "solved": discovery is not None,
        "time_to_solution": float(discovery["elapsed"]) if discovery else None,
        "censored_time": float(discovery["elapsed"]) if discovery else seconds,
        "parent_wall": parent_wall,
        "winning_worker": int(discovery["worker"]) if discovery else None,
        "winning_seed": int(discovery["seed"]) if discovery else None,
        "iterations": sum(int(row["iterations"]) for row in rows),
        "worker_rows": rows,
    }


def summarize(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    """Aggregate true verifier-passing discoveries and conservative censoring."""
    groups: Dict[Tuple[int, int], List[Dict[str, object]]] = {}
    for row in rows:
        groups.setdefault((int(row["L"]), int(row["workers"])), []).append(row)
    one_worker_rmst = {
        length: mean(float(row["censored_time"]) for row in values)
        for (length, workers), values in groups.items()
        if workers == 1
    }
    output = []
    for (length, workers), values in sorted(groups.items()):
        solved_times = [
            float(row["time_to_solution"]) for row in values if row["solved"]
        ]
        restricted_mean = mean(float(row["censored_time"]) for row in values)
        baseline = one_worker_rmst.get(length)
        output.append({
            "L": length,
            "workers": workers,
            "trials": len(values),
            "verified_solutions": len(solved_times),
            "success_rate": len(solved_times) / len(values),
            "mean_time_to_solution_when_solved": mean(solved_times) if solved_times else None,
            "median_time_to_solution_when_solved": median(solved_times) if solved_times else None,
            "restricted_mean_time": restricted_mean,
            "speed_factor_vs_one_worker": baseline / restricted_mean if baseline else None,
        })
    return output


def paired_speed_interval(
    rows: Sequence[Dict[str, object]],
    baseline_workers: int = 1,
    optimized_workers: int = 8,
    samples: int = 10_000,
    seed: int = 20260825,
) -> Dict[str, object]:
    """Bootstrap the paired restricted-mean speed factor by trial.

    A failed run contributes its full predeclared censoring horizon through
    ``censored_time``.  Resampling whole trial IDs preserves the paired seed
    structure and avoids treating workers from one portfolio as independent
    observations.
    """
    if samples <= 0:
        raise ValueError("samples must be positive")
    selected = {
        baseline_workers: {},
        optimized_workers: {},
    }
    for row in rows:
        workers = int(row["workers"])
        if workers in selected:
            selected[workers][int(row["trial"])] = float(row["censored_time"])
    trial_ids = sorted(set(selected[baseline_workers]) & set(selected[optimized_workers]))
    if not trial_ids:
        raise ValueError("no paired trials for requested worker counts")

    def ratio(ids: Sequence[int]) -> float:
        baseline = mean(selected[baseline_workers][trial] for trial in ids)
        optimized = mean(selected[optimized_workers][trial] for trial in ids)
        return baseline / optimized

    rng = random.Random(seed)
    estimates = sorted(
        ratio(tuple(rng.choice(trial_ids) for _ in trial_ids))
        for _ in range(samples)
    )
    lower_index = int(0.025 * samples)
    upper_index = max(lower_index, int(0.975 * samples) - 1)
    return {
        "baseline_workers": baseline_workers,
        "optimized_workers": optimized_workers,
        "paired_trials": len(trial_ids),
        "speed_factor": ratio(trial_ids),
        "bootstrap_samples": samples,
        "confidence_level": 0.95,
        "lower": estimates[lower_index],
        "upper": estimates[upper_index],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--L", type=int, default=28)
    parser.add_argument("--workers", type=int, nargs="+", default=(1, 2, 4, 8))
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/benchmark_parallel_solutions.json"),
    )
    args = parser.parse_args()
    if args.trials <= 0:
        raise ValueError("trials must be positive")
    rows: List[Dict[str, object]] = []
    for trial in range(args.trials):
        base_seed = args.seed + trial * 10_000
        for workers in counterbalanced_counts(args.workers, trial):
            row = run_trial(args.L, base_seed, args.seconds, workers)
            row["trial"] = trial
            rows.append(row)
            print(
                "trial={} workers={} solved={} time={}".format(
                    trial, workers, row["solved"],
                    "{:.6f}s".format(row["time_to_solution"])
                    if row["time_to_solution"] is not None else "not reached",
                ),
                flush=True,
            )
    payload = {
        "rows": rows,
        "summary": summarize(rows),
        "paired_speed_interval": paired_speed_interval(
            rows,
            baseline_workers=min(args.workers),
            optimized_workers=max(args.workers),
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))
    print(json.dumps(payload["paired_speed_interval"], indent=2))
    print("saved {}".format(args.output))


if __name__ == "__main__":
    main()
