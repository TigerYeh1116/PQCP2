"""Measure spawn-based independent SA scaling without changing a trajectory.

Every worker runs the existing production GCP/structured-SA path with repair
and Z3 disabled.  Worker roots are disjoint, so checkpoints, logs, best files,
and temporary ``L.txt`` files cannot race.  This is a throughput experiment;
production result merging remains a separate, parent-owned responsibility.
"""

import argparse
import json
import multiprocessing as mp
from pathlib import Path
from statistics import mean, median
import sys
import tempfile
from time import perf_counter
import traceback
from typing import Dict, Iterable, List, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from solver.pipeline import PipelineConfig, run_pipeline
from solver.parallel_pipeline import SEED_STRIDE, derived_worker_seed


def worker_seed(base_seed: int, worker_index: int) -> int:
    """Return transparent, nonoverlapping seeds for independent trajectories."""
    return derived_worker_seed(base_seed, worker_index)


def _worker_entry(
    length: int,
    seed: int,
    seconds: float,
    worker_index: int,
    root: str,
    queue: object,
) -> None:
    """Run one isolated existing pipeline and return only JSON-safe metrics."""
    try:
        worker_root = Path(root) / "workers" / "w{:02d}".format(worker_index)
        started = perf_counter()
        result = run_pipeline(PipelineConfig(
            L=length,
            seed=seed,
            seconds=seconds,
            gcp=True,
            repair=False,
            z3=False,
            checkpoint_interval=max(1.0, seconds + 1.0),
            progress_interval=max(1.0, seconds + 1.0),
            root=worker_root,
        ))
        wall = perf_counter() - started
        queue.put({
            "ok": True,
            "worker": worker_index,
            "seed": seed,
            "wall": wall,
            "iterations": result.state.iteration,
            "restarts": result.state.restart_index,
            "best_score": result.state.best_score,
            "verified": bool(result.verification.is_valid or result.verified_paths),
        })
    except BaseException as error:  # pragma: no cover - child failure plumbing
        queue.put({
            "ok": False,
            "worker": worker_index,
            "seed": seed,
            "error": "{}: {}".format(type(error).__name__, error),
            "traceback": traceback.format_exc(),
        })


def run_parallel_case(
    length: int,
    base_seed: int,
    seconds: float,
    workers: int,
    root: Path,
) -> Dict[str, object]:
    """Run ``workers`` independent spawn processes for one wall-clock case."""
    if seconds <= 0:
        raise ValueError("seconds must be positive")
    if not isinstance(workers, int) or isinstance(workers, bool) or workers <= 0:
        raise ValueError("workers must be a positive integer")
    context = mp.get_context("spawn")
    queue = context.Queue()
    processes = []
    started = perf_counter()
    for worker_index in range(workers):
        seed = worker_seed(base_seed, worker_index)
        process = context.Process(
            target=_worker_entry,
            args=(length, seed, seconds, worker_index, str(root), queue),
            name="pqcp-benchmark-w{:02d}".format(worker_index),
        )
        process.start()
        processes.append(process)
    rows = [queue.get() for _ in processes]
    for process in processes:
        process.join()
    wall = perf_counter() - started
    failures = [row for row in rows if not row.get("ok")]
    if failures:
        raise RuntimeError("parallel worker failed: {}".format(failures[0]))
    rows.sort(key=lambda row: int(row["worker"]))
    iterations = sum(int(row["iterations"]) for row in rows)
    return {
        "L": length,
        "base_seed": base_seed,
        "seconds_per_worker": seconds,
        "workers": workers,
        "parent_wall": wall,
        "total_iterations": iterations,
        "aggregate_iterations_per_second": iterations / wall if wall else 0.0,
        "minimum_best_score": min(int(row["best_score"]) for row in rows),
        "median_best_score": median(int(row["best_score"]) for row in rows),
        "verified": sum(bool(row["verified"]) for row in rows),
        "worker_rows": rows,
    }


def summarize(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    """Aggregate repeats and report speedup relative to one spawn worker."""
    grouped: Dict[Tuple[int, int], List[Dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((int(row["L"]), int(row["workers"])), []).append(row)
    one_worker_rates = {
        length: mean(float(row["aggregate_iterations_per_second"]) for row in values)
        for (length, workers), values in grouped.items()
        if workers == 1
    }
    output = []
    for (length, workers), values in sorted(grouped.items()):
        rate = mean(float(row["aggregate_iterations_per_second"]) for row in values)
        baseline = one_worker_rates.get(length)
        output.append({
            "L": length,
            "workers": workers,
            "repeats": len(values),
            "mean_parent_wall": mean(float(row["parent_wall"]) for row in values),
            "mean_aggregate_iterations_per_second": rate,
            "speedup_vs_one_worker": rate / baseline if baseline else None,
            "minimum_best_score": min(int(row["minimum_best_score"]) for row in values),
            "median_of_median_best_scores": median(
                float(row["median_best_score"]) for row in values
            ),
            "verified": sum(int(row["verified"]) for row in values),
        })
    return output


def counterbalanced_counts(counts: Sequence[int], repeat: int) -> Tuple[int, ...]:
    """Rotate worker-count order so thermal bias is not tied to one count."""
    values = tuple(counts)
    if not values:
        raise ValueError("at least one worker count is required")
    offset = repeat % len(values)
    return values[offset:] + values[:offset]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", type=int, nargs="+", default=(44, 46))
    parser.add_argument("--workers", type=int, nargs="+", default=(1, 2, 4, 8))
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/benchmark_parallel_search.json"),
    )
    args = parser.parse_args()
    if args.repeats <= 0:
        raise ValueError("repeats must be positive")
    rows: List[Dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="pqcp-parallel-benchmark-") as directory:
        benchmark_root = Path(directory)
        for repeat in range(args.repeats):
            for length in args.lengths:
                for workers in counterbalanced_counts(args.workers, repeat):
                    case = run_parallel_case(
                        length,
                        args.seed + repeat * 1_000_000,
                        args.seconds,
                        workers,
                        benchmark_root / "L{}-r{}-w{}".format(length, repeat, workers),
                    )
                    case["repeat"] = repeat
                    rows.append(case)
                    print(
                        "L={} workers={} rate={:.0f}/s min_score={} wall={:.3f}s".format(
                            length, workers,
                            case["aggregate_iterations_per_second"],
                            case["minimum_best_score"], case["parent_wall"],
                        ),
                        flush=True,
                    )
    payload = {"rows": rows, "summary": summarize(rows)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))
    print("saved {}".format(args.output))


if __name__ == "__main__":
    main()
