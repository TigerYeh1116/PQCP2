"""Paired wall-clock benchmark of production search with and without Z3."""

import argparse
import json
from pathlib import Path
from statistics import mean, median
import sys
import tempfile
from time import perf_counter


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from solver.pipeline import PipelineConfig, run_pipeline


def run_case(
    length: int,
    seed: int,
    seconds: float,
    method: str,
    timeout: float,
    trigger_score: int,
):
    """Run one production GCP/structured-SA case in an isolated directory."""
    with tempfile.TemporaryDirectory(prefix="pqcp-z3-ablation-") as directory:
        started = perf_counter()
        result = run_pipeline(PipelineConfig(
            L=length, seed=seed, seconds=seconds, gcp=True,
            repair=method in ("repair_only", "z3"), z3=method == "z3",
            z3_timeout=timeout, z3_trigger_score=trigger_score,
            checkpoint_interval=max(1.0, seconds + timeout + 1.0),
            progress_interval=max(1.0, seconds + timeout + 1.0),
            root=Path(directory),
        ))
        wall = perf_counter() - started
    completion_result = result.z3_result if isinstance(result.z3_result, dict) else {}
    completion_solved = bool(completion_result.get("solved", False))
    exact_z3_executed = bool(completion_result.get("z3_executed", False))
    return {
        "L": length, "seed": seed, "method": method,
        "trigger_score": trigger_score,
        "budget": seconds, "wall": wall,
        "iterations": result.state.iteration,
        "restarts": result.state.restart_index,
        "best_score": result.state.best_score,
        "verified": bool(
            result.verification.is_valid
            or result.verified_paths
            or completion_solved
        ),
        # PipelineResult retained its historical field name, but it counts
        # every completion attempt (guidance/beam included), not only Z3.
        "completion_attempts": result.z3_trigger_count,
        "exact_z3_executed": exact_z3_executed,
        "wall_overrun": max(0.0, wall - seconds),
    }


def summarize(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row["L"], row["method"]), []).append(row)
    output = []
    for (length, method), values in sorted(groups.items()):
        output.append({
            "L": length, "method": method, "runs": len(values),
            "median_score": median(row["best_score"] for row in values),
            "mean_score": mean(row["best_score"] for row in values),
            "mean_iterations": mean(row["iterations"] for row in values),
            "mean_iterations_per_second": mean(
                row["iterations"] / row["wall"] for row in values
            ),
            "mean_wall": mean(row["wall"] for row in values),
            "completion_attempts": sum(row["completion_attempts"] for row in values),
            "exact_z3_executions": sum(row["exact_z3_executed"] for row in values),
            "mean_wall_overrun": mean(row["wall_overrun"] for row in values),
            "verified": sum(row["verified"] for row in values),
        })
    return output


def method_order(length: int, seed: int):
    """Counterbalance thermal/cache order deterministically across paired runs."""
    methods = ("sa_only", "repair_only", "z3")
    offset = (length + seed) % len(methods)
    return methods[offset:] + methods[:offset]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", type=int, nargs="+", default=[44, 46])
    parser.add_argument("--seeds", type=int, nargs="+", default=[123, 456, 789, 1024, 2026])
    parser.add_argument("--seconds", type=float, default=15.0)
    parser.add_argument("--z3-timeout", type=float, default=5.0)
    parser.add_argument(
        "--trigger-score", type=int, default=16,
        help="fixed objective threshold shared by repair-only and Z3 runs",
    )
    parser.add_argument("--output", type=Path, default=Path("results/benchmark_z3_removal.json"))
    args = parser.parse_args()
    rows = []
    for length in args.lengths:
        for seed in args.seeds:
            for method in method_order(length, seed):
                rows.append(run_case(
                    length, seed, args.seconds, method, args.z3_timeout,
                    args.trigger_score,
                ))
    payload = {"rows": rows, "summary": summarize(rows)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))
    print("saved {}".format(args.output))


if __name__ == "__main__":
    main()
