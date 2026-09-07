"""Fixed-evaluation ablation for safely transferred structured PQCP search.

This benchmark compares navigation energies under the same seed set and
proposal count.  It deliberately reports swap evaluations, not only wall
time, because multiscale energies have different per-evaluation costs.
"""

import argparse
import json
from pathlib import Path
from statistics import mean, median
import sys
from time import perf_counter
from typing import Dict, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from solver.checkpoint import SearchParameters
from solver.search_runner import SearchRunner
from solver.target_profiles import canonical_target_content_profiles
from solver.weight_constraints import canonical_weight_pairs


MODES = (
    "legacy_baseline",
    "fixed_target_full",
    "fixed_target_full_e2",
    "fixed_target_multiscale",
)


def encoded_profiles(length: int):
    """Return checkpoint-safe canonical target/content profiles."""
    return tuple(
        (p.k, p.eta, p.a_even_ones, p.a_odd_ones,
         p.b_even_ones, p.b_odd_ones)
        for p in canonical_target_content_profiles(length)
    )


def run_case(length: int, seed: int, iterations: int, samples: int, mode: str) -> Dict[str, object]:
    """Run one deterministic fixed-evaluation structured-search case."""
    profiles = encoded_profiles(length)
    if not profiles:
        return {"L": length, "seed": seed, "mode": mode, "skipped": "no necessary target profile"}
    if mode == "legacy_baseline":
        parameters = SearchParameters(
            stagnation_iterations=max(100, iterations // 4),
            weight_pairs=canonical_weight_pairs(length),
            acceptance_mode="objective_plus_target_pair",
            proposal_samples=samples,
            objective_energy_weight=3,
        )
    else:
        parameters = SearchParameters(
            stagnation_iterations=max(100, iterations // 4),
            target_content_profiles=profiles,
            preserve_alternating_content=True,
            acceptance_mode=mode,
            proposal_samples=samples,
        )
    runner = SearchRunner.new(length, seed, parameters)
    initial = runner.state.best_score
    started = perf_counter()
    for _ in range(iterations):
        runner.step()
    elapsed = perf_counter() - started
    return {
        "L": length, "seed": seed, "mode": mode,
        "initial_score": initial, "best_score": runner.state.best_score,
        "iterations": iterations, "swap_evaluations": iterations * samples,
        "restarts": runner.state.restart_index, "elapsed": elapsed,
        "evaluations_per_second": iterations * samples / elapsed if elapsed else 0.0,
    }


def summarize(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    """Aggregate scores and evaluation rates by L/mode."""
    groups = {}
    for row in rows:
        if "best_score" in row:
            groups.setdefault((row["L"], row["mode"]), []).append(row)
    result = []
    for (length, mode), values in sorted(groups.items()):
        scores = [int(row["best_score"]) for row in values]
        rates = [float(row["evaluations_per_second"]) for row in values]
        result.append({
            "L": length, "mode": mode, "runs": len(values),
            "min_score": min(scores), "median_score": median(scores),
            "mean_score": mean(scores), "max_score": max(scores),
            "mean_evaluations_per_second": mean(rates),
            "solutions": sum(score == 0 for score in scores),
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", type=int, nargs="+", default=[8, 44, 46])
    parser.add_argument("--seeds", type=int, nargs="+", default=[123, 456, 789, 1024, 2026])
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--proposal-samples", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("results/structured_search_ablation.json"))
    args = parser.parse_args()
    rows = [
        run_case(length, seed, args.iterations, args.proposal_samples, mode)
        for length in args.lengths for seed in args.seeds for mode in MODES
    ]
    summary = summarize(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"runs": rows, "summary": summary}, indent=2) + "\n", encoding="utf-8")
    print("Structured search ablation")
    for row in summary:
        print(
            "L={L} {mode}: min={min_score} median={median_score} mean={mean_score:.2f} "
            "max={max_score} solutions={solutions}/{runs} rate={mean_evaluations_per_second:.0f}/s".format(**row)
        )
    for row in rows:
        if "skipped" in row:
            print("L={} {}: {}".format(row["L"], row["mode"], row["skipped"]))
    print("saved {}".format(args.output))


if __name__ == "__main__":
    main()
