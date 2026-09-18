"""Paired wall-clock evaluation of C stagnation-handling strategies."""

import argparse
from collections import Counter
from pathlib import Path
from statistics import mean, median
from tempfile import TemporaryDirectory
from typing import Dict, List, Sequence
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from solver.c_backend import CSearchConfig, run_c_search


DEFAULT_SEEDS = (123, 456, 789, 1024, 2026)


def _run(length: int, seed: int, seconds: float, method: str,
         root: Path, executable: Path) -> Dict[str, object]:
    scores: List[int] = []
    energies: List[int] = []

    def collect(_a, _b, metadata) -> None:
        scores.append(int(metadata["score"]))
        energies.append(int(metadata["energy"]))

    result = run_c_search(CSearchConfig(
        L=length, seed=seed, seconds=seconds, threads=4,
        root=root, executable=executable, echo=False, emit_elites=True,
        candidate_count=2, direct_swap_sampling=True,
        quarter_frequency_pruning=True, half_shift_seeding=True,
        adaptive_kick=method in ("adaptive", "reactive"),
        quench=method in ("quench", "energy_quench", "reactive", "full", "portfolio",
                          "light", "factor2"),
        objective_quench=method != "energy_quench",
        energy_mode=method if method in ("full", "portfolio", "light", "factor2") else "multiscale",
    ), elite_callback=collect)
    return {
        "method": method,
        "seed": seed,
        "elapsed": result.elapsed,
        "restarts": result.restarts,
        "moves": result.moves,
        "swaps": result.swap_evaluations,
        "elite_count": len(scores),
        "best_score": min(scores) if scores else None,
        "median_score": median(scores) if scores else None,
        "score_le_16": sum(score <= 16 for score in scores),
        "score_le_32": sum(score <= 32 for score in scores),
        "best_energy": min(energies) if energies else None,
        "verified_candidates": result.verified_candidates,
        "new_solutions": result.new_solutions,
    }


def summarize(rows: Sequence[Dict[str, object]], baseline: str,
              challenger: str) -> None:
    print("method runs best_min best_mean median_local <=16 <=32 verified moves/s")
    for method in (baseline, challenger):
        selected = [row for row in rows if row["method"] == method]
        best = [int(row["best_score"]) for row in selected
                if row["best_score"] is not None]
        local_weight = sum(int(row["elite_count"]) for row in selected)
        weighted_median = mean(float(row["median_score"]) for row in selected
                               if row["median_score"] is not None)
        rate = sum(int(row["moves"]) for row in selected) / sum(
            float(row["elapsed"]) for row in selected
        )
        print("{} {} {} {:.2f} {:.2f} {} {} {} {:.0f}".format(
            method, len(selected), min(best), mean(best), weighted_median,
            sum(int(row["score_le_16"]) for row in selected),
            sum(int(row["score_le_32"]) for row in selected),
            sum(int(row["verified_candidates"]) for row in selected), rate,
        ))
        print("  completed local bests={}".format(local_weight))
    paired = Counter()
    by_key = {(int(row["seed"]), str(row["method"])): row for row in rows}
    for seed in sorted({int(row["seed"]) for row in rows}):
        old = by_key[(seed, baseline)]["best_score"]
        new = by_key[(seed, challenger)]["best_score"]
        outcome = ("{}_win".format(challenger) if new < old else
                   "{}_win".format(baseline) if old < new else "tie")
        paired[outcome] += 1
        print("seed={} {}={} {}={} {}".format(
            seed, baseline, old, challenger, new, outcome))
    print("paired={}".format(dict(paired)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--L", type=int, default=68)
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    methods = ("legacy", "energy_quench", "quench", "adaptive", "reactive", "full",
               "portfolio", "light", "factor2")
    parser.add_argument("--baseline", choices=methods, default="legacy")
    parser.add_argument("--challenger", choices=methods,
                        default="adaptive")
    args = parser.parse_args()
    seeds = tuple(int(value) for value in args.seeds.split(","))
    if args.seconds <= 0 or not seeds:
        raise ValueError("seconds and seeds must be positive/nonempty")
    if args.baseline == args.challenger:
        raise ValueError("baseline and challenger must differ")
    rows = []
    with TemporaryDirectory(prefix="pqcp-stagnation-") as raw:
        base = Path(raw)
        executable = base / "pqcp_search_c"
        for index, seed in enumerate(seeds):
            # Alternate order to reduce systematic thermal/order bias.
            compared = (args.baseline, args.challenger)
            order = compared if index % 2 == 0 else tuple(reversed(compared))
            for method in order:
                row = _run(
                    args.L, seed, args.seconds, method,
                    base / "{}_{}".format(method, seed), executable,
                )
                rows.append(row)
                print(row)
    summarize(rows, args.baseline, args.challenger)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
