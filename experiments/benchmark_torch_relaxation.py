"""Compare PyTorch relaxation with current SA on controlled and real elites."""

import argparse
import json
from pathlib import Path
import random
import re
import statistics
import sys
from time import perf_counter
from typing import Dict, List, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from solver.checkpoint import SearchParameters
from solver.correlation import full_correlation_profile
from solver.objective import pqcp_objective
from solver.search_runner import (
    DEFAULT_GUIDED_OBJECTIVE_ENERGY_WEIGHT,
    DEFAULT_GUIDED_PROPOSAL_SAMPLES,
    SearchRunner,
)
from solver.torch_relaxation import RelaxationParameters, relax_candidate
from solver.verifier import verify_pqcp


Pair = Tuple[Tuple[int, ...], Tuple[int, ...]]


def load_verified_pairs(path: Path, length: int) -> Tuple[Pair, ...]:
    """Load independently verified pairs from the project's established file."""
    records = re.findall(r"^a=([01]+)\nb=([01]+)$", path.read_text(encoding="utf-8"), re.MULTILINE)
    pairs = []
    for raw_a, raw_b in records:
        a, b = tuple(map(int, raw_a)), tuple(map(int, raw_b))
        if len(a) == length and len(b) == length and verify_pqcp(a, b).is_valid:
            pairs.append((a, b))
    if not pairs:
        raise ValueError("no verified length-{} pairs in {}".format(length, path))
    return tuple(pairs)


def perturb_by_swaps(pair: Pair, swaps: int, rng: random.Random) -> Pair:
    """Apply legal random fixed-weight swaps without using target gradients."""
    values = [list(pair[0]), list(pair[1])]
    for _ in range(swaps):
        sequence = values[rng.randrange(2)]
        zeros = [index for index, bit in enumerate(sequence) if bit == 0]
        ones = [index for index, bit in enumerate(sequence) if bit == 1]
        zero, one = rng.choice(zeros), rng.choice(ones)
        sequence[zero], sequence[one] = 1, 0
    return tuple(values[0]), tuple(values[1])


def combined_hamming(left: Pair, right: Pair) -> int:
    """Return literal combined A/B Hamming distance for a controlled center."""
    return sum(x != y for x, y in zip(left[0], right[0])) + sum(
        x != y for x, y in zip(left[1], right[1])
    )


def run_sa_for_seconds(pair: Pair, seed: int, seconds: float) -> Dict[str, object]:
    """Run the current production SA navigation for the exact wall budget."""
    parameters = SearchParameters(
        stagnation_iterations=None,
        acceptance_mode="objective_plus_target_pair",
        initial_temperature=8.0,
        proposal_samples=DEFAULT_GUIDED_PROPOSAL_SAMPLES,
        objective_energy_weight=DEFAULT_GUIDED_OBJECTIVE_ENERGY_WEIGHT,
    )
    runner = SearchRunner.from_candidate(pair[0], pair[1], seed, parameters)
    deadline = perf_counter() + seconds
    solved = False
    while perf_counter() < deadline:
        outcome = runner.step()
        solved = solved or outcome.verified_solution is not None
    return {
        "score": runner.state.best_score,
        "verified": solved,
        "iterations": runner.state.iteration,
        "a": list(runner.state.best_a),
        "b": list(runner.state.best_b),
    }


def controlled_benchmark(
    pairs: Sequence[Pair], swaps: Sequence[int], trials: int, steps: int, seed: int,
) -> List[Dict[str, object]]:
    """Compare recovery from withheld fixed-weight perturbations at equal wall time."""
    rows = []
    for radius in swaps:
        for trial in range(trials):
            trial_seed = seed + 10_000 * radius + trial
            center = pairs[trial % len(pairs)]
            candidate = perturb_by_swaps(center, radius, random.Random(trial_seed))
            initial_profile = full_correlation_profile(*candidate)
            started = perf_counter()
            result = relax_candidate(
                *candidate,
                RelaxationParameters(steps=steps, seed=trial_seed),
            )
            torch_seconds = perf_counter() - started
            sa = run_sa_for_seconds(candidate, trial_seed, torch_seconds)
            rows.append({
                "kind": "controlled", "swaps": radius, "trial": trial,
                "seed": trial_seed, "initial_score": pqcp_objective(initial_profile),
                "initial_distance": combined_hamming(candidate, center),
                "torch_score": result.best_score, "torch_verified": result.verified,
                "torch_seconds": torch_seconds, "torch_steps": result.steps,
                "torch_distance_to_center": combined_hamming((result.best_a, result.best_b), center),
                "sa_score": sa["score"], "sa_verified": sa["verified"],
                "sa_iterations": sa["iterations"],
                "sa_distance_to_center": combined_hamming(
                    (tuple(sa["a"]), tuple(sa["b"])), center
                ),
            })
    return rows


def load_real_elites(directory: Path, length: int) -> Tuple[Tuple[Path, Pair], ...]:
    """Load exact candidates persisted by genuine long-running searches."""
    elites = []
    for path in sorted(directory.glob("L{}_*.json".format(length))):
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw_a = payload.get("A", payload.get("a"))
        raw_b = payload.get("B", payload.get("b"))
        if raw_a is None or raw_b is None:
            continue
        a, b = tuple(raw_a), tuple(raw_b)
        if len(a) == length and len(b) == length:
            elites.append((path, (a, b)))
    return tuple(elites)


def real_elite_benchmark(
    elites: Sequence[Tuple[Path, Pair]], steps: int, seed: int,
) -> List[Dict[str, object]]:
    """Measure score improvement on states for which the true basin is unknown."""
    rows = []
    for index, (path, pair) in enumerate(elites):
        result = relax_candidate(
            *pair, RelaxationParameters(steps=steps, seed=seed + index)
        )
        rows.append({
            "kind": "real_elite", "source": str(path),
            "initial_score": result.initial_score, "torch_score": result.best_score,
            "torch_verified": result.verified, "torch_steps": result.steps,
        })
    return rows


def summarize(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """Return transparent accuracy summaries without redefining a miss as success."""
    controlled = [row for row in rows if row["kind"] == "controlled"]
    real = [row for row in rows if row["kind"] == "real_elite"]
    by_radius = {}
    for radius in sorted({row["swaps"] for row in controlled}):
        selected = [row for row in controlled if row["swaps"] == radius]
        by_radius[str(radius)] = {
            "count": len(selected),
            "torch_verified": sum(bool(row["torch_verified"]) for row in selected),
            "sa_verified": sum(bool(row["sa_verified"]) for row in selected),
            "torch_median_score": statistics.median(row["torch_score"] for row in selected),
            "sa_median_score": statistics.median(row["sa_score"] for row in selected),
            "torch_median_distance": statistics.median(row["torch_distance_to_center"] for row in selected),
            "sa_median_distance": statistics.median(row["sa_distance_to_center"] for row in selected),
        }
    return {
        "controlled_by_swaps": by_radius,
        "controlled_total": len(controlled),
        "torch_verified_total": sum(bool(row["torch_verified"]) for row in controlled),
        "sa_verified_total": sum(bool(row["sa_verified"]) for row in controlled),
        "real_elite_count": len(real),
        "real_elite_improved": sum(row["torch_score"] < row["initial_score"] for row in real),
        "real_elite_verified": sum(bool(row["torch_verified"]) for row in real),
        "real_elite_initial_median": statistics.median(
            row["initial_score"] for row in real
        ) if real else None,
        "real_elite_torch_median": statistics.median(
            row["torch_score"] for row in real
        ) if real else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--L", type=int, default=44)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=24680)
    parser.add_argument("--swaps", type=int, nargs="+", default=(1, 2, 3, 4))
    parser.add_argument("--known", type=Path)
    parser.add_argument("--best-dir", type=Path, default=Path("results/best"))
    parser.add_argument("--output", type=Path, default=Path("results/benchmark_torch_relaxation.json"))
    args = parser.parse_args()
    pairs = load_verified_pairs(args.known or Path("{}.txt".format(args.L)), args.L)
    rows = controlled_benchmark(pairs, args.swaps, args.trials, args.steps, args.seed)
    rows.extend(real_elite_benchmark(load_real_elites(args.best_dir, args.L), args.steps, args.seed + 1_000_000))
    payload = {
        "L": args.L, "trials": args.trials, "steps": args.steps,
        "summary": summarize(rows), "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    print("saved {}".format(args.output))


if __name__ == "__main__":
    main()
