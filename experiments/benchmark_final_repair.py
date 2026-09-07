"""Compare algorithms intended to finish a complete near-PQCP elite."""

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

from solver.beam_repair import beam_repair
from solver.checkpoint import SearchParameters
from solver.search_runner import (
    DEFAULT_GUIDED_OBJECTIVE_ENERGY_WEIGHT,
    DEFAULT_GUIDED_PROPOSAL_SAMPLES,
    SearchRunner,
)
from solver.torch_relaxation import RelaxationParameters, relax_candidate
from solver.verifier import verify_pqcp
from solver.z3_guidance import guided_completion
from solver.z3_solver import solve_with_z3


Pair = Tuple[Tuple[int, ...], Tuple[int, ...]]


def benchmark(
    lengths: Sequence[int],
    trials: int,
    max_swaps: int,
    seed: int,
    z3_cases_per_radius: int,
    z3_timeout_ms: int,
) -> Dict[str, object]:
    """Run paired controlled repairs; no miss is interpreted as global UNSAT."""
    rows: List[Dict[str, object]] = []
    for length in lengths:
        pairs = _load_pairs(Path("{}.txt".format(length)), length)
        for swaps in range(1, max_swaps + 1):
            for trial in range(trials):
                trial_seed = seed + length * 100_000 + swaps * 1_000 + trial
                center = pairs[trial % len(pairs)]
                candidate = _perturb(center, swaps, random.Random(trial_seed))

                started = perf_counter()
                guidance = guided_completion(
                    *candidate, top_k=10, max_profile_lower_bound=max_swaps * 2
                )
                rows.append(_row(length, swaps, trial, "guidance", guidance.solved, perf_counter() - started))

                beam = beam_repair(*candidate, max_depth=4, beam_width=30)
                rows.append(_row(length, swaps, trial, "beam", beam.solved, beam.elapsed_time))

                started = perf_counter()
                torch_result = relax_candidate(
                    *candidate, RelaxationParameters(steps=400, seed=trial_seed)
                )
                rows.append(_row(length, swaps, trial, "torch", torch_result.verified, perf_counter() - started))

                sa_solved, sa_iterations = _continued_sa(
                    candidate, trial_seed, beam.elapsed_time
                )
                rows.append(_row(
                    length, swaps, trial, "continued_sa", sa_solved,
                    beam.elapsed_time, iterations=sa_iterations,
                ))

                if trial < z3_cases_per_radius:
                    z3 = solve_with_z3(
                        length, *candidate, radius=2 * swaps,
                        timeout_ms=z3_timeout_ms,
                    )
                    row = _row(length, swaps, trial, "z3", z3.status == "SAT" and z3.verified, z3.elapsed_time)
                    row["status"] = z3.status
                    row["reason"] = z3.reason
                    rows.append(row)
    return {
        "lengths": list(lengths), "trials": trials, "max_swaps": max_swaps,
        "z3_cases_per_radius": z3_cases_per_radius,
        "z3_timeout_ms": z3_timeout_ms,
        "rows": rows, "summary": aggregate(rows),
    }


def aggregate(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """Summarize each paired method and length without hiding UNKNOWN."""
    result = {}
    for length in sorted({int(row["L"]) for row in rows}):
        result[str(length)] = {}
        for method in sorted({str(row["method"]) for row in rows if row["L"] == length}):
            selected = [row for row in rows if row["L"] == length and row["method"] == method]
            result[str(length)][method] = {
                "cases": len(selected),
                "solved": sum(bool(row["solved"]) for row in selected),
                "success_rate": sum(bool(row["solved"]) for row in selected) / len(selected),
                "median_seconds": statistics.median(float(row["elapsed"]) for row in selected),
                "status_counts": {
                    status: sum(row.get("status") == status for row in selected)
                    for status in ("SAT", "UNSAT", "UNKNOWN")
                } if method == "z3" else None,
            }
    return result


def _continued_sa(candidate: Pair, seed: int, seconds: float) -> Tuple[bool, int]:
    parameters = SearchParameters(
        stagnation_iterations=None,
        acceptance_mode="objective_plus_target_pair",
        initial_temperature=8.0,
        proposal_samples=DEFAULT_GUIDED_PROPOSAL_SAMPLES,
        objective_energy_weight=DEFAULT_GUIDED_OBJECTIVE_ENERGY_WEIGHT,
    )
    runner = SearchRunner.from_candidate(*candidate, seed, parameters)
    deadline = perf_counter() + seconds
    solved = False
    while perf_counter() < deadline:
        outcome = runner.step()
        solved = solved or outcome.verified_solution is not None
    return solved, runner.state.iteration


def _row(length: int, swaps: int, trial: int, method: str, solved: bool,
         elapsed: float, **extra: object) -> Dict[str, object]:
    return {
        "L": length, "swaps": swaps, "trial": trial, "method": method,
        "solved": solved, "elapsed": elapsed, **extra,
    }


def _load_pairs(path: Path, length: int) -> Tuple[Pair, ...]:
    records = re.findall(r"^a=([01]+)\nb=([01]+)$", path.read_text(encoding="utf-8"), re.MULTILINE)
    pairs = []
    for raw_a, raw_b in records:
        pair = tuple(map(int, raw_a)), tuple(map(int, raw_b))
        if len(pair[0]) == len(pair[1]) == length and verify_pqcp(*pair).is_valid:
            pairs.append(pair)
    if not pairs:
        raise ValueError("no verified pairs in {}".format(path))
    return tuple(pairs)


def _perturb(pair: Pair, swaps: int, rng: random.Random) -> Pair:
    values = [list(pair[0]), list(pair[1])]
    for _ in range(swaps):
        sequence = values[rng.randrange(2)]
        zero = rng.choice([index for index, bit in enumerate(sequence) if bit == 0])
        one = rng.choice([index for index, bit in enumerate(sequence) if bit == 1])
        sequence[zero], sequence[one] = 1, 0
    return tuple(values[0]), tuple(values[1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", type=int, nargs="+", default=(44, 46))
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--max-swaps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=13579)
    parser.add_argument("--z3-cases-per-radius", type=int, default=1)
    parser.add_argument("--z3-timeout-ms", type=int, default=5_000)
    parser.add_argument("--output", type=Path, default=Path("results/benchmark_final_repair.json"))
    args = parser.parse_args()
    payload = benchmark(
        args.lengths, args.trials, args.max_swaps, args.seed,
        args.z3_cases_per_radius, args.z3_timeout_ms,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    print("saved {}".format(args.output))


if __name__ == "__main__":
    main()
