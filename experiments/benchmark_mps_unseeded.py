"""Equal-wall-time first-discovery tests with no known-solution initialization.

Each method/L/seed owns an initially empty directory. No project L.txt,
checkpoint or solution seed is copied or read. Full verifier PASS, not a
low navigation energy, defines success. Unfinished runs are right-censored;
the capped time is reported separately from observed time-to-first-solution.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from statistics import median
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from solver.checkpoint import atomic_write_json
from solver.correlation import full_correlation_profile
from solver.objective import pqcp_objective
from solver.search_runner import append_verified_solution_if_new
from solver.torch_population import PopulationSearchConfig, TorchPopulationSearch
from solver.verifier import verify_pqcp


def run_trial(length, seed, method, seconds, root, *, device="cuda", population=128, islands=2):
    """Run PyTorch alone to first verified discovery or the common deadline."""
    root.mkdir(parents=True, exist_ok=False)
    if device == "cuda":
        torch.cuda.synchronize()
    started = perf_counter()
    config = PopulationSearchConfig(
        length, seed=seed, device=device, population_size=population,
        islands_per_profile=islands, elite_count=min(16, population),
        refinement=(method == "refined"),
    )
    search = TorchPopulationSearch(config)
    progression = []
    previous = None
    first = None
    generations = 0
    new = 0
    while perf_counter() - started < seconds:
        if method == "bootstrap" and generations:
            # Reproduce the discarded-learning behavior of the old source:
            # each cycle is a fresh one-generation model, with a derived seed.
            search = TorchPopulationSearch(PopulationSearchConfig(
                **{**asdict(config), "seed": seed + generations * 1_000_003}
            ))
        batch = search.step()
        generations += 1
        score = int(batch.scores.min().item())
        now = perf_counter() - started
        if previous is None or score < previous:
            previous = score
            i = int(batch.scores.flatten().argmin().item())
            row, col = divmod(i, population)
            a, b = batch.bits[row, col].to(torch.int32).cpu().tolist()
            profile = full_correlation_profile(a, b)
            if profile != batch.profiles[row, col].cpu().tolist() or pqcp_objective(profile) != score:
                raise RuntimeError("CUDA improvement fails exact recomputation")
            progression.append({"elapsed": now, "generation": generations,
                                "score": score, "A": a, "B": b, "profile": profile})
        zero = (batch.scores == 0).nonzero().cpu().tolist()
        for row, col in zero:
            a, b = batch.bits[row, col].to(torch.int32).cpu().tolist()
            result = verify_pqcp(a, b)
            if not result.is_valid or list(result.profile) != batch.profiles[row, col].cpu().tolist():
                raise RuntimeError("CUDA zero fails independent verifier")
            if first is None:
                first = perf_counter() - started
            new += append_verified_solution_if_new(length, tuple(a), tuple(b), root)
        if first is not None:
            break
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = perf_counter() - started
    record = {
        "L": length, "seed": seed, "method": method, "device": device,
        "config": asdict(config), "budget_seconds": seconds, "elapsed": elapsed,
        "generations": generations, "first_verified_seconds": first,
        "censored": first is None, "capped_time": min(first, seconds) if first is not None else seconds,
        "best_score": previous, "new_verified": new, "progression": progression,
        "known_solution_inputs": 0, "uses_c": False, "uses_z3": False,
    }
    atomic_write_json(root / "trial.json", record)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", nargs="+", type=int, default=[20, 28, 44, 68])
    parser.add_argument("--seeds", nargs="+", type=int, default=[123, 456, 789])
    parser.add_argument("--methods", nargs="+", choices=["bootstrap", "cem", "refined"],
                        default=["bootstrap", "cem", "refined"])
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument("--population", type=int, default=128)
    parser.add_argument("--islands", type=int, default=2)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.seconds <= 0:
        parser.error("--seconds must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    atomic_write_json(args.output / "manifest.json", {
        "torch": torch.__version__, "device": args.device, "known_solution_inputs": 0,
        "population_source_sha256": hashlib.sha256((ROOT / "solver/torch_population.py").read_bytes()).hexdigest(),
        "benchmark_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "args": {**vars(args), "output": str(args.output)},
    })
    records = []
    for length in args.lengths:
        for index, seed in enumerate(args.seeds):
            methods = args.methods[index % len(args.methods):] + args.methods[:index % len(args.methods)]
            for method in methods:
                root = args.output / "L{}_{}_seed{}".format(length, method, seed)
                record = run_trial(length, seed, method, args.seconds, root, device=args.device,
                                   population=args.population, islands=args.islands)
                records.append(record)
                with (args.output / "runs.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record) + "\n")
                print("L={} {:9s} seed={} time={:.3f}s best={} first={} verified={}".format(
                    length, method, seed, record["elapsed"], record["best_score"],
                    record["first_verified_seconds"], record["new_verified"]), flush=True)
    summary = []
    for length in args.lengths:
        for method in args.methods:
            runs = [r for r in records if r["L"] == length and r["method"] == method]
            summary.append({"L": length, "method": method, "runs": len(runs),
                            "successes": sum(not r["censored"] for r in runs),
                            "median_capped_time": median(r["capped_time"] for r in runs),
                            "best_scores": [r["best_score"] for r in runs]})
    atomic_write_json(args.output / "summary.json", {"groups": summary})
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
