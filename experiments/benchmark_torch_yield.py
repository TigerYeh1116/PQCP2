"""Paired actual-new-PQCP measurement for the persistent CUDA/C pipeline.

Every arm receives the SAME frozen inventory for deduplication only, never
initialization. It runs real CUDA formation, real C completion and unchanged
verification/writing. Writer wrappers record successful appends AFTER calling
the real writer; they neither select candidates nor alter search decisions.
Wall time includes startup, bank construction, verification and shutdown.
No-Z3 controls isolate this optimization. No automatic long benchmark in pytest.
"""

import argparse
from contextlib import ExitStack
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import statistics
import sys
import threading
from time import perf_counter
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from solver.checkpoint import atomic_write_json
from solver.search_runner import append_verified_solution_if_new
from solver.torch_hybrid_runner import TorchHybridConfig, run_torch_hybrid_search
from solver.torch_search import require_device
from solver.verifier import verify_pqcp


VARIANTS = ("current", "spectral", "early", "optimized", "cached", "metal",
            "mixed", "wide", "feasibility", "feasibility_long", "frequency",
            "math_psd", "math_divisor", "math_lattice", "math_variance", "math_combined",
            "math_lattice_bootstrap")


def variant_config(base: TorchHybridConfig, variant: str) -> TorchHybridConfig:
    """Change only the named optimization; retain an explicit current arm."""
    if variant not in VARIANTS:
        raise ValueError("unknown variant")
    feasible = variant in ("feasibility", "feasibility_long")
    mixed = variant in ("mixed", "wide", "frequency") or feasible
    accelerator_optimized = variant == "metal" or mixed
    early = variant in ("early", "optimized") or accelerator_optimized
    cached = variant in ("cached", "optimized") or accelerator_optimized
    return replace(
        base,
        continuous_batch_size=base.continuous_batch_size * (4 if variant == "wide" else 1),
        continuous_loss_backend="spectral" if variant in ("spectral", "optimized") else "profile",
        continuous_optimization_mode="douglas_rachford" if feasible else "relaxed",
        continuous_steps_per_restart=(20000 if variant == "feasibility_long"
                                      else base.continuous_steps_per_restart),
        continuous_frequency_pruning=variant == "frequency",
        continuous_mathematical_loss={
            "math_psd": "psd_cap", "math_divisor": "divisor_lift", "math_lattice": "lattice",
            "math_variance": "variance",
            "math_combined": "combined",
            "math_lattice_bootstrap": "lattice_bootstrap",
        }.get(variant, "none"),
        fast_observation=cached,
        # The historical "metal" arm keeps its surrounding optimization
        # controls, but CUDA observation uses the equivalent PyTorch path.
        observation_backend="torch",
        completion_pair_seed_percent=50 if mixed else 100,
        initial_formation_seconds=8.0 if early else None,
    )


def run_trial(config: TorchHybridConfig, variant: str, inventory: str) -> dict:
    """Measure actual novel, independently verified pairs in an isolated root."""
    root = Path(config.root)
    root.mkdir(parents=True, exist_ok=False)
    (root / "{}.txt".format(config.L)).write_text(inventory, encoding="utf-8")
    events, messages = [], []
    lock = threading.RLock()
    started = perf_counter()

    def recorder(source):
        def write(length, a, b, directory):
            new = append_verified_solution_if_new(length, a, b, directory)
            if new:
                elapsed = perf_counter() - started
                check = verify_pqcp(a, b)
                if not check.is_valid:
                    raise RuntimeError("real writer accepted an invalid PQCP")
                event = {"elapsed": elapsed, "source": source, "A": list(a), "B": list(b),
                         "profile": list(check.profile), "verified": True}
                with lock:
                    events.append(event)
                    with (root / "solutions.jsonl").open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(event) + "\n")
            return new
        return write

    def output(line):
        if line.startswith("[CUDA]"):
            with lock:
                messages.append({"elapsed": perf_counter() - started, "line": line})

    with ExitStack() as stack:
        for module, source in (("solver.torch_search", "cuda-continuous"),
                               ("solver.torch_hybrid_runner", "cuda-polish"),
                               ("solver.c_backend", "c-completion")):
            stack.enter_context(patch(module + ".append_verified_solution_if_new", recorder(source)))
        result = run_torch_hybrid_search(config, line_callback=output)
    elapsed = perf_counter() - started
    events.sort(key=lambda event: event["elapsed"])
    unique = {tuple(sorted((tuple(e["A"]), tuple(e["B"])))) for e in events}
    if len(unique) != len(events) or result.new_solutions != len(events):
        raise RuntimeError("writer events disagree with unique-pair/counter accounting")
    report = {
        "L": config.L, "seed": config.seed, "variant": variant,
        "budget": config.seconds, "elapsed": elapsed,
        "first_new_seconds": events[0]["elapsed"] if events else None,
        "censored": not events, "new_pairs": len(events),
        "new_per_second": len(events) / elapsed,
        "inventory_sha256": hashlib.sha256(inventory.encode()).hexdigest(),
        "config": {**asdict(config), "root": str(root)},
        "result": {**asdict(result), "last_seed_bank": str(result.last_seed_bank) if result.last_seed_bank else None},
        "events": events, "progress": messages,
    }
    atomic_write_json(root / "report.json", report)
    return report


def summarize(rows: list) -> list:
    """Keep failures censored; never silently drop them from time averages."""
    summary = []
    for length, variant in sorted({(r["L"], r["variant"]) for r in rows}):
        group = [r for r in rows if r["L"] == length and r["variant"] == variant]
        hits = [r["first_new_seconds"] for r in group if r["first_new_seconds"] is not None]
        total_new = sum(r["new_pairs"] for r in group)
        total_time = sum(r["elapsed"] for r in group)
        summary.append({
            "L": length, "variant": variant, "runs": len(group),
            "successful_runs": len(hits), "new_pairs": total_new,
            "total_elapsed": total_time, "new_per_second": total_new / total_time,
            "seconds_per_new": total_time / total_new if total_new else None,
            "mean_first_new": statistics.mean(hits) if len(hits) == len(group) else None,
            "median_first_new_successes_only": statistics.median(hits) if hits else None,
        })
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", type=int, nargs="+", default=[44])
    parser.add_argument("--seeds", type=int, nargs="+", default=[123, 456, 789])
    parser.add_argument("--seconds", type=float, default=90)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS,
                        default=["current", "optimized"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if not 0 < args.seconds < float("inf") or len(set(args.seeds)) != len(args.seeds):
        parser.error("finite positive seconds and distinct seeds required")
    require_device("cuda")
    args.output.mkdir(parents=True, exist_ok=False)
    inventories = {length: (ROOT / "{}.txt".format(length)).read_text(encoding="utf-8")
                   if (ROOT / "{}.txt".format(length)).exists() else "" for length in args.lengths}
    sources = sorted(str(path.relative_to(ROOT)) for path in (ROOT / "solver").glob("*.py"))
    sources += ["csrc/pqcp_search.c", "main.py", "experiments/benchmark_torch_yield.py"]
    atomic_write_json(args.output / "manifest.json", {
        "sources": {name: {"sha256": hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),
                            "text": (ROOT/name).read_text(encoding="utf-8")} for name in sources},
        "inventories": {str(length): text for length, text in inventories.items()},
        "argv": vars(args) | {"output": str(args.output)},
    })
    rows = []
    for length in args.lengths:
        for index, seed in enumerate(args.seeds):
            order = args.variants[index % len(args.variants):] + args.variants[:index % len(args.variants)]
            for variant in order:
                root = args.output.resolve() / "L{}_seed{}_{}".format(length, seed, variant)
                config = variant_config(TorchHybridConfig(
                    length, seed=seed, seconds=args.seconds, continuous_batch_size=2048 if length <= 58 else 1024,
                    root=root), variant)
                row = run_trial(config, variant, inventories[length])
                rows.append(row)
                with (args.output / "runs.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row) + "\n")
                print(json.dumps({key: row[key] for key in (
                    "L", "seed", "variant", "elapsed", "first_new_seconds", "new_pairs")}), flush=True)
                atomic_write_json(args.output / "summary.json", {"summary": summarize(rows)})
    for row in summarize(rows):
        print(json.dumps(row), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
