"""Paired ablation of genuinely new PQCP mathematical loss components.

This first measures exact projected candidate quality, not solution-rate
speedup.  Each arm has identical FKM initialization, Adam settings, batch,
wall-clock budget, exact archive scoring and optional discrete polish.  Known
PQCP files are not read as seeds.  A weak short result is diagnostic only;
end-to-end verified-pair trials are required before changing defaults.
"""

import argparse
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.benchmark_torch_loss import run_case
from solver.checkpoint import atomic_write_json
from solver.torch_hybrid_runner import TorchHybridConfig
from solver.torch_search import require_device


MODES = ("none", "psd_cap", "divisor_lift", "lattice", "variance", "combined",
         "lattice_bootstrap")


def summarize(rows: list) -> list:
    """Summarize only exact integer post-projection/post-polish quantities."""
    result = []
    for length, mode in sorted({(r["L"], r["mathematical_loss"]) for r in rows}):
        group = [r for r in rows if r["L"] == length and r["mathematical_loss"] == mode]
        result.append({
            "L": length, "mathematical_loss": mode, "runs": len(group),
            "projected_min": min(r["projected_best_score"] for r in group),
            "projected_median": statistics.median(r["projected_best_score"] for r in group),
            "target_median": statistics.median(r["target_best"] for r in group),
            "post_polish_median": statistics.median(r["post_polish_best_score"] for r in group),
            "epochs_median": statistics.median(r["epochs"] for r in group),
            "new_solutions": sum(r["new_solutions"] for r in group),
        })
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", type=int, nargs="+", default=[44, 46, 68])
    parser.add_argument("--seeds", type=int, nargs="+", default=[123, 456, 789])
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--seconds", type=float, default=20)
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--no-polish", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if not 0 < args.seconds < float("inf") or len(set(args.seeds)) != len(args.seeds):
        parser.error("finite positive seconds and distinct seeds required")
    require_device(args.device)
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for length in args.lengths:
        for index, seed in enumerate(args.seeds):
            order = args.modes[index % len(args.modes):] + args.modes[:index % len(args.modes)]
            fingerprints = set()
            for mode in order:
                root = args.output / "L{}_seed{}_{}".format(length, seed, mode)
                config = TorchHybridConfig(length, seed=seed, device=args.device,
                                           continuous_batch_size=args.batch, root=root)
                row = run_case(config, "balanced", args.seconds,
                               polish=not args.no_polish, mathematical_loss=mode)
                fingerprints.add(row["initial_fingerprint"])
                rows.append(row)
                with (args.output / "runs.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row) + "\n")
                print(json.dumps({key: row[key] for key in (
                    "L", "seed", "mathematical_loss", "epochs",
                    "projected_best_score", "target_best", "post_polish_best_score",
                    "new_solutions")}), flush=True)
            if len(fingerprints) != 1:
                raise RuntimeError("paired mathematical losses did not share initialization")
            atomic_write_json(args.output / "summary.json", {"summary": summarize(rows)})
    for row in summarize(rows):
        print(json.dumps(row), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
