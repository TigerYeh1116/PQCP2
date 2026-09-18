"""Measure exact discrete completion of persisted PyTorch/MPS elites.

The benchmark is deliberately isolated: it copies the current ``L.txt`` into
an output directory and never changes the official result file.  Every C hit
still passes the normal independent Python verifier and A/B-swap deduplication.
"""

import argparse
import json
from pathlib import Path
import shutil
import sys
from time import perf_counter


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from solver.c_backend import CSearchConfig, run_c_search
from solver.checkpoint import atomic_write_json
from solver.torch_hybrid import write_elite_pair_seed_bank


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints" / "L44_torch.json")
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=576)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "torch_hybrid_benchmark")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.seconds <= 0 or args.threads <= 0:
        raise ValueError("seconds and threads must be positive")
    data = json.loads(args.checkpoint.read_text(encoding="utf-8"))
    if data.get("format") != "pqcp-torch-search":
        raise ValueError("not a PyTorch search checkpoint")
    length = int(data["config"]["L"])
    destination = args.output.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    official = ROOT / "{}.txt".format(length)
    if official.is_file():
        shutil.copy2(official, destination / official.name)
    pair_bank = destination / "results" / "torch_elites_L{}.txt".format(length)
    records = write_elite_pair_seed_bank(
        data["archive"], pair_bank, length=length,
        compression=bool(data["config"].get("compression", True)),
    )
    before = sum(1 for line in (destination / official.name).read_text().splitlines()
                 if line.startswith("a=")) if official.is_file() else 0
    started = perf_counter()
    result = run_c_search(CSearchConfig(
        L=length, seed=args.seed, seconds=args.seconds, threads=args.threads,
        root=destination, echo=True, pair_seed_path=pair_bank,
    ))
    wall_time = perf_counter() - started
    report = {
        "L": length, "checkpoint": str(args.checkpoint.resolve()),
        "source_elapsed": data.get("elapsed"), "source_best_score": data.get("best", {}).get("score"),
        "elite_count": len(records), "seed": args.seed, "seconds": args.seconds,
        "threads": args.threads, "wall_time": wall_time,
        "restarts": result.restarts, "moves": result.moves,
        "swap_evaluations": result.swap_evaluations,
        "verified_candidates": result.verified_candidates,
        "new_solutions": result.new_solutions,
        "preexisting_solution_count": before,
        "pair_seed_bank": str(pair_bank), "isolated_root": str(destination),
    }
    atomic_write_json(destination / "benchmark.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
