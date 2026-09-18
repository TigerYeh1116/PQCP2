"""Isolated benchmark of MPS-expanded verified neighborhoods plus completion."""

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
from solver.torch_hybrid import write_elite_pair_seed_bank
from solver.torch_seed_expansion import (
    SeedExpansionConfig, expand_solution_neighborhoods, load_verified_solution_seeds,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--L", type=int, default=44)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--seconds", type=float, default=12)
    parser.add_argument("--clones", type=int, default=8)
    parser.add_argument("--radius", type=int, default=8)
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    official = ROOT / "{}.txt".format(args.L)
    shutil.copy2(official, output / official.name)
    started = perf_counter()
    known = load_verified_solution_seeds(official, args.L)
    expanded = expand_solution_neighborhoods(known, SeedExpansionConfig(
        args.L, seed=args.seed, clones_per_solution=args.clones,
        max_swap_radius=args.radius, limit=args.limit,
    ))
    pair_bank = output / "results" / "mps_expanded_seeds.txt"
    records = write_elite_pair_seed_bank(
        expanded, pair_bank, length=args.L, retarget=False,
    )
    formation = perf_counter() - started
    result = run_c_search(CSearchConfig(
        L=args.L, seed=args.seed, seconds=args.seconds, threads=8,
        root=output, echo=False, pair_seed_path=pair_bank,
    ))
    print(json.dumps({
        "L": args.L, "seed": args.seed, "known": len(known),
        "expanded": len(records), "formation_seconds": formation,
        "completion_seconds": result.elapsed,
        "verified_candidates": result.verified_candidates,
        "new_solutions": result.new_solutions,
        "swap_evaluations": result.swap_evaluations,
        "total_seconds": perf_counter() - started,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
