"""Bounded, isolated end-to-end benchmark of MPS population search."""

import argparse
from pathlib import Path
import shutil
import sys
from time import perf_counter


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from solver.correlation import full_correlation_profile
from solver.objective import pqcp_objective
from solver.search_runner import append_verified_solution_if_new
from solver.torch_population import PopulationSearchConfig, TorchPopulationSearch
from solver.verifier import verify_pqcp


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--L", type=int, default=44)
    parser.add_argument("--seconds", type=float, default=25)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--islands", type=int, default=8)
    parser.add_argument("--population", type=int, default=512)
    parser.add_argument("--elites", type=int, default=48)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    official = ROOT / "{}.txt".format(args.L)
    if official.exists():
        shutil.copy2(official, output / official.name)
    search = TorchPopulationSearch(PopulationSearchConfig(
        args.L, seed=args.seed, islands_per_profile=args.islands,
        population_size=args.population, elite_count=args.elites,
    ))
    started = perf_counter()
    verified = new = 0
    while perf_counter() - started < args.seconds:
        batch = search.step()
        zero = (batch.scores == 0).nonzero().cpu().tolist()
        for distribution, member in zero:
            a = batch.bits[distribution, member, 0].to(torch.int32).cpu().tolist()
            b = batch.bits[distribution, member, 1].to(torch.int32).cpu().tolist()
            exact = full_correlation_profile(a, b)
            if pqcp_objective(exact) or not verify_pqcp(a, b).is_valid:
                raise RuntimeError("population zero failed independent verifier")
            verified += 1
            new += append_verified_solution_if_new(args.L, tuple(a), tuple(b), output)
        print(
            "generation={} elapsed={:.3f}s candidates={} best={} energy={} verified={} new={}".format(
                search.generation, perf_counter() - started, search.candidate_evaluations,
                search.best["score"], search.best["energy"], verified, new,
            ), flush=True,
        )
        del batch
    elapsed = perf_counter() - started
    print(
        "FINAL elapsed={:.3f}s generations={} candidates={} rate={:.0f}/s "
        "best={} verified={} new={}".format(
            elapsed, search.generation, search.candidate_evaluations,
            search.candidate_evaluations / elapsed, search.best["score"], verified, new,
        ), flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
