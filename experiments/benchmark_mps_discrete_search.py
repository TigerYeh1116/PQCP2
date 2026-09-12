"""End-to-end CUDA discrete-search diagnostic from fresh FKM/content seeds.

This is an isolated benchmark, not the one-click entry point.  It measures
whether batched exact same-parity swap search can turn fresh PyTorch lanes into
independently verified PQCPs.  The official ``L.txt`` is copied and never
modified by this experiment.
"""

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
from solver.torch_polish import polish_batch
from solver.torch_search import TorchSearch, TorchSearchConfig, require_device
from solver.verifier import verify_pqcp


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--L", type=int, default=44)
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--candidates", type=int, default=32)
    parser.add_argument("--kick-interval", type=int, default=100)
    parser.add_argument("--adam-steps", type=int, default=0)
    parser.add_argument("--full-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if min(args.seconds, args.batch, args.steps, args.candidates) <= 0:
        raise ValueError("positive budget and batch controls are required")
    if args.adam_steps < 0 or args.kick_interval < 0:
        raise ValueError("adam steps and kick interval must be nonnegative")
    device = require_device("cuda")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    official = ROOT / "{}.txt".format(args.L)
    if official.exists():
        shutil.copy2(official, output / official.name)
    search = TorchSearch(TorchSearchConfig(
        args.L, seed=args.seed, device="cuda", batch_size=args.batch,
        fkm_pool_size=64, continuous_kernel="dft",
        compression=not args.full_only,
    ), output)
    generator = torch.Generator().manual_seed(args.seed + 9_000_001)
    started = perf_counter()
    generations = proposals = verified = new = 0
    best_score = None
    while perf_counter() - started < args.seconds:
        for _ in range(args.adam_steps):
            search.step()
        initial = search.model.project(search.theta.detach())
        draws = torch.rand(
            args.steps, args.batch, args.candidates, 4, generator=generator,
        ).to(device)
        torch.cuda.synchronize()
        result = polish_batch(
            search.model, initial, draws, proposal_policy="opposite",
            kick_interval=args.kick_interval, compression=not args.full_only,
        )
        torch.cuda.synchronize()
        proposals += result.proposals
        scores = result.scores.to(torch.int64).cpu().tolist()
        batch_best = min(scores)
        best_score = batch_best if best_score is None else min(best_score, batch_best)
        zero_lanes = [index for index, score in enumerate(scores) if score == 0]
        if zero_lanes:
            pairs = ((1 - result.signs[zero_lanes]) / 2).to(torch.int32).cpu().tolist()
            for a, b in pairs:
                profile = full_correlation_profile(a, b)
                if pqcp_objective(profile) != 0 or not verify_pqcp(a, b).is_valid:
                    raise RuntimeError("CUDA zero-score candidate failed independent verification")
                verified += 1
                new += append_verified_solution_if_new(args.L, tuple(a), tuple(b), output)
        generations += 1
        elapsed = perf_counter() - started
        print(
            "generation={} elapsed={:.2f}s proposals={} best={} verified={} new={}".format(
                generations, elapsed, proposals, best_score, verified, new,
            ), flush=True,
        )
        del draws, result, initial
        torch.cuda.empty_cache()
        search.generation += 1
        search._initialize_batch()  # benchmark-only rebirth; production wrapper owns this lifecycle
    print(
        "FINAL elapsed={:.3f}s generations={} proposals={} rate={:.0f}/s "
        "best={} verified={} new={}".format(
            perf_counter() - started, generations, proposals,
            proposals / max(perf_counter() - started, 1e-9), best_score, verified, new,
        ), flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
