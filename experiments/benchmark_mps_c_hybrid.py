"""Isolated CUDA large-batch formation followed by exact compiled completion."""

import argparse
import json
from pathlib import Path
import shutil
import sys
from time import perf_counter


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from solver.c_backend import CSearchConfig, run_c_search
from solver.correlation import full_correlation_profile
from solver.objective import pqcp_objective
from solver.torch_hybrid import write_elite_pair_seed_bank
from solver.torch_polish import discrete_energy
from solver.torch_search import TorchSearch, TorchSearchConfig, require_device


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--L", type=int, default=44)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--cuda-seconds", "--mps-seconds", dest="cuda_seconds", type=float, default=4)
    parser.add_argument("--c-seconds", type=float, default=20)
    parser.add_argument("--batch", type=int, default=8192)
    parser.add_argument("--elites", type=int, default=2000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    require_device("cuda")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    official = ROOT / "{}.txt".format(args.L)
    if official.exists():
        shutil.copy2(official, output / official.name)
    started = perf_counter()
    search = TorchSearch(TorchSearchConfig(
        args.L, seed=args.seed, batch_size=args.batch,
        continuous_kernel="dft", optimization_mode="relaxed",
    ), output)
    while perf_counter() - started < args.cuda_seconds:
        search.step()
    with torch.no_grad():
        signs = search.model.project(search.theta)
        profiles = search.model.correlation(signs)
        scores = search.model.discrete_scores(profiles)
        energies = discrete_energy(profiles, search.model.targets, compression=True)
        selected = torch.topk(energies, args.elites, largest=False).indices.cpu().tolist()
        bits = ((1 - signs[selected]) / 2).to(torch.int32).cpu().tolist()
        gpu_profiles = profiles[selected].to(torch.int32).cpu().tolist()
    records = []
    for rank, (lane, pair, profile) in enumerate(zip(selected, bits, gpu_profiles)):
        exact = full_correlation_profile(pair[0], pair[1])
        if exact != profile:
            raise RuntimeError("CUDA elite profile failed full recomputation")
        target = search.model.profiles[lane]
        records.append({
            "A": pair[0], "B": pair[1], "profile": exact,
            "score": pqcp_objective(exact), "iteration": rank,
            "k": target.k, "sign": target.eta,
        })
    pair_bank = output / "results" / "cuda_target_elites.txt"
    written = write_elite_pair_seed_bank(
        records, pair_bank, length=args.L, retarget=False,
    )
    cuda_elapsed = perf_counter() - started
    del search, signs, profiles, scores, energies
    torch.cuda.empty_cache()
    result = run_c_search(CSearchConfig(
        L=args.L, seed=args.seed, seconds=args.c_seconds, threads=8,
        root=output, echo=True, pair_seed_path=pair_bank,
    ))
    print(json.dumps({
        "L": args.L, "seed": args.seed, "cuda_elapsed": cuda_elapsed,
        "cuda_elites": len(written), "completion_elapsed": result.elapsed,
        "swap_evaluations": result.swap_evaluations,
        "verified_candidates": result.verified_candidates,
        "new_solutions": result.new_solutions,
        "total_elapsed": perf_counter() - started,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
