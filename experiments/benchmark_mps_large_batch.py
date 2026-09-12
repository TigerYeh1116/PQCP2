"""Isolated large-batch CUDA ablation without writing giant checkpoints."""

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

from solver.checkpoint import atomic_write_json
from solver.correlation import full_correlation_profile
from solver.objective import pqcp_objective
from solver.search_runner import append_verified_solution_if_new
from solver.torch_polish import discrete_energy, polish_stream
from solver.torch_search import BatchedPQCP, TorchSearch, TorchSearchConfig, require_device
from solver.verifier import verify_pqcp


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--L", type=int, default=44)
    parser.add_argument("--seconds", type=float, default=25)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--batch", type=int, default=8192)
    parser.add_argument("--observe", type=int, default=25)
    parser.add_argument("--mode", choices=("relaxed", "straight_through"), default="relaxed")
    parser.add_argument("--polish-elites", type=int, default=0)
    parser.add_argument("--polish-steps", type=int, default=1000)
    parser.add_argument("--polish-candidates", type=int, default=8)
    parser.add_argument("--polish-block", type=int, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    require_device("cuda")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    official = ROOT / "{}.txt".format(args.L)
    if official.exists():
        shutil.copy2(official, output / official.name)
    config = TorchSearchConfig(
        args.L, seed=args.seed, batch_size=args.batch,
        observation_interval=args.observe, continuous_kernel="dft",
        optimization_mode=args.mode,
    )
    initialized = perf_counter()
    search = TorchSearch(config, output)
    initialization_seconds = perf_counter() - initialized
    started = perf_counter()
    search.observe()
    progression = [{"elapsed": 0.0, "epoch": 0, "score": search.best["score"]}]
    while perf_counter() - started < args.seconds:
        search.step()
        search.elapsed = perf_counter() - started
        if search.epoch % args.observe == 0:
            before = search.best["score"]
            search.observe()
            if search.best["score"] < before:
                progression.append({
                    "elapsed": search.elapsed, "epoch": search.epoch,
                    "score": search.best["score"],
                })
                print(progression[-1], flush=True)
    if search.last_observed_epoch != search.epoch:
        search.observe(update_stagnation=False)
    torch.cuda.synchronize()
    adam_elapsed = perf_counter() - started
    polish_report = None
    if args.polish_elites:
        if not 0 < args.polish_elites <= args.batch:
            raise ValueError("polish elites must be in 1..batch")
        with torch.no_grad():
            signs = search.model.project(search.theta)
            hard_profile = search.model.correlation(signs)
            scores = search.model.discrete_scores(hard_profile)
            energies = discrete_energy(
                hard_profile, search.model.targets,
                compression=config.compression,
            )
            # Completion pursues each lane's fixed Project target, therefore
            # its input must be ranked by that same exact target energy.  The
            # old unrestricted-objective ranking could select a pair close to
            # a different k and hand it to the wrong basin.
            selected = torch.topk(energies, args.polish_elites, largest=False).indices
            selected_signs = signs[selected]
            selected_profiles = tuple(search.model.profiles[int(i)] for i in selected.cpu().tolist())
            polish_model = BatchedPQCP(selected_profiles, search.device)
            torch.cuda.synchronize()
            polish_started = perf_counter()
            polished = polish_stream(
                polish_model, selected_signs, steps=args.polish_steps,
                candidates=args.polish_candidates, seed=args.seed + 8_000_003,
                block_steps=min(args.polish_block, args.polish_steps), proposal_policy="opposite",
                kick_interval=min(args.polish_block, 1000, args.polish_steps),
                compression=not args.full_only if hasattr(args, "full_only") else True,
                progress=lambda completed, best, proposals: print(
                    "polish steps={} best={} proposals={}".format(
                        completed, best, proposals
                    ), flush=True,
                ),
            )
            torch.cuda.synchronize()
            polish_elapsed = perf_counter() - polish_started
            polished_scores = polished.scores.to(torch.int64).cpu().tolist()
            verified_count = new_count = 0
            for lane in [i for i, score in enumerate(polished_scores) if score == 0]:
                a, b = ((1 - polished.signs[lane]) / 2).to(torch.int32).cpu().tolist()
                exact = full_correlation_profile(a, b)
                if pqcp_objective(exact) or not verify_pqcp(a, b).is_valid:
                    raise RuntimeError("polished CUDA zero failed independent verifier")
                verified_count += 1
                new_count += append_verified_solution_if_new(args.L, tuple(a), tuple(b), output)
            polish_report = {
                "elites": args.polish_elites, "steps": args.polish_steps,
                "candidates": args.polish_candidates, "elapsed": polish_elapsed,
                "proposals": polished.proposals, "best_score": min(polished_scores),
                "input_min_score": int(scores[selected].min().item()),
                "input_min_energy": int(energies[selected].min().item()),
                "input_max_energy": int(energies[selected].max().item()),
                "verified": verified_count, "new_solutions": new_count,
            }
            print("polish {}".format(polish_report), flush=True)
    elapsed = perf_counter() - started
    verification = verify_pqcp(search.best["A"], search.best["B"])
    report = {
        "L": args.L, "seed": args.seed, "mode": args.mode, "batch": args.batch,
        "kernel": "dft", "initialization_seconds": initialization_seconds,
        "adam_elapsed": adam_elapsed, "elapsed": elapsed, "epochs": search.epoch,
        "projected_evaluations": search.candidate_evaluations,
        "best_score": search.best["score"], "verified": verification.is_valid,
        "new_solutions": search.new_solutions, "progression": progression,
        "best": search.best, "polish": polish_report,
    }
    atomic_write_json(output / "report.json", report)
    print(json.dumps({key: value for key, value in report.items() if key != "best"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
