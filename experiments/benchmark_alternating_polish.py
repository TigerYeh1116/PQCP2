"""Bounded fixed-work diagnostic, NOT an end-to-end new-PQCP speed benchmark.

Both modes receive identical random fixed-content words, no saved solutions.
Directions cost two delta evaluations per proposal; report work and time, not
just proposal counts. CPU is for correctness; choose MPS/CUDA on the target host.
"""

import argparse
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from solver.checkpoint import atomic_write_json
from solver.compressed_completion import CompressedLiftPool
from solver.correlation import full_correlation_profile
from solver.objective import pqcp_objective
from solver.target_profiles import target_content_profiles
from solver.torch_alternating import alternating_polish_stream, fold_error
from solver.torch_device import resolve_torch_device
from solver.torch_polish import polish_stream
from solver.torch_search import BatchedPQCP
from solver.verifier import verify_pqcp


def run_case(length, seed, batch, steps, candidates, phase_steps, device):
    """Fresh independent run; compare both modes in alternating execution order."""
    targets = target_content_profiles(length, decimation_reduced=True)
    model = BatchedPQCP(tuple(targets[i % len(targets)] for i in range(batch)), device)
    initial = model.project(torch.randn(batch, 2, length,
                            generator=torch.Generator().manual_seed(seed)).to(device))
    rows = []
    for mode in (("baseline", "alternating") if seed % 2 else ("alternating", "baseline")):
        # Synchronize initialization without relying on device-specific APIs.
        model.correlation(initial).cpu().tolist()
        started = perf_counter()
        options = dict(steps=steps, candidates=candidates, seed=seed + 8000003)
        if mode == "baseline":
            result = polish_stream(model, initial, **options, block_steps=steps,
                                   proposal_policy="opposite", kick_interval=min(250, steps))
        else:
            result = alternating_polish_stream(model, initial, **options, phase_steps=phase_steps)
        words = ((1 - result.signs) / 2).int().cpu().tolist()
        profiles = result.profile.int().cpu().tolist()
        scores = result.scores.int().cpu().tolist()
        compression_words = ([] if result.compression_signs is None else
                             ((1 - result.compression_signs) / 2).int().cpu().tolist())
        elapsed = perf_counter() - started
        pool = CompressedLiftPool(length)
        jobs, verified, exact_lanes = set(), 0, 0
        for i, (a, b) in enumerate(words + compression_words):
            profile = full_correlation_profile(a, b)
            target = model.profiles[i % batch]
            if i < batch:
                assert profile == profiles[i] and pqcp_objective(profile) == scores[i]
            verified += int(verify_pqcp(a, b).is_valid)
            for job in pool.offer(a, b, target.k, target.eta):
                jobs.add(job.canonical_key)
                pool.submit(job)
        observed = result.compression_profile if result.compression_profile is not None else result.profile
        errors = fold_error(observed, model.targets).cpu().tolist()
        exact_lanes = sum(error == 0 for error in errors)
        rows.append(dict(L=length, seed=seed, method=mode, batch=batch, steps=steps,
                         candidates=candidates, phase_steps=phase_steps, device=str(device),
                         elapsed=elapsed, proposals=result.proposals,
                         evaluation_seconds=perf_counter() - started - elapsed,
                         delta_evaluations=result.proposals + result.direction_proposals,
                         accepted=result.accepted_swaps, direction_accepted=result.direction_accepted,
                         minimum_score=min(scores), mean_score=sum(scores) / len(scores),
                         mean_fold_error=sum(errors) / len(errors), exact_compression_lanes=exact_lanes,
                         unique_compressed_jobs=len(jobs), verified_candidates=verified,
                         exact_profiles=True))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--L", type=int, nargs="+", default=[46, 86])
    parser.add_argument("--seeds", type=int, nargs="+", default=[123, 456])
    parser.add_argument("--device", choices=["cpu", "mps", "cuda", "auto"], default="cpu")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--phase-steps", type=int, default=16)
    parser.add_argument("--output", type=Path, default=Path("results/alternating_polish_diagnostic.json"))
    args = parser.parse_args()
    if any(length < 6 or length % 4 != 2 for length in args.L):
        parser.error("lengths must be >=6 and 2 mod 4")
    if min(args.batch, args.steps, args.candidates, args.phase_steps) <= 0:
        parser.error("work controls must be positive")
    device = resolve_torch_device(args.device)
    rows = []
    for length in args.L:
        for seed in args.seeds:
            rows.extend(run_case(length, seed, args.batch, args.steps, args.candidates, args.phase_steps, device))
    atomic_write_json(args.output, {"kind": "fixed_work_diagnostic_not_speedup_proof", "runs": rows})
    print("L   seed mode         polish-s best mean-score fold-error exact-lanes jobs")
    for row in rows:
        print("{L:<3} {seed:<4} {method:<13} {elapsed:7.3f} {minimum_score:4} {mean_score:10.1f} "
              "{mean_fold_error:10.1f} {exact_compression_lanes:11} {unique_compressed_jobs:4}".format(**row))
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
