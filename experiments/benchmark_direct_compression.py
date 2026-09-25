"""Bounded direct-compression diagnostic plus optional paired lift ablation.

No saved PQCPs are read; no L.txt is written. This tests a candidate producer,
not end-to-end new-solution throughput. No speedup claim from a short run.
"""

import argparse
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from solver.checkpoint import atomic_write_json
from solver.compressed_lift import solve_compressed_lift
from solver.target_profiles import canonical_target_content_profiles
from solver.torch_compressed_search import search_compressed
from solver.torch_device import resolve_torch_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--L", type=int, nargs="+", default=[14, 46, 86])
    parser.add_argument("--seeds", type=int, nargs="+", default=[123, 456])
    parser.add_argument("--device", choices=("cpu", "mps", "cuda", "auto"), default="cpu")
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--lift-jobs", type=int, default=4,
                        help="maximum jobs per run for paired Z3 checks; 0 disables checks")
    parser.add_argument("--lift-timeout-ms", type=int, default=1000)
    parser.add_argument("--orientation-steps", type=int, default=128)
    parser.add_argument("--output", type=Path, default=Path("results/direct_compression_diagnostic.json"))
    args = parser.parse_args()
    if min(args.batch, args.steps, args.candidates, args.orientation_steps, args.lift_timeout_ms) <= 0 or args.lift_jobs < 0:
        parser.error("work controls must be positive; lift-jobs may be zero")
    if any(not 6 <= length <= 94 or length % 4 != 2 for length in args.L):
        parser.error("supported lengths: 6<=L<=94, L=2 mod 4")
    device = resolve_torch_device(args.device)
    rows = []
    for length in args.L:
        ts = canonical_target_content_profiles(length)
        if not ts:
            parser.error("L={} has no necessary target content profiles".format(length))
        for seed in args.seeds:
            started = perf_counter()
            result = search_compressed(tuple(ts[i % len(ts)] for i in range(args.batch)),
                device=device, seed=seed, steps=args.steps, candidates=args.candidates)
            elapsed = perf_counter() - started  # includes CPU export and exact checks
            row = dict(L=length, seed=seed, device=str(device), elapsed=elapsed,
                       proposals=result.proposals, minimum_energy=int(result.energies.min().item()),
                       mean_energy=float(result.energies.mean().item()), exact_jobs=len(result.jobs),
                       jobs_per_minute=60*len(result.jobs)/elapsed, lifts=[])
            for i, job in enumerate(result.jobs[:args.lift_jobs]):
                pair = []
                # Alternate order; same exact job, time limit and unconstrained Z3 model.
                for navigation in ((0, args.orientation_steps) if i % 2 == 0 else (args.orientation_steps, 0)):
                    r = solve_compressed_lift(job.compressed_a, job.compressed_b, job.target.k,
                        job.target.eta, args.lift_timeout_ms, job.target,
                        preferred_pair=(job.source_a, job.source_b), orientation_search_steps=navigation,
                        orientation_search_seed=seed ^ int(job.identifier[:8], 16))
                    pair.append(dict(orientation_steps=navigation, status=r.status, verified=r.verified,
                                     z3_executed=r.z3_executed, gf2_rank=r.gf2_rank,
                                     gf2_free_bits=r.gf2_free_bits,
                                     elapsed=r.construction_time+r.solve_time,
                                     orientation_evaluations=r.orientation_search_evaluations,
                                     orientation_feasible=r.orientation_search_feasible,
                                     A=list(r.a) if r.a else None, B=list(r.b) if r.b else None))
                row["lifts"].append({"job": job.identifier, "runs": pair})
            rows.append(row)
            print("L={} seed={} time={:.3f}s min_E={} exact_jobs={} lift_pairs={}".format(
                  length, seed, elapsed, row["minimum_energy"], len(result.jobs), len(row["lifts"])), flush=True)
            atomic_write_json(args.output, {"kind": "bounded_diagnostic_not_speedup_proof",
                "batch": args.batch, "steps": args.steps, "candidates": args.candidates,
                "lift_timeout_ms": args.lift_timeout_ms, "runs": rows})
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
