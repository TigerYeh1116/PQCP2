"""Find a NEW L44 equivalence class using coarse MITM -> middle MITM -> Z3.

Experimental, not a replacement for main.py. Generate coarse candidates from
fixed-content FKM, never from 44.txt. Read 44.txt only for duplicate rejection
in the existing independently verified atomic writer. Continue until a new
solution is appended, or Ctrl+C. Every attempt is logged before moving on;
UNKNOWN is retained as unresolved. No SAT/UNSAT inference from time limits.
"""

import argparse
from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys
from time import perf_counter

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from experiments.benchmark_middle_mitm import bounded_run
from experiments.research_compressed_mitm import (
    coarse_row_pairs, content_necklaces, contents, margin_compatible,
)
from solver.checkpoint import atomic_write_json
from solver.compressed_equivalence import compressed_job_equivalence_key
from solver.frequency_constraints import quarter_frequency_feasible
from solver.search_runner import append_verified_solution_if_new
from solver.structured_energy import (
    compress_binary_sequence, fold_profile,
    integer_periodic_autocorrelation_profile as pacf, target_profile,
)
from solver.target_profiles import canonical_target_content_profiles, pair_content
from solver.verifier import verify_pqcp


def coarse_jobs(profile):
    """Complete coarse necklace join for a fixed L44 target/content case.

    Uses only exact row-sum, squared-norm, and CRT margin necessary conditions.
    This reuses the small research generator; no known PQCP or SA seed input.
    """
    if profile.L != 44:
        raise ValueError("this bounded coarse enumeration is for L44 only")
    n = 11
    target = tuple(t//4 for t in fold_profile(target_profile(44,profile.k,profile.eta),4))
    rows = coarse_row_pairs(profile)
    sides = []
    for side,weight in enumerate(profile.weight_pair):
        values = []
        for hist in contents(n):
            if sum(c*x for c,x in zip(hist,(-2,-1,0,1,2))) != (44-2*weight)//2:
                continue
            norm = sum(c*x*x for c,x in zip(hist,(-2,-1,0,1,2)))
            mask = sum(1 << i for i,pair in enumerate(rows) if margin_compatible(hist,pair[side]))
            values.append((hist,norm,mask))
        sides.append(values)
    live = [set(),set()]
    for ha,qa,ma in sides[0]:
        for hb,qb,mb in sides[1]:
            if qa+qb == target[0] and ma & mb:
                live[0].add(ha)
                live[1].add(hb)
    masks = [{h:m for h,_,m in values} for values in sides]
    index = defaultdict(list)
    for hist in sorted(live[1]):
        for b in content_necklaces(hist):
            index[pacf(b)[:6]].append((b,masks[1][hist]))
    pairs = []
    for hist in sorted(live[0]):
        for a in content_necklaces(hist):
            needed = tuple(t-x for t,x in zip(target[:6],pacf(a)))
            for b,mask in index.get(needed,()):
                if mask & masks[0][hist]:
                    pairs.append((tuple(2*x for x in a),tuple(2*x for x in b)))
    return pairs


def record_sat(result, job, root, output, attempt):
    """Persist the full audit BEFORE the official writer checks novelty."""
    a,b = tuple(result["A"]),tuple(result["B"])
    check = verify_pqcp(a,b)
    if not result["verified"] or not check.is_valid:
        raise RuntimeError("refusing unverified SAT")
    p = job["target"]
    if (check.profile != target_profile(44,p["k"],p["eta"])
            or compress_binary_sequence(a,4) != tuple(job["compressed_a"])
            or compress_binary_sequence(b,4) != tuple(job["compressed_b"])
            or pair_content(a,b) != tuple(p[k] for k in (
                "a_even_ones","a_odd_ones","b_even_ones","b_odd_ones"))):
        raise RuntimeError("SAT changed target, compression or content")
    payload = dict(result,job_spec=job,full_profile=check.profile,
                   nonzero_shifts=check.nonzero_shifts,
                   nonzero_values=[check.profile[u] for u in check.nonzero_shifts])
    snapshot = output / ("sat_{:06d}.json".format(attempt))
    atomic_write_json(snapshot,payload)
    new = append_verified_solution_if_new(44,a,b,root)
    payload.update(new_solution_written=new,snapshot=str(snapshot))
    atomic_write_json(snapshot,payload)
    return new,payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed",type=int,default=123)
    parser.add_argument("--seconds-per-job",type=float,default=12)
    parser.add_argument("--binary-slice-ms",type=int,default=1000)
    parser.add_argument("--output-dir",type=Path,required=True)
    args = parser.parse_args()
    if args.seconds_per_job <= 0 or args.binary_slice_ms <= 0:
        parser.error("positive per-job budgets required")
    root = Path(__file__).resolve().parents[1]
    out = args.output_dir.resolve()
    out.mkdir(parents=True,exist_ok=False)
    if (root/"44.txt").exists():
        shutil.copy2(root/"44.txt",out/"44.before.txt")
    profiles = [p for p in canonical_target_content_profiles(44) if quarter_frequency_feasible(p)]
    rng = random.Random(args.seed)
    started,attempt = perf_counter(),0
    unresolved = []
    seen = set()
    atomic_write_json(out/"run.json",dict(seed=args.seed,seconds_per_job=args.seconds_per_job,
        binary_slice_ms=args.binary_slice_ms,profiles=[asdict(p) for p in profiles],
        source="fresh fixed-content FKM coarse enumeration, no known PQCP seeds"))

    def execute(job, seconds, slice_ms, stream):
        nonlocal attempt
        attempt += 1
        atomic_write_json(out/"current_job.json",dict(job=job,attempt=attempt))
        result = bounded_run(job,"mitm",seconds,250000,slice_ms)
        result.update(attempt=attempt,job_spec=job,total_elapsed=perf_counter()-started)
        stream.write(json.dumps(result,sort_keys=True)+"\n")
        stream.flush()
        print("attempt={} elapsed={:.1f}s job={} status={} middle={} binary={}".format(
            attempt,result["total_elapsed"],job["id"],result["status"],
            result.get("middle_models",0),result.get("binary_started",0)),flush=True)
        if result["status"] == "ERROR":
            raise RuntimeError(result["reason"])
        if result["status"] == "SAT":
            new,payload = record_sat(result,job,root,out,attempt)
            print("verified SAT: {}".format("NEW, appended to 44.txt" if new else "equivalent to an existing solution"),flush=True)
            if new:
                atomic_write_json(out/"new_solution.json",payload)
                return True
        if result["status"] != "UNSAT":
            unresolved.append(job)
        return False

    with (out/"attempts.jsonl").open("x") as stream:
        try:
            for p in profiles:
                pairs = coarse_jobs(p)
                print("fresh coarse matches={} target={}".format(len(pairs),p),flush=True)
                rng.shuffle(pairs)
                for ca,cb in pairs:
                    key = compressed_job_equivalence_key(ca,cb,p,4)
                    if key in seen:
                        continue
                    seen.add(key)
                    job = dict(compressed_a=ca,compressed_b=cb,target=asdict(p),source="fresh coarse FKM/MITM")
                    job["id"] = hashlib.sha256(json.dumps(job,sort_keys=True).encode()).hexdigest()[:20]
                    if execute(job,args.seconds_per_job,args.binary_slice_ms,stream):
                        return
            # UNKNOWN/SAT-duplicate parents remain unresolved for novelty.
            # Raise time allowances rather than interpreting short tests as
            # proofs. New full models within an already-SAT child may require
            # future model blocking; this experiment makes no exhaustion claim.
            round_index = 1
            while unresolved:
                pending,unresolved = unresolved,[]
                rng.shuffle(pending)
                for job in pending:
                    if execute(job,args.seconds_per_job*(2**round_index),
                               args.binary_slice_ms*(2**round_index),stream):
                        return
                round_index += 1
            raise RuntimeError("all tested parents exhausted; no new solution found")
        except KeyboardInterrupt:
            print("Interrupted; attempts and current job preserved in {}".format(out),flush=True)


if __name__ == "__main__":
    main()
