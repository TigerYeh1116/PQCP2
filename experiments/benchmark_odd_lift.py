"""Paired old/new Z3 encoding benchmark, NOT time-to-new-PQCP measurement.

Read saved compressed jobs only (ignore source sequences, solutions, status).
Without --input-dir, use bounded exhaustive small ternary catalogs.
Never write official L.txt files or mutate production checkpoints.
"""

import argparse
from dataclasses import asdict
from itertools import product
import json
from pathlib import Path
import random
import statistics
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from solver.checkpoint import atomic_write_json
from solver.compressed_lift import solve_compressed_lift
from solver.structured_energy import (
    fold_profile, integer_periodic_autocorrelation_profile as pacf, target_profile,
)
from solver.target_profiles import TargetContentProfile


def small_requests(L, limit):
    if L not in (6, 10, 14):
        raise ValueError("without saved jobs, use L=6,10,14; no large exhaustive generation")
    words = list(product((-2, 0, 2), repeat=L // 2))
    random.Random(L).shuffle(words)
    signatures = {w: pacf(w) for w in words}
    buckets = {}
    for word, sig in signatures.items():
        buckets.setdefault(sig, []).append(word)
    requests = []
    # Interleave target classes rather than fill the quota with one target.
    generators = []
    def matching(k, eta):
        wanted = fold_profile(target_profile(L, k, eta), 2)
        for a in words:
            complement = tuple(t - x for t, x in zip(wanted, signatures[a]))
            for b in buckets.get(complement, ()):
                yield dict(compressed_a=a, compressed_b=b, k=k, eta=eta, content=None)
    for k, eta in product(range(1, L // 2), (-1, 1)):
        generators.append(matching(k, eta))
    while generators and len(requests) < limit:
        survivors = []
        for iterator in generators:
            request = next(iterator, None)
            if request is not None:
                requests.append(request)
                survivors.append(iterator)
                if len(requests) == limit:
                    break
        generators = survivors
    return requests


def saved_requests(directory, L, limit):
    requests, seen = [], set()
    for path in sorted(directory.glob("L{}_*.json".format(L))):
        data = json.loads(path.read_text())
        if data.get("factor", 2) != 2:
            continue
        a, b = tuple(data["compressed_a"]), tuple(data["compressed_b"])
        if len(a) * 2 != L or len(b) != len(a):
            continue
        content = data["content_profile"]
        p = TargetContentProfile(**content)
        if p.L != L:
            raise ValueError("mismatched saved target length")
        key = (a, b, p)
        if key in seen:
            continue
        seen.add(key)
        requests.append(dict(compressed_a=a, compressed_b=b, k=p.k, eta=p.eta,
                             content=content, source=str(path)))
        if len(requests) >= limit:
            break
    return requests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--L", type=int, default=14)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout-ms", type=int, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.L < 6 or args.L % 4 != 2 or min(args.limit, args.repeats, args.timeout_ms) < 1:
        parser.error("positive limits and L=2 mod 4, L>=6 required")
    if args.output.exists():
        parser.error("choose a new output path; existing results are not overwritten")
    try:
        requests = (saved_requests(args.input_dir, args.L, args.limit) if args.input_dir
                    else small_requests(args.L, args.limit))
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))
    if not requests:
        parser.error("no saved matched jobs: cannot claim an end-to-end benchmark")
    # Exclude optional-module import cost from the paired timings.
    import z3
    from solver.orientation_gf2 import odd_orientation_system
    z3.SolverFor("QF_FD")
    rows, pairs = [], []
    tick = perf_counter()
    for repeat in range(args.repeats):
        for index, request in enumerate(requests):
            p = TargetContentProfile(**request["content"]) if request["content"] else None
            outcomes, times = {}, {}
            order = (False, True) if (repeat + index) % 2 == 0 else (True, False)
            for enabled in order:
                started = perf_counter()
                result = solve_compressed_lift(
                    request["compressed_a"], request["compressed_b"], request["k"], request["eta"],
                    args.timeout_ms, p, orientation_preprocessing=enabled)
                elapsed = perf_counter() - started
                outcomes[enabled], times[enabled] = result, elapsed
                rows.append(dict(case=index, repeat=repeat, preprocessing=enabled,
                                 elapsed=elapsed, result=asdict(result)))
            old, new = outcomes[False], outcomes[True]
            complete = old.status != "UNKNOWN" and new.status != "UNKNOWN"
            if complete and old.status != new.status:
                raise RuntimeError("old/new exact classification disagreement")
            speedup = times[False] / times[True] if complete else None
            pairs.append(dict(case=index, repeat=repeat, complete=complete, speedup=speedup))
            print("case={} repeat={} old={} {:.4f}s new={} {:.4f}s rank={} free={}".format(
                index, repeat, old.status, times[False], new.status, times[True],
                new.gf2_rank, new.gf2_free_bits), flush=True)
    completed = [p["speedup"] for p in pairs if p["complete"]]
    summary = dict(L=args.L, cases=len(requests), repeats=args.repeats,
                   elapsed=perf_counter() - tick, completed_pairs=len(completed),
                   median_completed_speedup=statistics.median(completed) if completed else None,
                   old_unknown=sum(r["result"]["status"] == "UNKNOWN" for r in rows if not r["preprocessing"]),
                   new_unknown=sum(r["result"]["status"] == "UNKNOWN" for r in rows if r["preprocessing"]))
    atomic_write_json(args.output, dict(summary=summary, requests=requests, rows=rows, pairs=pairs,
                     timeout_ms=args.timeout_ms,
                     caveat="Fixed compressed-job benchmark only; not new-PQCP discovery speed."))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
