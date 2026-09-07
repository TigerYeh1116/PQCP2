"""Paired production-C search benchmark with independent solution verification.

Both executables are built from the same source; PQCP_REFERENCE_KERNEL selects
the original arithmetic. Runs are sequential and counterbalanced. The same
compressed FKM bank, seed, thread count and candidate count are used. Existing
L.txt pairs (including A/B exchange) are excluded, without modifying those
files. Timeouts are recorded as censored, never as a discovery.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import queue
import re
import platform
import signal
import subprocess
import sys
import tempfile
import threading
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from solver.c_backend import ensure_compressed_fkm_seed_bank
from solver.checkpoint import atomic_write_json
from solver.verifier import verify_pqcp
from solver.search_runner import append_verified_solution_if_new


def pair_key(a, b):
    """Use the production equality convention: exact bits or A/B exchange."""
    return tuple(sorted((a, b)))


def summarize(records):
    """Keep censored first-hit means undefined and remove cross-run duplicates."""
    groups = {}
    for row in records:
        groups.setdefault((row["L"], row["method"]), []).append(row)
    summary = []
    for (length, method), rows in sorted(groups.items()):
        exposure = sum(row["elapsed"] for row in rows)
        unique = {pair_key(hit["A"], hit["B"])
                  for row in rows for hit in row["hits"]}
        all_hit = all(row["first_new_seconds"] is not None for row in rows)
        summary.append({
            "L": length, "method": method, "trials": len(rows),
            "total_elapsed": exposure, "unique_new_pairs": len(unique),
            "per_run_new_hits": sum(row["new_solutions"] for row in rows),
            "seconds_per_unique_new_pair": exposure / len(unique) if unique else None,
            "first_hit_mean_if_uncensored": (
                sum(row["first_new_seconds"] for row in rows) / len(rows)
                if all_hit else None),
            "censored_trials": sum(row["censored"] for row in rows),
            "moves_per_second": sum(row["moves"] for row in rows) / exposure,
        })
    return summary


def save_verified_discoveries(records, root=ROOT):
    """After measurements, append new verified pairs using the production writer."""
    counts = {}
    for row in records:
        length = row["L"]
        counts.setdefault(length, 0)
        for hit in row["hits"]:
            a, b = tuple(map(int, hit["A"])), tuple(map(int, hit["B"]))
            if len(a) != length or len(b) != length:
                raise ValueError("benchmark candidate length mismatch")
            counts[length] += int(append_verified_solution_if_new(length, a, b, root))
    return counts


def trial(executable, length, seed, seconds, threads, bank, directory, existing,
          quench=False, move_pool=False, quarter_pruning=False, first_new=False):
    """Measure one bounded search, including executable startup and verification."""
    environment = {k: v for k, v in os.environ.items() if not k.startswith("PQCP_")}
    environment.update(PQCP_SEED=str(seed), PQCP_THREADS=str(threads),
                       PQCP_CANDIDATES="2", PQCP_FKM_SEEDS=str(bank))
    if quench:
        environment["PQCP_QUENCH"] = "1"
    if move_pool:
        environment["PQCP_MOVE_POOL"] = "1"
    if quarter_pruning:
        environment["PQCP_QUARTER_PRUNING"] = "1"
    started = perf_counter()
    process = subprocess.Popen([str(executable), str(length)], cwd=directory,
                               env=environment, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True)
    lines = queue.Queue()

    def reader():
        for line in process.stdout:
            lines.put(line)
        lines.put(None)

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    stopped = False
    hits = []
    seen = set(existing)
    stats = None
    try:
        while True:
            elapsed = perf_counter() - started
            if elapsed >= seconds and not stopped:
                if process.poll() is None:
                    process.send_signal(signal.SIGINT)
                stopped = True
            try:
                line = lines.get(timeout=0.01)
            except queue.Empty:
                if elapsed > seconds + 30:
                    raise RuntimeError("search did not stop after SIGINT")
                continue
            if line is None:
                break
            match = re.search(r"restart=(\d+), moves=(\d+), swap evaluations=(\d+), valid results=(\d+)", line)
            if match:
                stats = dict(zip(("restarts", "moves", "evaluations", "valid"),
                                 map(int, match.groups())))
            if line.startswith("PQCP_CANDIDATE "):
                fields = dict(part.split("=", 1) for part in line.split()[1:])
                a, b = fields["a"], fields["b"]
                verification = verify_pqcp(a, b)
                if len(a) != length or not verification.is_valid:
                    raise RuntimeError("C candidate failed independent verifier")
                key = pair_key(a, b)
                if key not in seen:
                    seen.add(key)
                    hits.append({"elapsed": perf_counter() - started, "A": a,
                                 "B": b, "profile": verification.profile})
                    if first_new and not stopped:
                        if process.poll() is None:
                            process.send_signal(signal.SIGINT)
                        stopped = True
        process.wait(timeout=5)
        if process.returncode != 0 or stats is None:
            raise RuntimeError("search exited without valid final statistics")
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
            process.wait(timeout=30)
        thread.join(timeout=5)
        process.stdout.close()
    elapsed = perf_counter() - started
    return {"L": length, "seed": seed, "threads": threads, "budget": seconds,
            "elapsed": elapsed, **stats, "new_solutions": len(hits),
            "first_new_seconds": hits[0]["elapsed"] if hits else None,
            "censored": not hits, "hits": hits,
            "first_new_stopping": first_new,
            "moves_per_second": stats["moves"] / elapsed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", nargs="+", type=int, default=[44, 46])
    parser.add_argument("--seeds", nargs="+", type=int, default=[123, 456, 789])
    parser.add_argument("--seconds", type=float, default=3)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--first-new", action="store_true",
                        help="stop each trial at its first verified novel pair or the time cap")
    parser.add_argument("--save-solutions", action="store_true",
                        help="after all paired trials, verify/deduplicate/append discoveries to L.txt")
    parser.add_argument("--methods", nargs="+", default=["reference", "optimized"],
                        choices=["reference", "optimized", "quench", "pool", "pool_quench",
                                 "pool_pruned", "pool_pruned_quench"])
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/c_kernel_benchmark.json")
    args = parser.parse_args()
    if args.seconds <= 0 or args.threads <= 0:
        parser.error("seconds and threads must be positive")
    records = []
    source = ROOT / "csrc/pqcp_search.c"
    source_text = source.read_text(encoding="utf-8")
    metadata = {
        "source_sha256": hashlib.sha256(source_text.encode("utf-8")).hexdigest(),
        "source_text": source_text,
        "platform": platform.platform(), "machine": platform.machine(),
        "seconds_per_run": args.seconds, "threads": args.threads,
        "seeds": args.seeds, "lengths": args.lengths, "methods": args.methods,
        "candidate_count": 2, "fkm_seeds_per_content": 64,
        "first_new_stopping": args.first_new,
        "deduplication": "exact pair or A/B exchange, excluding preexisting L.txt",
        "timing": "process startup through candidate independent verification",
    }
    with tempfile.TemporaryDirectory(prefix="pqcp-kernel-benchmark-") as raw:
        directory = Path(raw)
        snapshot = directory / "pqcp_search_snapshot.c"
        snapshot.write_text(source_text, encoding="utf-8")
        executables = {}
        for method in ("reference", "optimized"):
            executable = directory / method
            flags = ["-DPQCP_REFERENCE_KERNEL"] if method == "reference" else []
            subprocess.run(["cc", "-O3", "-std=c11", "-pthread", *flags,
                            str(snapshot), "-lm", "-o",
                            str(executable)], check=True, capture_output=True)
            executables[method] = executable
        for length in args.lengths:
            existing_path = ROOT / "{}.txt".format(length)
            existing = set()
            if existing_path.exists():
                existing = {pair_key(a, b) for a, b in re.findall(
                    r"^a=([01]+)\nb=([01]+)$", existing_path.read_text(), re.M)}
            for index, seed in enumerate(args.seeds):
                bank = ensure_compressed_fkm_seed_bank(length, seed, directory)
                methods = args.methods[:]
                if index % 2:
                    methods.reverse()
                for method in methods:
                    row = trial(executables["reference" if method == "reference" else "optimized"],
                                length, seed, args.seconds, args.threads, bank,
                                directory, existing, quench=method.endswith("quench"),
                                move_pool=method.startswith("pool"),
                                quarter_pruning="pruned" in method,
                                first_new=args.first_new)
                    row["method"] = method
                    row["fkm_seed_bank_sha256"] = hashlib.sha256(bank.read_bytes()).hexdigest()
                    records.append(row)
                    atomic_write_json(args.output, {"metadata": metadata, "records": records})
                    print("L={} seed={} {} moves/s={:.0f} new={} first={}".format(
                        length, seed, method, row["moves_per_second"],
                        row["new_solutions"], row["first_new_seconds"]), flush=True)
    print("Saved {}".format(args.output))
    summary = summarize(records)
    atomic_write_json(args.output, {"metadata": metadata, "records": records,
                                   "summary": summary})
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.save_solutions:
        imported = save_verified_discoveries(records)
        print("New verified pairs appended to L.txt: {}".format(imported))


if __name__ == "__main__":
    main()
