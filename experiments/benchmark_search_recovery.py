"""Measure the actual no-argument main.py entry, including startup and writing.

Runs use isolated copies of the current result inventory. The entry receives
L, seed and 'no Z3' on stdin, exactly like VS Code Run. At the first verified
novel pair (or deadline) it receives Ctrl+C/SIGINT. Source/result artifacts are
retained; verified discoveries are appended to the real project afterwards.
No search implementation is replaced or mocked by this benchmark.
"""

import argparse
import hashlib
import json
from pathlib import Path
import queue
import re
import signal
import subprocess
import sys
import threading
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from solver.checkpoint import atomic_write_json
from solver.search_runner import append_verified_solution_if_new
from solver.verifier import verify_pqcp


def pairs(text):
    """Mirror the official exact/A-B-swap deduplication convention."""
    return {tuple(sorted(p)) for p in re.findall(r"^a=([01]+)\nb=([01]+)$", text, re.M)}


def trial(length, seed, seconds, directory):
    directory.mkdir(parents=True, exist_ok=False)
    official = ROOT / "{}.txt".format(length)
    snapshot = official.read_text(encoding="utf-8") if official.exists() else ""
    existing = pairs(snapshot)
    result_file = directory / official.name
    result_file.write_text(snapshot, encoding="utf-8")
    events = queue.Queue()
    started = perf_counter()
    process = subprocess.Popen([sys.executable, str(ROOT / "main.py")], cwd=directory,
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1)
    process.stdin.write("{}\n{}\n\n".format(length, seed))
    process.stdin.close()

    def read_output():
        for line in process.stdout:
            events.put(line)
        events.put(None)

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()
    first = None
    interrupted_at = None
    with (directory / "terminal.log").open("w", encoding="utf-8") as log:
        try:
            while True:
                now = perf_counter()
                if now - started >= seconds and interrupted_at is None and process.poll() is None:
                    process.send_signal(signal.SIGINT)
                    interrupted_at = now
                if interrupted_at is not None and now - interrupted_at > 40:
                    raise RuntimeError("main.py did not shut down after Ctrl+C")
                try:
                    line = events.get(timeout=0.02)
                except queue.Empty:
                    continue
                if line is None:
                    break
                log.write(line)
                log.flush()
                if "已追加寫入" in line and first is None:
                    added = pairs(result_file.read_text(encoding="utf-8")) - existing
                    if added:
                        for a, b in added:
                            if not verify_pqcp(a, b).is_valid:
                                raise RuntimeError("main wrote an invalid PQCP")
                        first = perf_counter() - started
                        if interrupted_at is None and process.poll() is None:
                            process.send_signal(signal.SIGINT)
                            interrupted_at = perf_counter()
            process.wait(timeout=5)
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=40)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    process.wait(timeout=5)
            reader.join(timeout=5)
            process.stdout.close()
    elapsed = perf_counter() - started
    if process.returncode != 0:
        raise RuntimeError("main.py exited with status {} (see terminal.log)".format(process.returncode))
    added = pairs(result_file.read_text(encoding="utf-8")) - existing
    hits = []
    for a, b in sorted(added):
        verification = verify_pqcp(a, b)
        if not verification.is_valid:
            raise RuntimeError("saved candidate failed independent verification")
        hits.append({"A": a, "B": b, "profile": list(verification.profile),
                     "new_to_project": append_verified_solution_if_new(length, tuple(map(int, a)), tuple(map(int, b)), ROOT)})
    row = {"L": length, "seed": seed, "budget_seconds": seconds,
           "first_new_seconds": first, "elapsed_including_shutdown": elapsed,
           "new_pairs": len(hits), "hits": hits, "censored": first is None,
           "inventory_sha256": hashlib.sha256(snapshot.encode()).hexdigest(),
           "exit_code": process.returncode, "stdin": "L,seed,no Z3",
           "entry": "main.py without arguments", "directory": str(directory)}
    atomic_write_json(directory / "result.json", row)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--L", type=int, default=44)
    parser.add_argument("--seeds", type=int, nargs="+", default=[90210])
    parser.add_argument("--seconds", type=float, default=300)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 < args.seconds < float("inf") or len(set(args.seeds)) != len(args.seeds):
        parser.error("a finite positive budget and distinct seeds are required")
    args.output.mkdir(parents=True, exist_ok=False)
    sources = ["main.py", "solver/c_backend.py", "csrc/pqcp_search.c", "solver/verifier.py"]
    atomic_write_json(args.output / "sources.json", {
        name: {"sha256": hashlib.sha256((ROOT / name).read_bytes()).hexdigest(),
               "text": (ROOT / name).read_text(encoding="utf-8")} for name in sources})
    rows = []
    for seed in args.seeds:
        row = trial(args.L, seed, args.seconds, args.output.resolve() / "seed{}".format(seed))
        rows.append(row)
        atomic_write_json(args.output / "summary.json", rows)
        print("L={} seed={} first_new={}s new={} exit={}".format(
            args.L, seed, row["first_new_seconds"], row["new_pairs"], row["exit_code"]), flush=True)


if __name__ == "__main__":
    main()
