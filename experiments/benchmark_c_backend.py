#!/usr/bin/env python3
"""Measure compiled compressed-FKM throughput without running a long search.

The benchmark reports the counters used by the reference C project.  An
optional ``--reference-source`` compiles that source with the same flags and
runs it under the same length, seed, thread, candidate, and wall-clock setup.
"""

import argparse
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
from time import perf_counter, sleep

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from solver.c_backend import (
    CSearchConfig,
    ensure_compressed_fkm_seed_bank,
    run_c_search,
)


STATS = re.compile(
    r"搜尋統計：restart=(\d+), moves=(\d+), "
    r"swap evaluations=(\d+), valid results=(\d+)"
)


def _reference_run(source: Path, length: int, seed: int, seconds: float,
                   threads: int, seed_bank: Path, directory: Path):
    executable = directory / "reference_search"
    subprocess.run(
        ("cc", "-O3", "-std=c11", "-pthread", str(source), "-lm", "-o", str(executable)),
        check=True,
    )
    environment = {
        **os.environ,
        "PQCP_SEED": str(seed),
        "PQCP_THREADS": str(threads),
        "PQCP_CANDIDATES": "2",
        "PQCP_FKM_SEEDS": str(seed_bank),
        "PQCP_USE_FKM_SEEDS": "1",
    }
    started = perf_counter()
    process = subprocess.Popen(
        (str(executable), str(length)), cwd=directory, env=environment,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    sleep(seconds)
    process.send_signal(signal.SIGINT)
    output = process.communicate(timeout=30)[0]
    elapsed = perf_counter() - started
    matches = STATS.findall(output)
    if not matches:
        raise RuntimeError("reference source returned no final counters")
    restarts, moves, swaps, valid = map(int, matches[-1])
    return {
        "elapsed": elapsed, "restarts": restarts, "moves": moves,
        "swaps": swaps, "valid": valid,
        "moves_per_second": moves / elapsed,
        "swaps_per_second": swaps / elapsed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--L", type=int, default=44)
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--reference-source", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="pqcp-c-throughput-") as raw:
        directory = Path(raw)
        current = run_c_search(CSearchConfig(
            L=args.L, seed=args.seed, seconds=args.seconds,
            threads=args.threads, root=directory, echo=False,
            seeds_per_content=16,
        ))
        print("C compressed-FKM throughput")
        print("L={} threads={} seconds={:.3f}".format(args.L, args.threads, current.elapsed))
        print("restart={} moves={} swap evaluations={}".format(
            current.restarts, current.moves, current.swap_evaluations
        ))
        print("moves/second={:.0f}".format(current.moves_per_second))
        print("swap evaluations/second={:.0f}".format(
            current.swap_evaluations_per_second
        ))
        if args.reference_source is not None:
            bank = ensure_compressed_fkm_seed_bank(
                args.L, args.seed, directory, seeds_per_content=16
            ).resolve()
            reference = _reference_run(
                args.reference_source.resolve(), args.L, args.seed,
                args.seconds, args.threads, bank, directory,
            )
            print("Reference C throughput")
            print("restart={} moves={} swap evaluations={}".format(
                reference["restarts"], reference["moves"], reference["swaps"]
            ))
            print("moves/second={:.0f}".format(reference["moves_per_second"]))
            print("swap evaluations/second={:.0f}".format(reference["swaps_per_second"]))
            print("current/reference moves ratio={:.3f}".format(
                current.moves_per_second / reference["moves_per_second"]
            ))
            print("current/reference swap ratio={:.3f}".format(
                current.swap_evaluations_per_second / reference["swaps_per_second"]
            ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
