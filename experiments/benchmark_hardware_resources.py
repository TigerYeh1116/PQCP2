"""Compare resource settings with the unchanged production search in isolated roots.

Inventory is snapshotted once, for deduplication only. No archived solution is
used as an initialization seed. Timeouts/no discoveries remain in the report.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from time import perf_counter


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", type=int, nargs="+", default=[44, 68])
    parser.add_argument("--seeds", type=int, nargs="+", default=[123])
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", choices=("legacy", "previous"), default="legacy")
    parser.add_argument("--candidate-batch-short", type=int)
    parser.add_argument("--candidate-batch-long", type=int)
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("seconds must be positive")
    if any(value is not None and value <= 0 for value in
           (args.candidate_batch_short, args.candidate_batch_long)):
        parser.error("candidate batches must be positive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    inventory = {}
    for length in args.lengths:
        source = ROOT / '{}.txt'.format(length)
        inventory[length] = source.read_bytes() if source.exists() else b''
    rows = []
    for length in args.lengths:
        for index, seed in enumerate(args.seeds):
            # Alternate order across seeds to reduce warm-up/order bias.
            variants = [args.reference, 'auto'] if index % 2 == 0 else ['auto', args.reference]
            for variant in variants:
                destination = output / '{}_L{}_seed{}'.format(variant, length, seed)
                destination.mkdir()
                if inventory[length]:
                    (destination / '{}.txt'.format(length)).write_bytes(inventory[length])
                command = [sys.executable, str(ROOT / 'main.py'), '--L', str(length),
                           '--seed', str(seed), '--seconds', str(args.seconds), '--torch-device', 'mps']
                if variant == 'legacy':
                    command += ['--torch-batch-size', str(2048 if length <= 58 else 1024),
                                '--hybrid-c-threads', '4']
                elif variant == 'previous':
                    command += ['--torch-batch-size', str(4096 if length <= 58 else 2048),
                                '--hybrid-c-threads', '6']
                else:
                    batch = args.candidate_batch_short if length <= 58 else args.candidate_batch_long
                    if batch is not None:
                        command += ['--torch-batch-size', str(batch)]
                started = perf_counter()
                with (destination / 'console.log').open('w') as log:
                    result = subprocess.run(command, cwd=destination, stdout=log, stderr=subprocess.STDOUT,
                                            timeout=args.seconds + 180)
                metrics = [json.loads(path.read_text()) for path in
                           destination.glob('results/search_metrics/*.json')]
                statistics = re.findall(
                    r'restart=(\d+), moves=(\d+), swap evaluations=(\d+), valid results=(\d+)',
                    (destination / 'console.log').read_text())
                row = dict(variant=variant, L=length, seed=seed, budget=args.seconds,
                           wall_seconds=perf_counter()-started, returncode=result.returncode,
                           inventory_sha256=hashlib.sha256(inventory[length]).hexdigest(),
                           command=command, metrics=metrics,
                           final_statistics=dict(zip(('restarts', 'moves', 'swap_evaluations', 'new_solutions'),
                                                     map(int, statistics[-1]))) if statistics else None)
                rows.append(row)
                (output / 'report.json').write_text(json.dumps(rows, indent=2, ensure_ascii=False))
                print('{} L={} seed={} exit={} wall={:.2f}s'.format(
                    variant, length, seed, result.returncode, row['wall_seconds']), flush=True)
                if result.returncode:
                    return result.returncode
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
