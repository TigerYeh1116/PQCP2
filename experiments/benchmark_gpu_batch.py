"""Measure synchronized MPS blocks and exact observation at larger batch sizes.

This is a stage benchmark, not GPU utilization or end-to-end discovery speed.
The normal loss, observation frequency, exact verification and archive are used.
"""

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=[44, 68])
    parser.add_argument("--multipliers", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()
    if args.steps <= 0 or args.steps % 25 or any(m <= 0 for m in args.multipliers):
        parser.error("steps must be a positive multiple of 25; multipliers must be positive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    import torch
    from solver.torch_hybrid_runner import TorchHybridConfig, _new_continuous_search

    inventory = {length: (ROOT / '{}.txt'.format(length)).read_bytes()
                 if (ROOT / '{}.txt'.format(length)).exists() else b'' for length in args.lengths}
    rows = []
    for length in args.lengths:
        for multiplier in args.multipliers:
            batch = (4096 if length <= 58 else 2048) * multiplier
            root = output / 'L{}_batch{}'.format(length, batch)
            root.mkdir()
            if inventory[length]:
                (root / '{}.txt'.format(length)).write_bytes(inventory[length])
            started = perf_counter()
            search = _new_continuous_search(TorchHybridConfig(
                length, device='mps', root=root, continuous_batch_size=batch,
                continuous_mathematical_loss='lattice'))
            torch.mps.synchronize()
            initialization = perf_counter() - started
            observed = perf_counter()
            search.observe()
            torch.mps.synchronize()
            observation = perf_counter() - observed
            compute = 0.0
            peak_driver = 0
            for _ in range(args.steps // 25):
                tick = perf_counter()
                for _ in range(25):
                    search.step()
                torch.mps.synchronize()
                compute += perf_counter() - tick
                tick = perf_counter()
                search.observe()
                torch.mps.synchronize()
                observation += perf_counter() - tick
                peak_driver = max(peak_driver, torch.mps.driver_allocated_memory())
            wall = perf_counter() - started
            row = dict(L=length, batch=batch, steps=args.steps, initialization_seconds=initialization,
                       synchronized_compute_seconds=compute, observation_seconds=observation,
                       wall_seconds=wall, lane_steps_per_second=batch*args.steps/wall,
                       sampled_peak_driver_bytes=peak_driver, new_solutions=search.new_solutions,
                       inventory_sha256=hashlib.sha256(inventory[length]).hexdigest())
            rows.append(row)
            print(json.dumps(row), flush=True)
            (output / 'report.json').write_text(json.dumps(rows, indent=2))
            del search
            gc.collect()
            torch.mps.empty_cache()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
