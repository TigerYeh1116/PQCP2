"""Short, configurable real-CUDA integration measurement (never run by pytest)."""

import argparse
from dataclasses import asdict
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from solver.checkpoint import atomic_write_json
from solver.torch_search import TorchSearch, TorchSearchConfig
from solver.verifier import verify_pqcp


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--L", type=int, default=44)
    parser.add_argument("--seconds", type=float, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--full-only", action="store_true")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "results/torch_cuda_smoke.json")
    args = parser.parse_args()
    if not 0 < args.seconds < float("inf"):
        parser.error("--seconds must be finite and positive")
    config = TorchSearchConfig(args.L, seed=args.seed, batch_size=args.batch_size,
                               compression=not args.full_only)
    started = perf_counter()
    search = TorchSearch(config, args.root)
    search.observe()
    torch.cuda.synchronize()
    initialization_seconds = perf_counter() - started
    initial = search.best["score"]
    progression = [{"elapsed": 0, "epoch": 0, "best_score": initial}]

    def progress(state):
        row = {"elapsed": state.elapsed, "epoch": state.epoch, "best_score": state.best["score"]}
        progression.append(row)
        print(row, flush=True)

    result = search.run(seconds=args.seconds, progress=progress, progress_interval=5)
    torch.cuda.synchronize()
    verification = verify_pqcp(search.best["A"], search.best["B"])
    assert list(verification.profile) == search.best["profile"]
    payload = {
        "config": asdict(config), "torch_version": torch.__version__,
        "parameter_device": str(search.theta.device),
        "gradient_device": str(search.theta.grad.device) if search.theta.grad is not None else None,
        "cuda_allocated_bytes": torch.cuda.memory_allocated(),
        "initialization_seconds": initialization_seconds,
        "total_wall_seconds": perf_counter() - started,
        "initial_score": initial, "result": result, "best": search.best,
        "progression": progression, "best_independently_verified": verification.is_valid,
        "interpretation": "correctness/integration smoke; not proof of time-to-solution improvement",
    }
    atomic_write_json(args.output, payload)
    print("CUDA batch={} L={} epochs={} initial={} best={} new={} wall={:.3f}s".format(
        config.batch_size, config.L, search.epoch, initial, search.best["score"],
        search.new_solutions, payload["total_wall_seconds"]))
    print("Saved {}".format(args.output))


if __name__ == "__main__":
    main()
