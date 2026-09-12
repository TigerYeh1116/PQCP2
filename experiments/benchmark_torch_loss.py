"""Isolated, paired wall-clock loss ablation; never uses known PQCP seeds.

Only the relaxed loss differs. FKM initialization, Adam, temperature, rebirth,
exact archive ranking and optional GPU polish settings are identical. Each
method receives the same formation budget; synchronized elapsed includes
observations. Initialization and final observation/polish are reported apart.
Outputs (including any solutions) stay inside a NEW experiment directory.
"""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from solver.checkpoint import atomic_write_json
from solver.torch_hybrid_runner import (
    TorchHybridConfig, _continuous_seed_records,
    _polish_continuous_records,
)
from solver.torch_search import TorchSearch, TorchSearchConfig, require_device


def synchronize(device: str) -> None:
    """Include queued accelerator work in measured runtime."""
    if device == "cuda":
        torch.cuda.synchronize()


def run_case(config: TorchHybridConfig, mode: str, seconds: float, *,
             polish: bool = True, mathematical_loss: str = "none") -> dict:
    """Evaluate one fresh FKM run, retaining exact best candidates in root."""
    if not 0 < seconds < float("inf"):
        raise ValueError("seconds must be finite and positive")
    started = perf_counter()
    # Construct the common configuration without starting a second trajectory.
    cfg = TorchSearchConfig(
        config.L, seed=config.seed, device=config.device,
        batch_size=config.continuous_batch_size,
        steps_per_restart=config.continuous_steps_per_restart,
        observation_interval=config.continuous_observation_interval,
        stagnation_steps=max(config.continuous_observation_interval * 8,
                             config.continuous_steps_per_restart // 2),
        archive_size=config.continuous_archive_size,
        elite_count=min(16, config.continuous_archive_size),
        continuous_kernel="dft", loss_mode=mode,
        mathematical_loss=mathematical_loss,
    )
    search = TorchSearch(cfg, config.root)
    fingerprint = hashlib.sha256(json.dumps(
        search.theta.detach().cpu().tolist(), separators=(",", ":")
    ).encode()).hexdigest()
    search.observe()
    synchronize(config.device)
    initialization = perf_counter() - started
    progression = [{"elapsed": 0.0, "epoch": 0, "score": search.best["score"]}]
    started = perf_counter()

    def observe(*, update_stagnation: bool = True) -> None:
        previous = search.best["score"]
        search.elapsed = perf_counter() - started
        search.observe(update_stagnation=update_stagnation)
        if search.best["score"] < previous:
            progression.append({"elapsed": perf_counter() - started,
                                "epoch": search.epoch, "score": search.best["score"]})

    while perf_counter() - started < seconds:
        stale = all(search.epoch - last >= cfg.stagnation_steps
                    for last in search.last_improved)
        if search.round_step >= cfg.steps_per_restart or stale:
            search.generation += 1
            search._initialize_batch()
            observe()
        search.step()
        if search.epoch % cfg.observation_interval == 0:
            observe()
        # Bound queued GPU work so the wall-clock budget is real, not submission time.
        synchronize(config.device)
    formation_elapsed = perf_counter() - started
    observed = perf_counter()
    if search.last_observed_epoch != search.epoch:
        observe(update_stagnation=False)
    synchronize(config.device)
    final_observation = perf_counter() - observed
    records = _continuous_seed_records(search, config.target_seed_count)
    target_best = min(record["target_energy"] for record in search.archive.values())
    polished, verified, new = records, 0, 0
    polish_started = perf_counter()
    if polish:
        polished, verified, new = _polish_continuous_records(config, search, records, 0)
    synchronize(config.device)
    polish_elapsed = perf_counter() - polish_started
    scores = [search.best["score"]] + [record["score"] for record in polished]
    if verified:
        scores.append(0)
    report = {
        "L": config.L, "seed": config.seed, "mode": mode,
        "mathematical_loss": mathematical_loss, "device": config.device,
        "budget": seconds, "batch": cfg.batch_size, "initial_fingerprint": fingerprint,
        "initialization_elapsed": initialization, "formation_elapsed": formation_elapsed,
        "final_observation_elapsed": final_observation, "polish_elapsed": polish_elapsed,
        "epochs": search.epoch, "generations": search.generation,
        "projected_best_score": search.best["score"], "target_best": target_best,
        "post_polish_best_score": min(scores),
        "polish_input_lanes": min(config.polish_elites, len(records)) if polish else 0,
        "polish_steps": config.polish_steps if polish else 0,
        "polish_candidates": config.polish_candidates if polish else 0,
        "verified_hits": search.verified_hits + verified,
        "new_solutions": search.new_solutions + new,
        "progression": progression, "best": search.best,
        "best_retained_after_polish": min(
            [search.best] + polished, key=lambda item: item["score"]
        ),
        "verified_solution_file": str(Path(config.root) / "{}.txt".format(config.L))
        if search.new_solutions + new else None,
    }
    atomic_write_json(Path(config.root) / "report.json", report)
    return report


def summarize(records: list) -> list:
    """Summarize real discrete scores, never compare differently scaled losses."""
    result = []
    for length, mode in sorted({(r["L"], r["mode"]) for r in records}):
        group = [r for r in records if r["L"] == length and r["mode"] == mode]
        scores = [r["projected_best_score"] for r in group]
        result.append({
            "L": length, "mode": mode, "runs": len(group),
            "projected_min": min(scores), "projected_median": statistics.median(scores),
            "projected_mean": statistics.mean(scores),
            "polished_median": statistics.median(r["post_polish_best_score"] for r in group),
            "target_median": statistics.median(r["target_best"] for r in group),
            "epochs_median": statistics.median(r["epochs"] for r in group),
            "new_solutions": sum(r["new_solutions"] for r in group),
        })
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", type=int, nargs="+", default=[44, 46, 68])
    parser.add_argument("--seeds", type=int, nargs="+", default=[123, 456, 789])
    parser.add_argument("--modes", nargs="+", choices=["legacy", "balanced", "projected"],
                        default=["legacy", "balanced", "projected"])
    parser.add_argument("--seconds", type=float, default=15)
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--no-polish", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if not 0 < args.seconds < float("inf"):
        parser.error("seconds must be finite and positive")
    require_device(args.device)
    args.output.mkdir(parents=True, exist_ok=False)
    reports = []
    # Warm common kernels/Adam outside measured runs; never read a result file.
    for mode in args.modes:
        warm = TorchSearch(TorchSearchConfig(
            4, batch_size=4, device=args.device, continuous_kernel="dft", loss_mode=mode),
            args.output / "warmup")
        for _ in range(3):
            warm.step()
        synchronize(args.device)
        del warm
    for length in args.lengths:
        for index, seed in enumerate(args.seeds):
            order = args.modes[index % len(args.modes):] + args.modes[:index % len(args.modes)]
            fingerprints = set()
            for mode in order:
                root = args.output / "L{}_seed{}_{}".format(length, seed, mode)
                config = TorchHybridConfig(length, seed=seed, device=args.device,
                                           continuous_batch_size=args.batch, root=root)
                report = run_case(config, mode, args.seconds, polish=not args.no_polish)
                fingerprints.add(report["initial_fingerprint"])
                reports.append(report)
                with (args.output / "runs.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(report) + "\n")
                print(json.dumps({k: report[k] for k in (
                    "L", "seed", "mode", "epochs", "projected_best_score", "target_best",
                    "post_polish_best_score", "new_solutions")}), flush=True)
            if len(fingerprints) != 1:
                raise RuntimeError("paired loss variants did not use identical initialization")
    summary = summarize(reports)
    atomic_write_json(args.output / "summary.json", {"summary": summary})
    for row in summary:
        print(json.dumps(row), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
