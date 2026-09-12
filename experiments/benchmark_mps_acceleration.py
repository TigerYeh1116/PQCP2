"""Opt-in CUDA kernel and verified-discovery comparisons; never run by pytest.

All methods start with the same frozen L.txt inventory. Each method retains
its discoveries across seeds, so repeats (including A/B exchange) do not count
again. New pairs are independently checked and published to the project only
after each run; publishing cannot change another method's frozen inventory.
Wall time includes initialization, observations and final checkpoint writing.
Throughput is reported separately and is NOT a discovery speedup certificate.
"""

import argparse
from contextlib import redirect_stdout
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys
import tempfile
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from solver.checkpoint import atomic_write_json
from solver.search_runner import append_verified_solution_if_new
from solver.correlation import full_correlation_profile
from solver.objective import pqcp_objective
from solver.target_profiles import TargetContentProfile, pair_content
from solver.torch_search import BatchedPQCP, TorchSearch, TorchSearchConfig, require_device
from solver.verifier import verify_pqcp


def recorded_pairs(text):
    """Use the production writer's exact/A-B-swap equivalence, not rotations."""
    return {tuple(sorted((a, b))) for a, b in re.findall(r"^a=([01]+)\nb=([01]+)$", text, re.MULTILINE)}


def exhaustive_swap_draws(length, batch, steps, generator):
    """Encode each same-parity unordered position pair once, with random tie order.

    The polish module rejects equal signs. This consequently evaluates every
    legal one-swap neighbor, without changing the neighborhood or verifier.
    Bounds constrain both persistent random draws and active candidate tensors.
    """
    half = length // 2
    patterns = [(float(g + 0.5) / 4, float(i + 0.5) / half, float(j + 0.5) / half, 0.0)
                for g in range(4) for i in range(half) for j in range(i + 1, half)]
    count = len(patterns)
    if steps * batch * count * 4 > 64000000 or batch * count * length > 4000000:
        raise ValueError("exhaustive probe memory bound: reduce elites, clones or steps")
    order = torch.rand(steps, batch, count, generator=generator).argsort(-1)
    draws = torch.tensor(patterns, dtype=torch.float32)[order]
    draws[..., 3] = torch.rand(steps, batch, count, generator=generator)
    return draws


def summarize(rows):
    """Aggregate genuine new pairs / actual exposure; zero baseline is undefined."""
    out = []
    for length, method in sorted({(r["L"], r["method"]) for r in rows}):
        group = [r for r in rows if (r["L"], r["method"]) == (length, method)]
        wall = sum(r["wall_seconds"] for r in group)
        count = sum(r["new_solutions"] for r in group)
        out.append({"L": length, "method": method, "runs": len(group),
                    "new_solutions": count, "wall_seconds": wall,
                    "new_per_second": count / wall,
                    "median_best_score": statistics.median(r["best_score"] for r in group)})
    for row in out:
        baseline = next((r for r in out if r["L"] == row["L"] and r["method"] == "direct"), None)
        row["rate_ratio_to_direct"] = (row["new_per_second"] / baseline["new_per_second"]
                                       if baseline and baseline["new_per_second"] > 0 else None)
    return out


def throughput(length, batch, steps, repeats):
    """Interleave synchronized optimizer measurements after per-kernel warmup."""
    samples = {key: [] for key in ("direct", "dft")}
    for repeat in range(repeats):
        for kernel in (("direct", "dft") if repeat % 2 == 0 else ("dft", "direct")):
            with tempfile.TemporaryDirectory(prefix="pqcp-mps-kernel-") as scratch:
                search = TorchSearch(TorchSearchConfig(length, batch_size=batch, continuous_kernel=kernel), Path(scratch))
                for _ in range(10):
                    search.step()
                torch.cuda.synchronize()
                start = perf_counter()
                for _ in range(steps):
                    search.step()
                torch.cuda.synchronize()
                samples[kernel].append(perf_counter() - start)
                del search
                torch.cuda.empty_cache()
    times = {key: statistics.median(value) for key, value in samples.items()}
    return {"L": length, "batch": batch, "steps": steps, "samples_seconds": samples,
            "median_seconds": times, "dft_throughput_ratio": times["direct"] / times["dft"],
            "pair_updates_per_second": {key: batch * steps / value for key, value in times.items()}}


def discovery(args):
    """Equal-budget seeded arms, isolated dedup inventories, auditable artifacts."""
    rows = []
    snapshot = {}
    for length in args.lengths:
        path = ROOT / "{}.txt".format(length)
        snapshot[length] = path.read_text(encoding="utf-8") if path.exists() else ""
        for kernel in ("direct", "dft"):
            directory = args.output / "L{}".format(length) / kernel
            directory.mkdir(parents=True)
            (directory / "{}.txt".format(length)).write_text(snapshot[length], encoding="utf-8")
    atomic_write_json(args.output / "inventory.json", {
        str(L): {"sha256": hashlib.sha256(text.encode()).hexdigest(), "pairs": len(recorded_pairs(text))}
        for L, text in snapshot.items()})
    for length in args.lengths:
        for seed_index, seed in enumerate(args.seeds):
            for kernel in (("direct", "dft") if seed_index % 2 == 0 else ("dft", "direct")):
                directory = args.output / "L{}".format(length) / kernel
                path = directory / "{}.txt".format(length)
                before = recorded_pairs(path.read_text(encoding="utf-8"))
                config = TorchSearchConfig(length, seed=seed, batch_size=args.batch_size, continuous_kernel=kernel)
                with (directory / "seed{}.log".format(seed)).open("w", encoding="utf-8") as log, redirect_stdout(log):
                    torch.cuda.synchronize()
                    started = perf_counter()
                    search = TorchSearch(config, directory)
                    torch.cuda.synchronize()
                    initialization = perf_counter() - started
                    result = search.run(seconds=max(0, args.seconds - initialization))
                    torch.cuda.synchronize()
                    wall = perf_counter() - started
                added = recorded_pairs(path.read_text(encoding="utf-8")) - before
                assert len(added) == result["new_solutions"]
                for a, b in sorted(added):
                    bits_a, bits_b = tuple(map(int, a)), tuple(map(int, b))
                    if not verify_pqcp(bits_a, bits_b).is_valid:
                        raise RuntimeError("benchmark result failed independent verification")
                    append_verified_solution_if_new(length, bits_a, bits_b, ROOT)
                row = {"L": length, "method": kernel, "seed": seed,
                       "wall_seconds": wall, "initialization_seconds": initialization,
                       "budget_seconds": args.seconds, "new_solutions": len(added),
                       "epoch": search.epoch, "best_score": result["best_score"],
                       "config": asdict(config), "pairs": [list(p) for p in sorted(added)]}
                rows.append(row)
                with (args.output / "runs.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row) + "\n")
                atomic_write_json(args.output / "summary.json", summarize(rows))
                print("L={} {} seed={} wall={:.2f}s new={} best={}".format(
                    length, kernel, seed, wall, len(added), result["best_score"]), flush=True)
                del search
                torch.cuda.empty_cache()
    return summarize(rows)


def polish_probe(args):
    """Bounded CUDA polish of recorded elites, NOT an end-to-end rate comparison."""
    from solver.torch_polish import closest_content_target, polish_batch

    records = {}
    sources = {}
    for path in args.checkpoints:
        raw = path.read_bytes()
        data = json.loads(raw)
        if data.get("format") == "pqcp-torch-search":
            incoming = data["archive"]
        elif "candidates" in data and "L" in data:
            if not args.retarget:
                raise ValueError("polish-result inputs require --retarget; no original target is assumed")
            incoming = [{"L": data["L"], **p} for p in data["candidates"]]
        else:
            raise ValueError("probe input must be a PyTorch checkpoint or prior polish result")
        sources[str(path)] = hashlib.sha256(raw).hexdigest()
        for payload in incoming:
            key = tuple(sorted((tuple(payload["A"]), tuple(payload["B"]))))
            records.setdefault(key, payload)
    elites = sorted(records.values(), key=lambda p: p["score"])[:args.elites]
    if not elites or len({p["L"] for p in elites}) != 1:
        raise ValueError("polish probe requires nonempty elites of the same length")
    selected = elites * args.clones
    length = selected[0]["L"]
    profiles = []
    for payload in elites:
        exact = full_correlation_profile(payload["A"], payload["B"])
        if exact != payload["profile"] or pqcp_objective(exact) != payload["score"]:
            raise ValueError("saved elite fails exact correlation/score consistency")
        profiles.append(closest_content_target(payload["A"], payload["B"], compression=not args.full_only)
                        if args.retarget else TargetContentProfile(length, payload["k"], payload["sign"],
                                                                  *pair_content(payload["A"], payload["B"])))
    profiles *= args.clones
    model = BatchedPQCP(profiles, require_device("cuda"))
    signs = 1 - 2 * torch.tensor([[p["A"], p["B"]] for p in selected], dtype=torch.float32, device="cuda")
    generator = torch.Generator().manual_seed(args.seeds[0])
    draws = (exhaustive_swap_draws(length, len(selected), args.steps, generator)
             if args.exhaustive else torch.rand(args.steps, len(selected), args.candidates, 4, generator=generator)).to("cuda")
    torch.cuda.synchronize()
    started = perf_counter()
    result = polish_batch(model, signs, draws, proposal_policy=args.proposal_policy,
                          kick_interval=args.kick_interval, compression=not args.full_only,
                          initial_temperature=1e-6 if args.exhaustive else 24.6,
                          final_temperature=1e-6 if args.exhaustive else 0.6)
    torch.cuda.synchronize()
    polish_seconds = perf_counter() - started
    bits = ((1 - result.signs) / 2).to(torch.int32).cpu().tolist()
    gpu_profiles = result.profile.cpu().tolist()
    scores = result.scores.cpu().tolist()
    candidates, unique, new = [], set(), 0
    for lane, ((a, b), gpu_profile, score) in enumerate(zip(bits, gpu_profiles, scores)):
        exact = full_correlation_profile(a, b)
        if exact != gpu_profile or pqcp_objective(exact) != score:
            raise RuntimeError("polish result differs from full integer recomputation")
        verification = verify_pqcp(a, b)
        if score == 0 and not verification.is_valid:
            raise RuntimeError("polish zero failed independent verifier")
        if verification.is_valid:
            unique.add(tuple(sorted((tuple(a), tuple(b)))))
            new += append_verified_solution_if_new(length, tuple(a), tuple(b), ROOT)
        candidates.append({"A": a, "B": b, "profile": exact, "score": int(score),
                           "initial_score": selected[lane]["score"], "verified": verification.is_valid,
                           "k": profiles[lane].k, "sign": profiles[lane].eta})
    summary = {"L": length, "lanes": len(selected), "steps": args.steps,
               "candidates_per_move": draws.shape[2], "proposal_policy": args.proposal_policy,
               "compression": not args.full_only, "kick_interval": args.kick_interval,
               "retarget": args.retarget,
               "exhaustive_quench": args.exhaustive,
               "kicks": result.kicks, "forced_swaps": result.forced_swaps,
               "polish_seconds": polish_seconds, "legal_swaps": result.legal_swaps,
               "accepted_swaps": result.accepted_swaps, "proposals": result.proposals,
               "initial_min_score": min(p["score"] for p in selected),
               "final_min_score": min(scores), "verified_unique_pairs": len(unique),
               "new_to_project": new, "sources": sources, "candidates": candidates,
               "interpretation": "polish feasibility probe; excludes cost of producing elites; NOT discovery speedup evidence"}
    atomic_write_json(args.output / "polish.json", summary)
    return {k: v for k, v in summary.items() if k != "candidates"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("throughput", "discovery", "polish"), default="throughput")
    parser.add_argument("--lengths", type=int, nargs="+", default=[44, 46])
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--seeds", type=int, nargs="+", default=[123, 456, 789])
    parser.add_argument("--checkpoints", type=Path, nargs="+", help="recorded elite sources for --mode polish")
    parser.add_argument("--elites", type=int, default=64)
    parser.add_argument("--clones", type=int, default=8)
    parser.add_argument("--candidates", type=int, default=1)
    parser.add_argument("--proposal-policy", choices=("positions", "opposite"), default="positions")
    parser.add_argument("--kick-interval", type=int, default=0, help="polish-only: 0 disables C-style stagnation kicks")
    parser.add_argument("--full-only", action="store_true", help="polish-only: omit folded energy for ablation")
    parser.add_argument("--retarget", action="store_true", help="polish-only: choose closest content-compatible target without transforming bits")
    parser.add_argument("--exhaustive", action="store_true", help="polish-only: enumerate every same-parity swap with a near-zero-temperature quench")
    parser.add_argument("--output", type=Path, required=True, help="a NEW artifact directory")
    args = parser.parse_args()
    if not 0 < args.seconds < float("inf") or min(args.steps, args.repeats, args.batch_size, args.elites, args.clones, args.candidates) <= 0:
        parser.error("budgets/counts must be finite and positive")
    if len(set(args.lengths)) != len(args.lengths) or len(set(args.seeds)) != len(args.seeds):
        parser.error("duplicate lengths or seeds would bias the comparison")
    if args.mode == "polish" and not args.checkpoints:
        parser.error("--mode polish requires --checkpoints")
    if args.kick_interval < 0 or (args.mode != "polish" and (args.kick_interval or args.full_only or args.retarget or args.exhaustive)):
        parser.error("kick/full-only/retarget options apply only to polish; kick interval must be nonnegative")
    if args.exhaustive and args.proposal_policy != "positions":
        parser.error("--exhaustive requires --proposal-policy positions")
    require_device("cuda")
    args.output.mkdir(parents=True, exist_ok=False)
    sources = ["solver/torch_search.py", "solver/torch_polish.py", "solver/correlation.py", "solver/objective.py",
               "solver/verifier.py", "experiments/benchmark_mps_acceleration.py"]
    # Preserve exact measured source, not just a hash of a subsequently edited file.
    for source in sources:
        destination = args.output / "source" / source
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / source).read_bytes())
    atomic_write_json(args.output / "manifest.json", {
        "arguments": {k: ([str(p) for p in v] if k == "checkpoints" and v else str(v) if isinstance(v, Path) else v)
                      for k, v in vars(args).items()},
        "torch": torch.__version__, "device": "cuda",
        "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources}})
    if args.mode == "throughput":
        rows = []
        for length in args.lengths:
            row = throughput(length, args.batch_size, args.steps, args.repeats)
            rows.append(row)
            atomic_write_json(args.output / "throughput.json", rows)
            print(json.dumps(row), flush=True)
    elif args.mode == "discovery":
        print(json.dumps(discovery(args), indent=2))
    else:
        print(json.dumps(polish_probe(args), indent=2))


if __name__ == "__main__":
    main()
