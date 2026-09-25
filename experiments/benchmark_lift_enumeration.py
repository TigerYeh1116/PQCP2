"""Measure distinct full PQCP classes per fixed compressed job, not just SATs.

Fresh jobs come from unseeded ternary search. Archived jobs are sampled from
checksummed mathematical work descriptions, not L.txt sequences. Selecting SAT
jobs is intentionally conditional: it measures the opportunity AFTER a first
solution, not time to discover that job. Saved A/B/hints are never solver seeds.
An optional L.txt snapshot is read only for reporting existing equivalence keys.
All candidates are saved to a separate report; official results/checkpoints
are never modified. UNKNOWN and budget limits are not exhaustion proofs.
"""

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import random
import re
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from solver.checkpoint import atomic_write_json
from solver.compressed_completion import CompressedLiftJob
from solver.lift_enumeration import enumerate_lifts
from solver.solution_equivalence import pqcp_equivalence_key
from solver.staged_completion import CompletionWorkStore, SlicePolicy, job_from_spec
from solver.verifier import verify_pqcp


def inventory(path, length):
    """Immutable read-only reference set, never a source of search candidates."""
    if path is None:
        return set(), None
    data = path.read_bytes()
    pairs = re.findall(r"^a=([01]+)\s*\n[bB]=([01]+)\s*$", data.decode("utf-8"), re.MULTILINE)
    keys = set()
    for a, b in pairs:
        aa, bb = tuple(map(int, a)), tuple(map(int, b))
        if len(aa) != length or not verify_pqcp(aa, bb).is_valid:
            raise ValueError("invalid baseline inventory pair")
        keys.add(pqcp_equivalence_key(aa, bb))
    if data.strip() and not pairs:
        raise ValueError("inventory contains no recognized a=/b= pairs")
    return keys, hashlib.sha256(data).hexdigest()


def archived_jobs(root, length, seeds, per_seed, phase, sample_seed):
    """Stratified deterministic sample; duplicate exact compressed orbits removed."""
    seen = set()
    output = []
    rng = random.Random(sample_seed)
    for seed in seeds:
        store = CompletionWorkStore(root, length, seed, SlicePolicy())
        paths = sorted(store.directory.glob("*.json"))
        rng.shuffle(paths)
        count = 0
        for path in paths:
            record = store.read(path.stem)  # verifies checksum/identity without writes
            if (record["kind"] != "lift" or record["spec"]["factor"] != 2
                    or phase != "all" and record["phase"] != phase):
                continue
            job = job_from_spec(record["spec"])
            if job.canonical_key in seen:
                continue
            seen.add(job.canonical_key)
            # Source words are provenance only; enumeration never uses them.
            output.append((job, {"source": str(path), "source_seed": seed,
                                 "source_phase": record["phase"], "sampling": "archived_"+phase}))
            count += 1
            if count >= per_seed:
                break
    return output


def summarize(rows, baseline):
    first = {tuple(r["models"][0]["equivalence_key"]) for r in rows if r["models"]}
    all_keys = {tuple(m["equivalence_key"]) for r in rows for m in r["models"]}
    return dict(jobs=len(rows), raw_models=sum(len(r["models"]) for r in rows),
                jobs_with_multiple_classes=sum(len(r["equivalence_keys"]) > 1 for r in rows),
                classes_per_job=[len(r["equivalence_keys"]) for r in rows],
                first_model_global_classes=len(first), all_model_global_classes=len(all_keys),
                additional_global_classes=len(all_keys-first),
                first_model_new_to_inventory=len(first-baseline),
                all_models_new_to_inventory=len(all_keys-baseline),
                extra_new_to_inventory=len(all_keys-first-baseline),
                elapsed=sum(r["elapsed"] for r in rows),
                first_model_time=sum(r["models"][0]["elapsed"] if r["models"] else r["elapsed"] for r in rows),
                terminals={status: sum(r["terminal"] == status for r in rows)
                           for status in ("EXHAUSTED", "UNKNOWN", "TIME_LIMIT", "MODEL_LIMIT")})


def compare_staged(selected, length, baseline_path, fingerprint):
    """Exercise real queues/checkpoints/writer in isolated temporary roots.

    Compare 1 vs 2 raw models (extra 200 ms) on the SAME fixed jobs. This
    includes actual persistence cost but NOT discovery of compressed jobs.
    Saved source words are replaced by zero provenance and GF(2) heuristic
    navigation is OFF in both arms: no historical solutions seed the search.
    Exact GF(2) preprocessing stays ON. Formal result files stay untouched.
    """
    inventory_bytes = baseline_path.read_bytes() if baseline_path else None
    if inventory_bytes is not None and hashlib.sha256(inventory_bytes).hexdigest() != fingerprint:
        raise ValueError("inventory changed during experiment; rerun on a stable snapshot")
    arms = []
    for cap in (1, 2):
        with TemporaryDirectory(prefix="pqcp-lift-enumeration-") as directory:
            root = Path(directory)
            print("Staged test result root (temporary):", root, flush=True)
            if inventory_bytes is not None:
                (root / "{}.txt".format(length)).write_bytes(inventory_bytes)
            store = CompletionWorkStore(root, length, 123,
                SlicePolicy(lift_max_models=cap, lift_extra_ms=200, initial_ms=1000,
                            maximum_ms=1000, rectangle_search=False))
            rows = []
            started = perf_counter()
            for original, _ in selected:
                job = replace(original, source_a=(0,)*length, source_b=(0,)*length)
                key = store.register("lift", asdict(job))
                if key is None:
                    continue
                reports = []
                tick = perf_counter()
                while store.read(key)["phase"] == "READY":
                    reports.append(store.advance_lift(key))
                record = store.read(key)
                progress = record.get("lift_enumeration", {})
                models = progress.get("models", [])
                if not models:
                    for result in reports:
                        if result["solved"]:
                            snapshot = json.loads(Path(result["snapshot"]).read_text(encoding="utf-8"))
                            models.append(dict(A=snapshot["A"], B=snapshot["B"], profile=snapshot["profile"],
                                equivalence_key=list(pqcp_equivalence_key(snapshot["A"], snapshot["B"]))))
                rows.append(dict(job=job.identifier, elapsed=perf_counter()-tick,
                    phase=record["phase"], terminal=progress.get("status", record["phase"]),
                    raw_models=sum(bool(r["solved"]) for r in reports),
                    new_to_inventory=sum(bool(r["new_solution_written"]) for r in reports),
                    z3_checks=sum(r["z3_calls"] for r in reports),
                    extra_ms=progress.get("spent_ms", 0.),
                    models=models))
            elapsed = perf_counter()-started
            arm = dict(max_models=cap, extra_budget_ms=200 if cap>1 else 0, jobs=rows,
                       elapsed=elapsed, raw_models=sum(r["raw_models"] for r in rows),
                       classes=len({tuple(m["equivalence_key"]) for r in rows for m in r["models"]}),
                       new_to_inventory=sum(r["new_to_inventory"] for r in rows))
            arms.append(arm)
            print("Staged cap={} jobs={} models={} new={} elapsed={:.4f}s".format(
                cap, len(rows), arm["raw_models"], arm["new_to_inventory"], elapsed), flush=True)
    return arms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--L", type=int, default=46)
    parser.add_argument("--source", choices=("fresh", "archived"), default="fresh")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--seeds", type=int, nargs="+", default=[123, 456, 789])
    parser.add_argument("--jobs-per-seed", type=int, default=4)
    parser.add_argument("--phase", choices=("SAT", "UNSAT", "READY", "WAITING", "all"), default="SAT")
    parser.add_argument("--sample-seed", type=int, default=20260921)
    parser.add_argument("--models", type=int, default=32)
    parser.add_argument("--seconds-per-job", type=float, default=3)
    parser.add_argument("--check-timeout-ms", type=int, default=1000)
    parser.add_argument("--orientation-steps", type=int, default=128)
    parser.add_argument("--fresh-steps", type=int, default=2048)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--staged-comparison", action="store_true",
                        help="also test production 1 vs 2 model policies in temporary directories")
    parser.add_argument("--output", type=Path, default=Path("results/lift_enumeration_experiment.json"))
    args = parser.parse_args()
    if min(args.jobs_per_seed, args.models, args.check_timeout_ms, args.fresh_steps) <= 0:
        parser.error("integer work limits must be positive")
    baseline, fingerprint = inventory(args.inventory, args.L)
    if args.source == "archived":
        selected = archived_jobs(args.root, args.L, args.seeds, args.jobs_per_seed, args.phase, args.sample_seed)
    else:
        import torch
        from solver.target_profiles import canonical_target_content_profiles
        from solver.torch_compressed_search import search_compressed
        ts = canonical_target_content_profiles(args.L)
        if not ts:
            parser.error("no necessary target profiles")
        selected = []
        seen = set()
        for seed in args.seeds:
            result = search_compressed(tuple(ts[i % len(ts)] for i in range(128)),
                device=torch.device("cpu"), seed=seed, steps=args.fresh_steps, candidates=8)
            count = 0
            for job in result.jobs:
                if job.canonical_key in seen:
                    continue
                seen.add(job.canonical_key)
                selected.append((job, {"sampling": "fresh", "source_seed": seed}))
                count += 1
                if count == args.jobs_per_seed:
                    break
    rows = []
    report = dict(kind="conditional_compressed_job_enumeration_not_pipeline_speedup",
                  L=args.L, model_limit=args.models, seconds_per_job=args.seconds_per_job,
                  check_timeout_ms=args.check_timeout_ms, orientation_steps=args.orientation_steps,
                  inventory_sha256=fingerprint, inventory_classes=len(baseline), runs=rows)
    print("job seed models classes terminal total_s first_s", flush=True)
    for i, (job, provenance) in enumerate(selected):
        live = {**provenance, "job": asdict(job), "models": [], "terminal": "RUNNING",
                "elapsed": 0., "equivalence_keys": []}
        rows.append(live)
        def save_model(model):
            live["models"].append(model)
            atomic_write_json(args.output, report)
        result = enumerate_lifts(job, max_models=args.models, seconds=args.seconds_per_job,
            check_timeout_ms=args.check_timeout_ms, orientation_steps=args.orientation_steps,
            seed=provenance["source_seed"] ^ int(job.identifier[:8], 16), on_model=save_model)
        live.update(asdict(result))
        report["summary"] = summarize(rows, baseline)
        atomic_write_json(args.output, report)
        print(i, provenance["source_seed"], len(result.models), len(result.equivalence_keys),
              result.terminal, round(result.elapsed, 4),
              round(result.models[0]["elapsed"], 4) if result.models else "-", flush=True)
    report["summary"] = summarize(rows, baseline)
    if args.staged_comparison:
        report["staged_comparison"] = compare_staged(selected, args.L, args.inventory, fingerprint)
    atomic_write_json(args.output, report)
    print(json.dumps(report["summary"], indent=2), flush=True)
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
