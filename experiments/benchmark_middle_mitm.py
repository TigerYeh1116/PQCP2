"""Paired L68 middle-MITM vs existing Z3 evaluation, never production search.

Read only mathematical specs from queued compression jobs, not their prior
timings, scores, source sequences or solutions. Deduplicate by proven job
equivalence and select in filename order. Each method runs alone in a fresh
process, with the SAME external wall deadline (including model construction).
No production checkpoint or L.txt is written. Output directory must be new.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import platform
import resource
import statistics
import sys
from time import perf_counter
import traceback

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from experiments.middle_mitm import solve_middle_mitm
from solver.checkpoint import atomic_write_json
from solver.compressed_equivalence import compressed_job_equivalence_key
from solver.structured_energy import fold_profile, integer_periodic_autocorrelation_profile, target_profile
from solver.target_profiles import TargetContentProfile


def validate_job(ca,cb,profile):
    """Validate the replayed mathematical problem, independently of origin."""
    if profile.L != 68:
        raise ValueError("this benchmark requires L68")
    key = compressed_job_equivalence_key(ca,cb,profile,4)
    actual = tuple(x+y for x,y in zip(integer_periodic_autocorrelation_profile(ca),
                                     integer_periodic_autocorrelation_profile(cb)))
    if actual != fold_profile(target_profile(68,profile.k,profile.eta),4):
        raise ValueError("invalid exact coarse job")
    return key


def read_jobs(directory: Path, limit: int):
    """Verify checksums, validate exact coarse targets, retain distinct jobs."""
    jobs, seen = [], set()
    for path in sorted(directory.glob("*.json")):
        data = json.loads(path.read_text())
        record = data["record"]
        digest = hashlib.sha256(json.dumps(record,sort_keys=True,separators=(",",":"),
                                          allow_nan=False).encode()).hexdigest()
        if digest != data["sha256"]:
            raise ValueError("checkpoint checksum mismatch: " + str(path))
        if record["kind"] != "lift" or record["spec"]["factor"] != 4:
            continue
        spec = record["spec"]
        profile = TargetContentProfile(**spec["target"])
        if profile.L != 68:
            continue
        ca,cb = tuple(spec["compressed_a"]),tuple(spec["compressed_b"])
        key = validate_job(ca,cb,profile)
        if key in seen:
            continue
        seen.add(key)
        jobs.append(dict(id=path.stem,source=str(path),compressed_a=ca,compressed_b=cb,
                         target=asdict(profile),source_sha256=data["sha256"]))
        if len(jobs) == limit:
            break
    return jobs


def read_manifest(path: Path, limit: int):
    """Replay saved inputs even if the production queue later changes."""
    jobs,seen = [],set()
    for job in json.loads(path.read_text())["jobs"]:
        key = validate_job(tuple(job["compressed_a"]),tuple(job["compressed_b"]),
                           TargetContentProfile(**job["target"]))
        if key not in seen:
            seen.add(key)
            jobs.append(job)
        if len(jobs) == limit:
            break
    return jobs


def run_baseline(ca,cb,profile,seconds,emit,binary_slice_ms=0):
    """Observe existing one-shot solver; don't change encoding/solver decisions."""
    import solver.multilevel_lift as module
    start = perf_counter()
    stats = dict(status="UNKNOWN",stage="prepare",verified=False,
                 middle_models=0,middle_checks=0,binary_started=0,binary_checks=0,
                 binary_unsat=0,unknown_children=0,binary_history=[])
    old_build,old_binary = module.build_middle_solver,module.solve_compressed_lift

    def build(*args,**kwargs):
        stats.update(stage="middle_construction",elapsed=perf_counter()-start)
        emit(stats)
        value = old_build(*args,**kwargs)
        stats.update(stage="middle_check",construction_seconds=perf_counter()-start,
                     elapsed=perf_counter()-start)
        emit(stats)
        return value

    def binary(*args,**kwargs):
        stats["binary_started"] += 1
        stats["middle_models"] += 1
        stats.setdefault("first_middle_seconds",perf_counter()-start)
        stats.update(stage="binary",elapsed=perf_counter()-start)
        if binary_slice_ms:
            # Same optional per-child scheduling limit for BOTH methods;
            # equations and underlying exact solver are unchanged.
            args = list(args)
            args[4] = min(args[4],binary_slice_ms)
        attempt = dict(middle_a=args[0],middle_b=args[1],status="IN_FLIGHT",
                       started_seconds=perf_counter()-start,check_timeout_ms=args[4])
        stats["binary_history"].append(attempt)
        emit(stats)
        child = old_binary(*args,**kwargs)
        attempt.update(status=child.status,construction_seconds=child.construction_time,
                       solve_seconds=child.solve_time)
        stats["binary_checks"] += int(child.z3_executed)
        stats["binary_unsat"] += int(child.status == "UNSAT")
        stats["unknown_children"] += int(child.status == "UNKNOWN")
        stats["stage"] = "complete" if child.status == "SAT" else "middle_check"
        stats["elapsed"] = perf_counter()-start
        emit(stats)
        return child

    module.build_middle_solver,module.solve_compressed_lift = build,binary
    try:
        result = module.solve_multilevel_lift(ca,cb,profile.k,profile.eta,
                         timeout_ms=max(1,int(seconds*1000)),content_profile=profile,
                         max_middle_models=16,cross_compression_pruning=True)
    finally:
        module.build_middle_solver,module.solve_compressed_lift = old_build,old_binary
    stats.update(status=result.status,reason=result.reason,verified=result.verified,
                 elapsed=perf_counter()-start,middle_models=result.middle_models,
                 middle_checks=result.middle_checks,binary_checks=result.binary_checks,
                 construction_seconds=result.construction_time,solve_seconds=result.solve_time,
                 unknown_children=result.unknown_children)
    if result.verified:
        stats.update(A=result.a,B=result.b,profile=result.profile)
    return stats


def _worker(connection,method,job,max_index,binary_slice_ms=0):
    """Run one fresh solver; import cost is recorded separately by parent."""
    try:
        import z3
        connection.send(dict(kind="ready",z3=z3.get_version_string()))
        seconds = connection.recv()
        def emit(stats):
            stats = dict(stats)
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            stats["peak_rss_mib"] = rss/(1024**2 if sys.platform == "darwin" else 1024)
            connection.send(dict(kind="progress",stats=stats))
        profile = TargetContentProfile(**job["target"])
        ca,cb = tuple(job["compressed_a"]),tuple(job["compressed_b"])
        if method == "baseline":
            result = run_baseline(ca,cb,profile,seconds,emit,binary_slice_ms)
        else:
            result = solve_middle_mitm(ca,cb,profile,seconds,max_index_entries=max_index,
                                       max_middle_models=16,progress=emit,binary_slice_ms=binary_slice_ms)
        emit(result)
        connection.send(dict(kind="final",stats=result))
    except BaseException:
        connection.send(dict(kind="final",stats=dict(status="ERROR",verified=False,
                                                     reason=traceback.format_exc())))
    finally:
        connection.close()


def bounded_run(job,method,seconds,max_index,binary_slice_ms=0):
    """An external deadline includes enumeration, hashing and all Z3 building."""
    ctx = mp.get_context("spawn")
    parent,child = ctx.Pipe()
    process = ctx.Process(target=_worker,args=(child,method,job,max_index,binary_slice_ms))
    launched = perf_counter()
    process.start()
    child.close()
    last = dict(status="UNKNOWN",verified=False,stage="starting")
    done = False
    try:
        if not parent.poll(30):
            raise RuntimeError("worker startup timeout")
        ready = parent.recv()
        if ready["kind"] != "ready":
            raise RuntimeError(str(ready))
        startup = perf_counter()-launched
        start = perf_counter()
        parent.send(seconds)
        while perf_counter()-start < seconds:
            if parent.poll(min(.05,max(0,seconds-(perf_counter()-start)))):
                message = parent.recv()
                last.update(message["stats"])
                # A verified SAT already published by the worker must not
                # be lost merely because the trailing final message races
                # with the parent's deadline.
                if message["kind"] == "final" or (last.get("status") == "SAT" and last.get("verified")):
                    done = True
                    break
        wall = perf_counter()-start
        if not done:
            last.update(status="UNKNOWN",verified=False,reason="external wall-clock deadline",
                        censored=True)
        else:
            last["censored"] = False
        last.update(job=job["id"],method=method,budget_seconds=seconds,
                    measured_wall_seconds=wall,startup_seconds=startup,
                    z3_version=ready["z3"])
        return last
    finally:
        if process.is_alive():
            process.terminate()
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join()
        parent.close()


def summarize(records):
    """Keep censored first-hit times distinct from reached-only medians."""
    methods = {}
    for method in sorted({r["method"] for r in records}):
        rows = [r for r in records if r["method"] == method]
        times = [r["first_middle_seconds"] for r in rows if "first_middle_seconds" in r]
        methods[method] = dict(
            runs=len(rows),verified=sum(r["verified"] for r in rows),
            first_middle_reached=len(times),first_middle_not_reached=len(rows)-len(times),
            median_first_middle_reached_only=statistics.median(times) if times else None,
            max_first_middle_reached=max(times) if times else None,
            binary_started=sum(r.get("binary_started",0) for r in rows),
            binary_unsat_observed=sum(r.get("binary_unsat",0) for r in rows),
            binary_unknown_observed=sum(r.get("unknown_children",0) for r in rows),
            parent_unsat=sum(r["status"] == "UNSAT" for r in rows),
            wall_seconds=sum(r["measured_wall_seconds"] for r in rows),
        )
    pairs = {}
    for row in records:
        pairs.setdefault(row["job"],{})[row["method"]] = row
    wins = dict(mitm_earlier=0,baseline_earlier=0,equal_reached=0,both_not_reached=0)
    for pair in pairs.values():
        if not all(m in pair for m in ("baseline","mitm")):
            continue
        b,m = pair["baseline"],pair["mitm"]
        if b["budget_seconds"] != m["budget_seconds"]:
            raise ValueError("paired budgets differ")
        bt,mt = b.get("first_middle_seconds"),m.get("first_middle_seconds")
        if bt is None and mt is None:
            wins["both_not_reached"] += 1
        elif bt is None or (mt is not None and mt < bt):
            wins["mitm_earlier"] += 1
        elif mt is None or bt < mt:
            wins["baseline_earlier"] += 1
        else:
            wins["equal_reached"] += 1
    return dict(methods=methods,paired_first_middle=wins,
                caveat="First-middle wins are not new-PQCP speedups; killed-worker counters/RSS are last-observed lower bounds.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,default=Path("checkpoints/z3/L68_seed123"))
    parser.add_argument("--manifest",type=Path,help="Replay a saved inputs.json instead of the live queue")
    parser.add_argument("--jobs",type=int,default=8)
    parser.add_argument("--seconds-per-job",type=float,default=10)
    parser.add_argument("--max-index-entries",type=int,default=250000)
    parser.add_argument("--binary-slice-ms",type=int,default=0,
                        help="Optional SAME binary check slice for both methods; 0 uses remaining budget")
    parser.add_argument("--methods",nargs="+",choices=("baseline","mitm"),default=["baseline","mitm"])
    parser.add_argument("--output-dir",type=Path,required=True)
    args = parser.parse_args()
    if min(args.jobs,args.seconds_per_job,args.max_index_entries) <= 0 or args.binary_slice_ms < 0:
        parser.error("budgets must be positive")
    jobs = read_manifest(args.manifest,args.jobs) if args.manifest else read_jobs(args.input,args.jobs)
    if not jobs:
        parser.error("no valid factor-4 L68 jobs")
    args.output_dir.mkdir(parents=True,exist_ok=False)
    atomic_write_json(args.output_dir / "inputs.json",dict(jobs=jobs,python=sys.version,
        platform=platform.platform(),budget=args.seconds_per_job,
        methods=args.methods,max_index_entries=args.max_index_entries,
        binary_slice_ms=args.binary_slice_ms,
        selection="filename order, equivalent jobs removed; prior search outcomes ignored"))
    records = []
    with (args.output_dir / "runs.jsonl").open("x") as stream:
        for i,job in enumerate(jobs):
            methods = args.methods if i%2 == 0 else list(reversed(args.methods))
            for method in methods:
                result = bounded_run(job,method,args.seconds_per_job,args.max_index_entries,args.binary_slice_ms)
                stream.write(json.dumps(result,sort_keys=True,allow_nan=False)+"\n")
                stream.flush()
                records.append(result)
                print("{} {} {} wall={:.2f}s middle={} binary_started={} verified={}".format(
                    job["id"][:10],method,result["status"],result["measured_wall_seconds"],
                    result.get("middle_models",0),result.get("binary_started",0),result["verified"]),flush=True)
                if result["status"] == "ERROR":
                    raise RuntimeError(result["reason"])
    atomic_write_json(args.output_dir / "summary.json",summarize(records))


if __name__ == "__main__":
    main()
