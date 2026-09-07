"""Medium-duration paired evaluation of fixed-weight baseline and enhanced search."""

import argparse
import json
from pathlib import Path
import statistics
import sys
from time import perf_counter
from typing import Any, Dict, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from solver.enhanced_search import EnhancedParameters, EnhancedSearch
from solver.checkpoint import atomic_write_json
from solver.correlation import full_correlation_profile
from solver.objective import pqcp_objective, pqcp_objective_breakdown
from solver.search_runner import append_verified_solution_if_new
from solver.verifier import verify_pqcp


DEFAULT_SEEDS = (123, 456, 789, 1024, 2026)
DEFAULT_CHECKPOINTS = (1, 5, 10, 20, 30, 60)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--L", type=int, nargs="+", default=[44, 46])
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--seconds-per-run", type=float, default=60.0)
    parser.add_argument("--output", type=Path, default=Path("results/medium_search.jsonl"))
    parser.add_argument("--best-directory", type=Path, default=Path("results/best"))
    parser.add_argument("--checkpoints", default=",".join(map(str, DEFAULT_CHECKPOINTS)))
    return parser.parse_args()


def parameters_for(method):
    """Return the fixed Checkpoint 9 parameter family without retuning either method."""
    common = dict(stagnation_iterations=25_000, fkm_pool_size=128)
    if method == "baseline":
        return EnhancedParameters(two_bit_samples=0, **common)
    if method == "enhanced":
        return EnhancedParameters(
            two_bit_samples=128,
            escape_policy="best_sampled",
            max_escape_fraction=0.25,
            **common
        )
    raise ValueError("unknown method: {}".format(method))


def _best_path(best_directory: Path, length: int, method: str, seed: int) -> Path:
    """Return the per-run, replaceable best-candidate path."""
    return Path(best_directory) / "L{}_{}_seed{}.json".format(length, method, seed)


def build_best_candidate_event(
    search: EnhancedSearch,
    method: str,
    elapsed: float,
    old_best_score: Optional[int],
) -> Dict[str, Any]:
    """Recompute one global-best candidate independently before recording it.

    The search's incremental state is only the source of A/B.  The profile,
    objective components, and verifier result below are recomputed with the
    correctness-baseline modules.  A mismatch is an implementation error,
    not a result to be silently logged.
    """
    state = search.state
    a, b = state.best_a, state.best_b
    profile = full_correlation_profile(a, b)
    score = pqcp_objective(profile)
    if score != state.best_score:
        raise RuntimeError("global-best score disagrees with full correlation recomputation")
    verification = verify_pqcp(a, b)
    if tuple(profile) != verification.profile:
        raise RuntimeError("verifier profile disagrees with full correlation recomputation")
    if score == 0 and not verification.is_valid:
        raise RuntimeError("score-zero candidate failed independent verification")
    nonzero_shifts = [shift for shift in range(1, state.L) if profile[shift] != 0]
    event = {
        "L": state.L,
        "method": method,
        "seed": state.seed,
        "iteration": state.iteration,
        "restart_index": state.restart_index,
        "elapsed": elapsed,
        "old_best_score": old_best_score,
        "new_best_score": score,
        # ``score`` retains compatibility with the existing aggregation code.
        "score": score,
        "A": list(a),
        "B": list(b),
        "profile": profile,
        "nonzero_shifts": nonzero_shifts,
        "nonzero_values": [profile[shift] for shift in nonzero_shifts],
        "objective_components": pqcp_objective_breakdown(profile),
        "verified": verification.is_valid,
    }
    return event


def persist_best_candidate(event: Dict[str, Any], best_directory: Path) -> Path:
    """Atomically replace only this run's non-official current-best snapshot."""
    path = _best_path(Path(best_directory), event["L"], event["method"], event["seed"])
    payload = {
        "L": event["L"], "method": event["method"], "seed": event["seed"],
        "iteration": event["iteration"], "restart": event["restart_index"],
        "restart_index": event["restart_index"], "elapsed": event["elapsed"],
        "score": event["new_best_score"], "A": event["A"], "B": event["B"],
        "profile": event["profile"], "nonzero_shifts": event["nonzero_shifts"],
        "nonzero_values": event["nonzero_values"],
        "objective_components": event["objective_components"], "verified": event["verified"],
    }
    atomic_write_json(path, payload)
    return path


def _record_verified_solution(event: Dict[str, Any]) -> None:
    """Record a verified score-zero result under the project's existing rule."""
    if event["new_best_score"] == 0:
        if not event["verified"]:
            raise RuntimeError("refusing to save an unverified score-zero candidate")
        append_verified_solution_if_new(
            event["L"], tuple(event["A"]), tuple(event["B"]), PROJECT_ROOT
        )


def _save_event(event: Dict[str, Any], best_directory: Path) -> Dict[str, Any]:
    """Persist a traceable snapshot before any official score-zero recording."""
    event = dict(event)
    event["best_candidate_path"] = str(persist_best_candidate(event, best_directory))
    _record_verified_solution(event)
    return event


def run_one(length, method, seed, seconds, checkpoints, best_directory=Path("results/best"), clock=perf_counter):
    """Run one method and preserve every initial/global-best A/B snapshot."""
    search = EnhancedSearch.new(length, seed, parameters_for(method))
    started = clock()
    initial_score = search.state.best_score
    improvements = [_save_event(
        build_best_candidate_event(search, method, 0.0, None), best_directory
    )]
    snapshots = {}
    pending = [point for point in checkpoints if point <= seconds]
    verified = False
    while not search.state.finished and clock() - started < seconds:
        previous_best = search.state.best_score
        search.step()
        elapsed = clock() - started
        if search.state.best_score < previous_best:
            improvements.append(_save_event(
                build_best_candidate_event(search, method, elapsed, previous_best), best_directory
            ))
        while pending and elapsed >= pending[0]:
            snapshots[str(pending.pop(0))] = search.state.best_score
        if search.state.best_score == 0:
            verified = improvements[-1]["verified"]
            break
    elapsed = clock() - started
    for point in pending:
        snapshots[str(point)] = search.state.best_score
    state = search.state
    return {
        "L": length,
        "method": method,
        "seed": seed,
        "elapsed": elapsed,
        "iterations": state.iteration,
        "restarts": state.restart_index,
        "initial_score": initial_score,
        "best_score": state.best_score,
        "improvements": improvements,
        "snapshots": snapshots,
        "escape_triggers": state.escape_triggers,
        "accepted_escapes": state.accepted_escapes,
        "rejected_escapes": state.rejected_escapes,
        "escape_overhead": state.escape_evaluation_seconds,
        "escapes_followed_by_global_best": state.escapes_followed_by_global_best,
        "verified": verified,
    }


def first_reach_time(record, threshold):
    """Return first measured global-best time at or below threshold, else None."""
    for event in record["improvements"]:
        if event["score"] <= threshold:
            return event["elapsed"]
    return None


def aggregate(records, checkpoints):
    """Calculate distribution, threshold, and score-time statistics for one method/L."""
    scores = [record["best_score"] for record in records]
    result = {
        "minimum": min(scores), "median": statistics.median(scores),
        "mean": statistics.mean(scores), "maximum": max(scores),
        "reached_15": _reach_summary(records, 15),
        "reached_10": _reach_summary(records, 10),
        "score_zero_count": sum(record["best_score"] == 0 for record in records),
        "verified_count": sum(record["verified"] for record in records),
        "progression": {},
    }
    for point in checkpoints:
        values = [record["snapshots"].get(str(point), record["best_score"]) for record in records]
        result["progression"][point] = {"median": statistics.median(values), "mean": statistics.mean(values)}
    return result


def paired_comparison(baseline, enhanced):
    """Compare same-seed final scores; lower is better."""
    base = {record["seed"]: record["best_score"] for record in baseline}
    extra = {record["seed"]: record["best_score"] for record in enhanced}
    if set(base) != set(extra):
        raise ValueError("paired comparison requires identical seed sets")
    wins = ties = losses = 0
    for seed in sorted(base):
        if extra[seed] < base[seed]:
            wins += 1
        elif extra[seed] == base[seed]:
            ties += 1
        else:
            losses += 1
    return {"enhanced_wins": wins, "ties": ties, "baseline_wins": losses}


def _reach_summary(records, threshold):
    times = [first_reach_time(record, threshold) for record in records]
    reached = [time for time in times if time is not None]
    return {"count": len(reached), "median_time": statistics.median(reached) if reached else None}


def main() -> None:
    """Execute the requested paired benchmark and append one complete JSONL record per run."""
    args = parse_args()
    if args.seconds_per_run <= 0:
        raise SystemExit("--seconds-per-run must be positive")
    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    checkpoints = tuple(float(value) for value in args.checkpoints.split(",") if value.strip())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    all_records = []
    with args.output.open("a", encoding="utf-8") as handle:
        for length in args.L:
            for method in ("baseline", "enhanced"):
                for seed in seeds:
                    record = run_one(
                        length, method, seed, args.seconds_per_run, checkpoints,
                        best_directory=args.best_directory,
                    )
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                    handle.flush()
                    all_records.append(record)
    print("Medium paired benchmark: seeds={} seconds/run={} output={}".format(
        len(seeds), args.seconds_per_run, args.output
    ))
    for length in args.L:
        baseline = [record for record in all_records if record["L"] == length and record["method"] == "baseline"]
        enhanced = [record for record in all_records if record["L"] == length and record["method"] == "enhanced"]
        base = aggregate(baseline, checkpoints)
        extra = aggregate(enhanced, checkpoints)
        paired = paired_comparison(baseline, enhanced)
        print("L={}".format(length))
        _print_method("baseline", base)
        _print_method("enhanced", extra)
        print("paired enhanced/tie/baseline={}/{}/{}".format(
            paired["enhanced_wins"], paired["ties"], paired["baseline_wins"]
        ))
        print("progression " + " ".join(
            "{}s:B{}/E{}".format(point, base["progression"][point]["median"], extra["progression"][point]["median"])
            for point in checkpoints
        ))


def _print_method(name, values):
    print("{} min/median/mean/max={}/{}/{:.2f}/{} <=15:{}/{} <=10:{}/{} zero={} verified={}".format(
        name, values["minimum"], values["median"], values["mean"], values["maximum"],
        values["reached_15"]["count"], values["reached_15"]["median_time"],
        values["reached_10"]["count"], values["reached_10"]["median_time"],
        values["score_zero_count"], values["verified_count"],
    ))


if __name__ == "__main__":
    main()
