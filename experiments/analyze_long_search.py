"""Measure long-run behavior of the existing FKM + fixed-weight SA trajectory."""

import argparse
import json
from pathlib import Path
import statistics
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from solver.checkpoint import SearchParameters
from solver.search_runner import SearchRunner


def parse_args():
    """Parse bounded baseline-run controls; no search algorithm controls are altered."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--L", type=int, default=44)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--seconds", type=float, default=0.0)
    parser.add_argument("--minutes", type=float, default=0.0)
    parser.add_argument("--hours", type=float, default=0.0)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--events", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--best-file", type=Path)
    parser.add_argument("--checkpoint-interval", type=float, default=60.0)
    parser.add_argument("--progress-interval", type=float, default=60.0)
    parser.add_argument("--stagnation-iterations", type=int, default=100_000)
    return parser.parse_args()


def main() -> int:
    """Run measurement-only observation and print score/restart/stagnation summaries."""
    args = parse_args()
    seconds = args.seconds + args.minutes * 60.0 + args.hours * 3600.0
    if seconds <= 0:
        raise SystemExit("provide a positive --seconds, --minutes, or --hours budget")
    if args.resume is not None:
        runner = SearchRunner.resume(args.resume)
        checkpoint = args.checkpoint or args.resume
    else:
        runner = SearchRunner.new(
            args.L,
            args.seed,
            SearchParameters(stagnation_iterations=args.stagnation_iterations),
        )
        checkpoint = args.checkpoint or Path("checkpoints/analysis_L{}_seed{}.json".format(args.L, args.seed))
    events_path = args.events or Path("logs/analysis_L{}_seed{}.jsonl".format(runner.state.L, runner.state.seed))
    best_path = args.best_file or Path("results/search_L{}_analysis_best.json".format(runner.state.L))
    events_path.parent.mkdir(parents=True, exist_ok=True)

    existing_events = _read_events(events_path)
    with events_path.open("a", encoding="utf-8") as event_file:
        if not existing_events:
            _write_event(event_file, {
                "event": "start",
                "elapsed_seconds": runner.state.elapsed_seconds,
                "iteration": runner.state.iteration,
                "restart_index": runner.state.restart_index,
                "best_score": runner.state.best_score,
                "seed": runner.state.seed + runner.state.restart_index,
                "temperature": runner.state.temperature,
            })

        def event_callback(outcome, state):
            restart = outcome.restart_info
            event_seed = state.seed + (restart.restart_index if restart is not None else state.restart_index)
            event_temperature = restart.temperature if restart is not None else state.temperature
            if outcome.improved_best:
                _write_event(event_file, {
                    "event": "improvement",
                    "elapsed_seconds": state.elapsed_seconds,
                    "iteration": state.iteration,
                    "restart_index": restart.restart_index if restart is not None else state.restart_index,
                    "old_best_score": outcome.old_best_score,
                    "new_best_score": outcome.new_best_score,
                    "seed": event_seed,
                    "temperature": event_temperature,
                })
            if restart is not None:
                _write_event(event_file, {
                    "event": "restart",
                    "elapsed_seconds": state.elapsed_seconds,
                    "restart_index": restart.restart_index,
                    "starting_score": restart.start_score,
                    "local_best_score": restart.local_best_score,
                    "start_iteration": restart.start_iteration,
                    "end_iteration": restart.end_iteration,
                    "iterations_spent": restart.end_iteration - restart.start_iteration,
                    "seed": state.seed + restart.restart_index,
                    "temperature": restart.temperature,
                    "reason": restart.reason,
                    "global_best_before": outcome.old_best_score,
                    "global_best_after": outcome.new_best_score,
                })

        def progress(state, _elapsed):
            print("L={} elapsed={:.1f}s iteration={} restart={} current={} best={} temperature={:.5f}".format(
                state.L, state.elapsed_seconds, state.iteration, state.restart_index,
                state.current_score, state.best_score, state.temperature,
            ), flush=True)

        runner.run(
            seconds=seconds,
            checkpoint_path=checkpoint,
            best_path=best_path,
            checkpoint_interval=args.checkpoint_interval,
            progress_interval=args.progress_interval,
            progress_callback=progress,
            event_callback=event_callback,
        )
    events = _read_events(events_path)
    _print_analysis(events, runner.state)
    print("events={}".format(events_path))
    print("checkpoint={}".format(checkpoint))
    print("best={}".format(best_path))
    return 0


def _write_event(handle, event):
    """Append and flush an event immediately; only rare events reach this path."""
    handle.write(json.dumps(event, sort_keys=True) + "\n")
    handle.flush()


def _read_events(path):
    """Read valid JSONL events, rejecting an interrupted/corrupt event log clearly."""
    if not path.exists():
        return []
    try:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except json.JSONDecodeError as error:
        raise RuntimeError("invalid event log {}: {}".format(path, error)) from error


def _print_analysis(events, state):
    """Report only measured progression, intervals, restart outcomes, and plateaus."""
    starts = [event for event in events if event["event"] == "start"]
    improvements = [event for event in events if event["event"] == "improvement"]
    restarts = [event for event in events if event["event"] == "restart"]
    progression = [(event["elapsed_seconds"], event.get("best_score", event.get("new_best_score")))
                   for event in starts + improvements]
    progression.sort()
    print("best-score progression:")
    for elapsed, score in progression:
        print("  {:.3f}s -> {}".format(elapsed, score))

    improvement_times = [elapsed for elapsed, _ in progression]
    intervals = [right - left for left, right in zip(improvement_times, improvement_times[1:])]
    interval_iterations = [
        right["iteration"] - left["iteration"]
        for left, right in zip([event for event in starts + improvements], [event for event in starts + improvements][1:])
    ]
    local_scores = [event["local_best_score"] for event in restarts]
    improving_restarts = sum(event["global_best_after"] < event["global_best_before"] for event in restarts)
    print("improvement intervals: count={} median_time={}s longest_time={}s median_iterations={}".format(
        len(intervals), _median_or_none(intervals), max(intervals) if intervals else None,
        _median_or_none(interval_iterations),
    ))
    print("restarts: count={} global-improving={} non-improving={} local_best_mean={} median={} min={} max={}".format(
        len(restarts), improving_restarts, len(restarts) - improving_restarts,
        _mean_or_none(local_scores), _median_or_none(local_scores),
        min(local_scores) if local_scores else None, max(local_scores) if local_scores else None,
    ))
    last_improvement_time = improvement_times[-1] if improvement_times else 0.0
    current_gap = max(0.0, state.elapsed_seconds - last_improvement_time)
    print("stagnation: current_without_improvement={:.3f}s longest_observed_interval={}s".format(
        current_gap, max(intervals) if intervals else None
    ))
    if not improvements:
        interpretation = "no measured global improvement after initialization; plateau evidence is strong for this run"
    elif current_gap > max(1.0, state.elapsed_seconds * 0.5):
        interpretation = "improvements occurred early but the latter half is plateau-like"
    else:
        interpretation = "improvements are still present within the latter half; no clear plateau conclusion yet"
    print("measurement interpretation: {}".format(interpretation))
    print("final best score={}".format(state.best_score))


def _median_or_none(values):
    """Return a concise optional median for an empty/nonempty measurement list."""
    return statistics.median(values) if values else None


def _mean_or_none(values):
    """Return a concise optional mean for an empty/nonempty measurement list."""
    return statistics.mean(values) if values else None


if __name__ == "__main__":
    raise SystemExit(main())
