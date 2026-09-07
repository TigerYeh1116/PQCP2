"""Fair wall-clock comparison: fixed-weight baseline vs sampled swap escape."""

import argparse
import statistics
from time import perf_counter
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from solver.enhanced_search import EnhancedParameters, EnhancedSearch


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--L", type=int, nargs="+", default=[44, 46, 58, 68])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--seconds", type=float, default=5.0, help="equal wall-clock budget per method/seed")
    parser.add_argument("--two-bit-samples", type=int, default=128)
    parser.add_argument("--stagnation-iterations", type=int, default=25_000)
    return parser.parse_args()


def run_method(length, seed, seconds, parameters):
    """Run one reproducible method with the caller-provided equal wall budget."""
    search = EnhancedSearch.new(length, seed, parameters)
    started = perf_counter()
    result = search.run(seconds)
    return {
        "score": result.state.best_score,
        "runtime": perf_counter() - started,
        "iterations": result.state.iteration,
        "restarts": result.state.restart_index,
        "escape_triggers": result.state.escape_triggers,
        "sampled_moves": result.state.sampled_moves,
        "accepted_escapes": result.state.accepted_escapes,
        "rejected_escapes": result.state.rejected_escapes,
        "escape_time": result.state.escape_evaluation_seconds,
        "escapes_followed_by_global_best": result.state.escapes_followed_by_global_best,
        "verified": result.verified,
    }


def summary(records):
    """Return score/rate statistics without selecting only a lucky run."""
    scores = [record["score"] for record in records]
    return {
        "minimum": min(scores),
        "median": statistics.median(scores),
        "mean": statistics.mean(scores),
        "maximum": max(scores),
        "runtime": statistics.mean(record["runtime"] for record in records),
        "iterations": statistics.mean(record["iterations"] for record in records),
        "restarts": statistics.mean(record["restarts"] for record in records),
    }


def main() -> None:
    """Print exactly comparable aggregate distributions across independent seeds."""
    args = parse_args()
    seeds = tuple(20_260_900 + index for index in range(args.seeds))
    baseline_parameters = EnhancedParameters(
        stagnation_iterations=args.stagnation_iterations,
        two_bit_samples=0,
    )
    enhanced_parameters = EnhancedParameters(
        stagnation_iterations=args.stagnation_iterations,
        two_bit_samples=args.two_bit_samples,
        escape_policy="best_sampled",
        max_escape_fraction=0.25,
    )
    print("Enhanced neighborhood benchmark")
    print("=" * 72)
    print("seeds={} equal_seconds_per_method={} samples={}".format(args.seeds, args.seconds, args.two_bit_samples))
    for length in args.L:
        baseline = [run_method(length, seed, args.seconds, baseline_parameters) for seed in seeds]
        enhanced = [run_method(length, seed, args.seconds, enhanced_parameters) for seed in seeds]
        base = summary(baseline)
        extra = summary(enhanced)
        print("L = {}".format(length))
        print("baseline score min/median/mean/max={minimum}/{median}/{mean:.2f}/{maximum} "
              "runtime={runtime:.3f}s iterations={iterations:.0f} restarts={restarts:.1f}".format(**base))
        print("enhanced score min/median/mean/max={minimum}/{median}/{mean:.2f}/{maximum} "
              "runtime={runtime:.3f}s iterations={iterations:.0f} restarts={restarts:.1f}".format(**extra))
        print("enhanced escapes triggers={} accepted={} rejected={} avg_samples_per_trigger={:.1f} "
              "escape_overhead={:.3f}s followed_by_global_best={} verified={}".format(
                  sum(record["escape_triggers"] for record in enhanced),
                  sum(record["accepted_escapes"] for record in enhanced),
                  sum(record["rejected_escapes"] for record in enhanced),
                  _safe_divide(sum(record["sampled_moves"] for record in enhanced),
                               sum(record["escape_triggers"] for record in enhanced)),
                  sum(record["escape_time"] for record in enhanced),
                  sum(record["escapes_followed_by_global_best"] for record in enhanced),
                  any(record["verified"] for record in enhanced),
              ))
        print("-" * 72)


def _safe_divide(numerator, denominator):
    return numerator / denominator if denominator else 0.0


if __name__ == "__main__":
    main()
