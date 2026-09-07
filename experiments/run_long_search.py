"""CLI for checkpointed, sequential, long-running FKM-initialized SA search."""

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from solver.checkpoint import SearchParameters
from solver.search_runner import (
    DEFAULT_GUIDED_OBJECTIVE_ENERGY_WEIGHT,
    DEFAULT_GUIDED_PROPOSAL_SAMPLES,
    SearchRunner,
    append_verified_solution_if_new,
)
from solver.weight_constraints import canonical_weight_pairs


PROJECT_LENGTHS = frozenset((44, 46, 58, 68, 86, 90, 94))


def parse_args():
    """Parse either a new run or a direct JSON checkpoint resume request."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--L", type=int, help="binary sequence length for a new run")
    parser.add_argument("--resume", type=Path, help="checkpoint JSON to resume")
    parser.add_argument("--seed", type=int, default=123, help="base seed for a new run")
    parser.add_argument("--seconds", type=float, default=0.0)
    parser.add_argument("--minutes", type=float, default=0.0)
    parser.add_argument("--hours", type=float, default=0.0)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--best-file", type=Path)
    parser.add_argument("--checkpoint-interval", type=float, default=60.0)
    parser.add_argument("--progress-interval", type=float, default=60.0)
    parser.add_argument("--stagnation-iterations", type=int, default=100_000)
    parser.add_argument("--max-iterations-per-restart", type=int)
    parser.add_argument("--max-restarts", type=int)
    parser.add_argument("--fkm-pool-size", type=int, default=128)
    parser.add_argument(
        "--proposal-samples", type=int, default=DEFAULT_GUIDED_PROPOSAL_SAMPLES,
        help="exact legal swaps sampled per SA proposal (1 reproduces the baseline)",
    )
    parser.add_argument("--weight", type=int)
    parser.add_argument(
        "--acceptance-mode", choices=(
            "objective", "target_pair_squared", "objective_plus_target_pair",
        ),
        help="override automatic Project-length guided acceptance",
    )
    return parser.parse_args()


def main() -> int:
    """Run safely, save on all normal/interrupt exits, and print saved paths."""
    args = parse_args()
    seconds = args.seconds + args.minutes * 60.0 + args.hours * 3600.0
    if seconds <= 0:
        raise SystemExit("provide a positive --seconds, --minutes, or --hours budget")
    if args.resume is not None:
        runner = SearchRunner.resume(args.resume)
        checkpoint_path = args.checkpoint or args.resume
    else:
        if args.L is None:
            raise SystemExit("--L is required unless --resume is supplied")
        guided = args.acceptance_mode in ("target_pair_squared", "objective_plus_target_pair") or (
            args.acceptance_mode is None and args.L in PROJECT_LENGTHS
        )
        weight_pairs = None
        if args.weight is None and args.L in PROJECT_LENGTHS:
            weight_pairs = canonical_weight_pairs(args.L)
            if not weight_pairs:
                raise SystemExit("L={} has no admissible Project weight pair".format(args.L))
        parameters = SearchParameters(
            fkm_pool_size=args.fkm_pool_size,
            weight=args.weight,
            weight_pairs=weight_pairs,
            stagnation_iterations=args.stagnation_iterations,
            max_iterations_per_restart=args.max_iterations_per_restart,
            max_restarts=args.max_restarts,
            acceptance_mode=(args.acceptance_mode or "objective_plus_target_pair") if guided else "objective",
            initial_temperature=30.0 if args.acceptance_mode == "target_pair_squared" else 8.0,
            proposal_samples=args.proposal_samples,
            objective_energy_weight=DEFAULT_GUIDED_OBJECTIVE_ENERGY_WEIGHT,
        )
        runner = SearchRunner.new(args.L, args.seed, parameters)
        checkpoint_path = args.checkpoint or Path("checkpoints/L{}.json".format(args.L))
    best_path = args.best_file or Path("results/search_L{}_best.json".format(runner.state.L))

    def progress(state, invocation_elapsed):
        print(
            "L={} elapsed={:.1f}s iterations={} restart={} current_score={} best_score={} "
            "temperature={:.6f} since_improvement={}".format(
                state.L, state.elapsed_seconds, state.iteration, state.restart_index,
                state.current_score, state.best_score, state.temperature,
                state.iteration - state.last_improvement_iteration,
            ),
            flush=True,
        )

    def record_solution(a, b, state):
        is_new = append_verified_solution_if_new(state.L, a, b, PROJECT_ROOT)
        print("verified solution {} in {}.txt".format("recorded" if is_new else "already present", state.L), flush=True)

    summary = runner.run(
        seconds=seconds,
        checkpoint_path=checkpoint_path,
        best_path=best_path,
        checkpoint_interval=args.checkpoint_interval,
        progress_interval=args.progress_interval,
        progress_callback=progress,
        on_verified_solution=record_solution,
    )
    if summary.interrupted:
        print("interrupted safely; checkpoint saved to {}".format(summary.checkpoint_path))
    else:
        print("time budget complete; checkpoint saved to {}".format(summary.checkpoint_path))
    print("best candidate saved to {}".format(summary.best_path))
    print("best_score={} iterations={} restarts={}".format(
        summary.state.best_score, summary.state.iteration, summary.state.restart_index
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
