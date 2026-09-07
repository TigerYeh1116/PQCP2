"""Equal-wall ablation of profile-aware PyTorch inside structured PQCP search."""

import argparse
import json
from pathlib import Path
import random
import re
from statistics import mean, median
import sys
from time import perf_counter
from typing import Dict, List, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from solver.checkpoint import SearchParameters
from solver.annealing import initialize_from_fkm_content_profile
from solver.search_runner import SearchRunner
from solver.target_profiles import (
    TargetContentProfile, canonical_target_content_profiles,
    pair_content, target_content_profiles,
)
from solver.torch_relaxation import (
    RelaxationParameters, relax_candidate_batch_for_profile,
    relax_candidate_for_profile,
)
from solver.verifier import verify_pqcp
from solver.torch_guidance import torch_gradient_kick
from solver.compression import CorrelationState
from solver.moves import (
    apply_weight_preserving_swap, sample_weight_preserving_swaps,
    trial_weight_preserving_swap,
)


METHODS = (
    "sa", "random_sa", "torch_warm", "torch_late",
    "torch_batch_warm", "torch_batch_late", "torch_random_warm", "torch_only",
    "torch_gradient_late",
    "random_kick_late", "sampled_kick_late",
    "kick_portfolio_late",
    "sampled_portfolio_late",
)


def encode_profiles(profiles: Sequence[TargetContentProfile]):
    return tuple(
        (p.k, p.eta, p.a_even_ones, p.a_odd_ones,
         p.b_even_ones, p.b_odd_ones) for p in profiles
    )


def search_parameters(profile: TargetContentProfile) -> SearchParameters:
    """Use one fixed subproblem so A/B comparisons cannot switch targets."""
    return SearchParameters(
        stagnation_iterations=None,
        target_content_profiles=encode_profiles((profile,)),
        preserve_alternating_content=True,
        acceptance_mode="fixed_target_full",
        proposal_samples=10,
    )


def run_sa_until(runner: SearchRunner, deadline: float, started: float) -> List[Dict[str, object]]:
    """Run exact SA to a deadline and retain only global improvements."""
    progression = [{"elapsed": perf_counter() - started, "score": runner.state.best_score}]
    while perf_counter() < deadline:
        outcome = runner.step()
        if outcome.improved_best:
            progression.append({"elapsed": perf_counter() - started, "score": runner.state.best_score})
    return progression


def run_wall_case(
    length: int,
    seed: int,
    seconds: float,
    method: str,
    torch_steps: int,
    torch_batch_size: int = 8,
) -> Dict[str, object]:
    """Run one method from the identical FKM state under an equal wall budget."""
    profiles = canonical_target_content_profiles(length)
    if not profiles:
        return {"L": length, "seed": seed, "method": method, "skipped": "no target profile"}
    # Rotate cases deterministically across seeds while keeping the same case
    # for all methods using that seed.
    profile = profiles[seed % len(profiles)]
    parameters = search_parameters(profile)
    initial = SearchRunner.new(length, seed, parameters)
    initial_pair = (initial.state.current_a, initial.state.current_b)
    initial_score = initial.state.current_score
    started = perf_counter()
    deadline = started + seconds
    torch_seconds = 0.0
    torch_initial_score = None
    torch_best_score = None
    reported_iterations = None
    iteration_offset = 0

    def random_candidate(candidate_seed: int):
        rng = random.Random(candidate_seed)
        def sequence(even_weight: int, odd_weight: int):
            result = [0] * length
            for parity, weight in ((0, even_weight), (1, odd_weight)):
                positions = list(range(parity, length, 2))
                rng.shuffle(positions)
                for index in positions[:weight]:
                    result[index] = 1
            return tuple(result)
        return (
            sequence(profile.a_even_ones, profile.a_odd_ones),
            sequence(profile.b_even_ones, profile.b_odd_ones),
        )

    if method == "sa":
        runner = initial
        progression = run_sa_until(runner, deadline, started)
    elif method == "random_sa":
        candidate = random_candidate(seed)
        runner = SearchRunner.from_candidate(*candidate, seed, parameters)
        progression = run_sa_until(runner, deadline, started)
    elif method == "torch_warm":
        torch_started = perf_counter()
        relaxed = relax_candidate_for_profile(
            *initial_pair, profile,
            RelaxationParameters(steps=torch_steps, seed=seed),
        )
        torch_seconds = perf_counter() - torch_started
        torch_initial_score, torch_best_score = relaxed.initial_score, relaxed.best_score
        runner = SearchRunner.from_candidate(relaxed.best_a, relaxed.best_b, seed, parameters)
        progression = [{"elapsed": torch_seconds, "score": runner.state.best_score}]
        progression.extend(run_sa_until(runner, deadline, started))
    elif method == "torch_late":
        runner = initial
        first_deadline = started + 0.65 * seconds
        progression = run_sa_until(runner, first_deadline, started)
        center = (runner.state.best_a, runner.state.best_b)
        torch_started = perf_counter()
        relaxed = relax_candidate_for_profile(
            *center, profile,
            RelaxationParameters(steps=torch_steps, seed=seed + 1_000_000),
        )
        torch_seconds = perf_counter() - torch_started
        torch_initial_score, torch_best_score = relaxed.initial_score, relaxed.best_score
        if relaxed.best_score < runner.state.best_score:
            progression.append({"elapsed": perf_counter() - started, "score": relaxed.best_score})
            runner = SearchRunner.from_candidate(
                relaxed.best_a, relaxed.best_b, seed + 2_000_000, parameters
            )
        progression.extend(run_sa_until(runner, deadline, started))
    elif method == "torch_batch_warm":
        torch_started = perf_counter()
        candidates = tuple(
            initialize_from_fkm_content_profile(
                profile, seed=seed + 10_000 * index, pool_size=128
            )
            for index in range(torch_batch_size)
        )
        relaxed = relax_candidate_batch_for_profile(
            candidates, profile,
            RelaxationParameters(steps=torch_steps, seed=seed),
        )
        torch_seconds = perf_counter() - torch_started
        torch_initial_score, torch_best_score = relaxed.initial_score, relaxed.best_score
        runner = SearchRunner.from_candidate(relaxed.best_a, relaxed.best_b, seed, parameters)
        progression = [{"elapsed": torch_seconds, "score": runner.state.best_score}]
        progression.extend(run_sa_until(runner, deadline, started))
    elif method == "torch_batch_late":
        runner = initial
        first_deadline = started + 0.65 * seconds
        progression = run_sa_until(runner, first_deadline, started)
        center = (runner.state.best_a, runner.state.best_b)
        torch_started = perf_counter()
        relaxed = relax_candidate_batch_for_profile(
            tuple(center for _ in range(torch_batch_size)), profile,
            RelaxationParameters(
                steps=torch_steps, seed=seed + 1_000_000, jitter=0.25,
            ),
        )
        torch_seconds = perf_counter() - torch_started
        torch_initial_score, torch_best_score = relaxed.initial_score, relaxed.best_score
        if relaxed.best_score < runner.state.best_score:
            progression.append({"elapsed": perf_counter() - started, "score": relaxed.best_score})
            runner = SearchRunner.from_candidate(
                relaxed.best_a, relaxed.best_b, seed + 2_000_000, parameters
            )
        progression.extend(run_sa_until(runner, deadline, started))
    elif method == "torch_random_warm":
        torch_started = perf_counter()
        candidates = tuple(
            random_candidate(seed + 10_000 * index)
            for index in range(torch_batch_size)
        )
        relaxed = relax_candidate_batch_for_profile(
            candidates, profile,
            RelaxationParameters(steps=torch_steps, seed=seed),
        )
        torch_seconds = perf_counter() - torch_started
        torch_initial_score, torch_best_score = relaxed.initial_score, relaxed.best_score
        runner = SearchRunner.from_candidate(relaxed.best_a, relaxed.best_b, seed, parameters)
        progression = [{"elapsed": torch_seconds, "score": runner.state.best_score}]
        progression.extend(run_sa_until(runner, deadline, started))
    elif method == "torch_only":
        best_a, best_b = initial_pair
        best_score = initial_score
        progression = [{"elapsed": 0.0, "score": best_score}]
        batches = 0
        torch_started = perf_counter()
        while perf_counter() < deadline:
            candidates = tuple(
                random_candidate(seed + batches * 1_000_000 + 10_000 * index)
                for index in range(torch_batch_size)
            )
            relaxed = relax_candidate_batch_for_profile(
                candidates, profile,
                RelaxationParameters(
                    steps=torch_steps, seed=seed + batches, jitter=0.25,
                ),
            )
            batches += 1
            if relaxed.best_score < best_score:
                best_a, best_b, best_score = relaxed.best_a, relaxed.best_b, relaxed.best_score
                progression.append({"elapsed": perf_counter() - started, "score": best_score})
                if relaxed.verified:
                    break
        torch_seconds = perf_counter() - torch_started
        torch_initial_score, torch_best_score = initial_score, best_score
        runner = SearchRunner.from_candidate(best_a, best_b, seed, parameters)
        reported_iterations = batches
    elif method == "torch_gradient_late":
        runner = initial
        first_deadline = started + 0.65 * seconds
        progression = run_sa_until(runner, first_deadline, started)
        prior = (runner.state.best_a, runner.state.best_b, runner.state.best_score)
        iteration_offset = runner.state.iteration
        torch_started = perf_counter()
        kicked_a, kicked_b = torch_gradient_kick(
            prior[0], prior[1], profile, steps=4, candidate_limit=32
        )
        torch_seconds = perf_counter() - torch_started
        runner = SearchRunner.from_candidate(
            kicked_a, kicked_b, seed + 2_000_000, parameters
        )
        # The kick is allowed uphill, but the run's global best is not lost.
        if prior[2] < runner.state.best_score:
            runner.state.best_a, runner.state.best_b, runner.state.best_score = prior
        progression.extend(run_sa_until(runner, deadline, started))
    elif method in ("random_kick_late", "sampled_kick_late"):
        runner = initial
        first_deadline = started + 0.65 * seconds
        progression = run_sa_until(runner, first_deadline, started)
        prior = (runner.state.best_a, runner.state.best_b, runner.state.best_score)
        iteration_offset = runner.state.iteration
        kick_started = perf_counter()
        kick_state = CorrelationState(prior[0], prior[1])
        kick_rng = random.Random(seed + 1_000_000)

        def fixed_energy(values):
            return sum(
                (values[shift] - (profile.target_value if shift == profile.k else 0)) ** 2
                for shift in range(1, length // 2 + 1)
            )

        for _ in range(4):
            candidates = sample_weight_preserving_swaps(
                kick_state.a, kick_state.b, kick_rng,
                1 if method == "random_kick_late" else 32,
                same_parity=True,
            )
            if not candidates:
                break
            if method == "random_kick_late":
                selected = candidates[0]
            else:
                selected = min(
                    candidates,
                    key=lambda move: fixed_energy(
                        trial_weight_preserving_swap(kick_state, move).profile
                    ),
                )
            apply_weight_preserving_swap(kick_state, selected)
        torch_seconds = perf_counter() - kick_started
        runner = SearchRunner.from_candidate(
            kick_state.a, kick_state.b, seed + 2_000_000, parameters
        )
        if prior[2] < runner.state.best_score:
            runner.state.best_a, runner.state.best_b, runner.state.best_score = prior
        progression.extend(run_sa_until(runner, deadline, started))
    elif method in ("kick_portfolio_late", "sampled_portfolio_late"):
        runner = initial
        prefix_deadline = started + 0.65 * seconds
        progression = run_sa_until(runner, prefix_deadline, started)
        prior = (runner.state.best_a, runner.state.best_b, runner.state.best_score)
        iteration_offset = runner.state.iteration
        kick_started = perf_counter()
        def fixed_energy(values):
            return sum(
                (values[shift] - (profile.target_value if shift == profile.k else 0)) ** 2
                for shift in range(1, length // 2 + 1)
            )

        def sampled_pair(kick_seed):
            sampled_state = CorrelationState(prior[0], prior[1])
            sampled_rng = random.Random(kick_seed)
            for _ in range(4):
                candidates = sample_weight_preserving_swaps(
                    sampled_state.a, sampled_state.b, sampled_rng, 32,
                    same_parity=True,
                )
                if not candidates:
                    break
                selected = min(
                    candidates,
                    key=lambda move: fixed_energy(
                        trial_weight_preserving_swap(sampled_state, move).profile
                    ),
                )
                apply_weight_preserving_swap(sampled_state, selected)
            return sampled_state.a, sampled_state.b

        first_pair = (
            torch_gradient_kick(
                prior[0], prior[1], profile, steps=4, candidate_limit=32
            )
            if method == "kick_portfolio_late"
            else sampled_pair(seed + 4_000_000)
        )
        second_pair = sampled_pair(seed + 1_000_000)
        torch_seconds = perf_counter() - kick_started

        gradient_runner = SearchRunner.from_candidate(
            first_pair[0], first_pair[1], seed + 2_000_000, parameters
        )
        gradient_runner.state.best_a, gradient_runner.state.best_b = prior[0], prior[1]
        gradient_runner.state.best_score = prior[2]
        split_deadline = started + 0.825 * seconds
        progression.extend(run_sa_until(gradient_runner, split_deadline, started))
        gradient_best = (
            gradient_runner.state.best_a, gradient_runner.state.best_b,
            gradient_runner.state.best_score,
        )
        iteration_offset += gradient_runner.state.iteration

        runner = SearchRunner.from_candidate(
            second_pair[0], second_pair[1], seed + 3_000_000, parameters
        )
        portfolio_best = min((prior, gradient_best), key=lambda item: item[2])
        runner.state.best_a, runner.state.best_b, runner.state.best_score = portfolio_best
        progression.extend(run_sa_until(runner, deadline, started))
    else:
        raise ValueError("unknown method")

    elapsed = perf_counter() - started
    verification = verify_pqcp(runner.state.best_a, runner.state.best_b)
    return {
        "L": length, "seed": seed, "method": method,
        "target_k": profile.k, "target_eta": profile.eta,
        "initial_score": initial_score,
        "best_score": runner.state.best_score,
        "verified": verification.is_valid,
        "elapsed": elapsed, "iterations": (
            runner.state.iteration + iteration_offset
            if reported_iterations is None else reported_iterations
        ),
        "torch_seconds": torch_seconds,
        "torch_initial_score": torch_initial_score,
        "torch_best_score": torch_best_score,
        "progression": progression,
    }


def summarize(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    groups = {}
    for row in rows:
        if "best_score" in row:
            groups.setdefault((row["L"], row["method"]), []).append(row)
    result = []
    for (length, method), values in sorted(groups.items()):
        scores = [int(row["best_score"]) for row in values]
        result.append({
            "L": length, "method": method, "runs": len(values),
            "min_score": min(scores), "median_score": median(scores),
            "mean_score": mean(scores), "max_score": max(scores),
            "verified": sum(bool(row["verified"]) for row in values),
            "mean_iterations": mean(int(row["iterations"]) for row in values),
            "mean_torch_seconds": mean(float(row["torch_seconds"]) for row in values),
        })
    return result


def paired(rows: Sequence[Dict[str, object]], challenger: str) -> Dict[str, int]:
    lookup = {(row["L"], row["seed"], row["method"]): row for row in rows if "best_score" in row}
    wins = ties = losses = 0
    for (length, seed, method), baseline in lookup.items():
        if method != "sa":
            continue
        other = lookup[(length, seed, challenger)]
        left, right = int(baseline["best_score"]), int(other["best_score"])
        wins += right < left
        ties += right == left
        losses += right > left
    return {"torch_wins": wins, "ties": ties, "sa_wins": losses}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", type=int, nargs="+", default=[44, 46])
    parser.add_argument("--seeds", type=int, nargs="+", default=[123, 456, 789, 1024, 2026])
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--torch-steps", type=int, default=100)
    parser.add_argument("--torch-batch-size", type=int, default=8)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--output", type=Path, default=Path("results/benchmark_torch_pipeline.json"))
    args = parser.parse_args()
    if args.seconds <= 0:
        raise ValueError("seconds must be positive")
    rows = [
        run_wall_case(length, seed, args.seconds, method, args.torch_steps, args.torch_batch_size)
        for length in args.lengths for seed in args.seeds for method in args.methods
    ]
    payload = {
        "seconds_per_run": args.seconds, "torch_steps": args.torch_steps,
        "torch_batch_size": args.torch_batch_size,
        "summary": summarize(rows),
        "paired": {
            method: paired(rows, method) for method in args.methods
            if method != "sa" and "sa" in args.methods
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": payload["summary"], "paired": payload["paired"]}, indent=2))
    print("saved {}".format(args.output))


if __name__ == "__main__":
    main()
