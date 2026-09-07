"""Paired ablation of compressed navigation energy and FKM phase selection.

This experiment changes no solver definition.  Every run uses the existing
exact target-content constraints, same-parity swaps, correlation state,
Project objective, and independent verifier.  It compares four navigation
policies:

``full``
    The production ``fixed_target_full`` energy and its optimized hot path.
``compressed_tiebreak``
    Metropolis acceptance still uses only full energy.  Exact E2 (and E4 when
    available) is consulted only to break equal-full-energy proposal ties.
``full_e2``
    Full + E2, matching ``fixed_target_full_e2``.
``multiscale``
    Full + 2 E2, plus 4 E4 when four divides L.

The seed policies begin from the same four FKM parity necklaces.  ``legacy``
retains the historical bounded-pool selection; ``phase_random`` changes the
odd-versus-even relative phase of each sequence; ``phase_top_q`` samples such
phase pairs, ranks them lexicographically by complete energy followed by exact
compressed components, and deterministically chooses among the best Q.  Seed
selection has domain-separated RNG streams and therefore does not consume the
SA move RNG.  Its cost is inside every reported wall-clock measurement.
"""

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
from statistics import mean, median
import sys
from time import perf_counter
from typing import Dict, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from solver.checkpoint import SearchParameters, atomic_write_json
from solver.correlation import full_correlation_profile
from solver.objective import pqcp_objective
from solver.search_runner import SearchRunner
from solver.structured_energy import structured_energy_breakdown
from solver.target_profiles import (
    TargetContentProfile,
    canonical_target_content_profiles,
)
from solver.verifier import verify_pqcp


ENERGY_MODES = ("full", "compressed_tiebreak", "full_e2", "multiscale")
SEED_POLICIES = ("legacy", "phase_random", "phase_top_q")
THRESHOLDS = (32, 16, 12, 8, 4, 0)


@dataclass(frozen=True)
class BenchmarkConfig:
    """Validated factorial benchmark controls."""

    lengths: Tuple[int, ...]
    seeds: Tuple[int, ...]
    iterations: Optional[int] = None
    seconds_per_run: Optional[float] = None
    proposal_samples: int = 10
    candidate_count: int = 16
    elite_count: int = 4
    energy_modes: Tuple[str, ...] = ENERGY_MODES
    seed_policies: Tuple[str, ...] = SEED_POLICIES
    fkm_pool_size: int = 128
    stagnation_iterations: Optional[int] = 100_000

    def __post_init__(self) -> None:
        if not self.lengths or any(
            not isinstance(value, int) or isinstance(value, bool)
            or value < 4 or value % 2
            for value in self.lengths
        ):
            raise ValueError("lengths must be nonempty even integers at least four")
        if len(set(self.lengths)) != len(self.lengths):
            raise ValueError("lengths must not contain duplicates")
        if not self.seeds or any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in self.seeds
        ):
            raise ValueError("seeds must be nonempty integers")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("paired benchmark seeds must be unique")
        if (self.iterations is None) == (self.seconds_per_run is None):
            raise ValueError("specify exactly one of iterations or seconds_per_run")
        if self.iterations is not None and (
            not isinstance(self.iterations, int)
            or isinstance(self.iterations, bool)
            or self.iterations <= 0
        ):
            raise ValueError("iterations must be a positive integer")
        if self.seconds_per_run is not None and (
            not math.isfinite(self.seconds_per_run) or self.seconds_per_run <= 0
        ):
            raise ValueError("seconds_per_run must be a finite positive number")
        for name, value in (
            ("proposal_samples", self.proposal_samples),
            ("candidate_count", self.candidate_count),
            ("elite_count", self.elite_count),
            ("fkm_pool_size", self.fkm_pool_size),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError("{} must be a positive integer".format(name))
        if self.elite_count > self.candidate_count:
            raise ValueError("elite_count must not exceed candidate_count")
        if self.stagnation_iterations is not None and (
            not isinstance(self.stagnation_iterations, int)
            or isinstance(self.stagnation_iterations, bool)
            or self.stagnation_iterations <= 0
        ):
            raise ValueError(
                "stagnation_iterations must be a positive integer or None"
            )
        if (
            not self.energy_modes
            or len(set(self.energy_modes)) != len(self.energy_modes)
            or any(mode not in ENERGY_MODES for mode in self.energy_modes)
        ):
            raise ValueError("energy_modes must be unique supported modes")
        if (
            not self.seed_policies
            or len(set(self.seed_policies)) != len(self.seed_policies)
            or any(policy not in SEED_POLICIES for policy in self.seed_policies)
        ):
            raise ValueError("seed_policies must be unique supported policies")


def encoded_profiles(length: int) -> Tuple[Tuple[int, int, int, int, int, int], ...]:
    """Return the canonical production target/content cases."""
    return tuple(
        (
            profile.k, profile.eta,
            profile.a_even_ones, profile.a_odd_ones,
            profile.b_even_ones, profile.b_odd_ones,
        )
        for profile in canonical_target_content_profiles(length)
    )


def parameters_for(
    length: int,
    energy_mode: str,
    seed_policy: str,
    proposal_samples: int,
    candidate_count: int,
    elite_count: int,
    fkm_pool_size: int,
    target_profile_index: int = 0,
    stagnation_iterations: Optional[int] = 100_000,
) -> SearchParameters:
    """Build production controls for one explicit factorial benchmark cell."""
    profiles = encoded_profiles(length)
    if not profiles:
        raise ValueError("L={} has no canonical target/content profile".format(length))
    if (
        not isinstance(target_profile_index, int)
        or isinstance(target_profile_index, bool)
    ):
        raise ValueError("target_profile_index must be an integer")
    offset = target_profile_index % len(profiles)
    profiles = profiles[offset:] + profiles[:offset]
    acceptance = {
        "full": "fixed_target_full",
        "compressed_tiebreak": "fixed_target_full_compressed_tiebreak",
        "full_e2": "fixed_target_full_e2",
        "multiscale": "fixed_target_multiscale",
    }[energy_mode]
    return SearchParameters(
        initial_temperature=8.0,
        fkm_pool_size=fkm_pool_size,
        fkm_seed_policy=seed_policy,
        fkm_candidate_count=candidate_count,
        fkm_elite_count=elite_count,
        stagnation_iterations=stagnation_iterations,
        target_content_profiles=profiles,
        preserve_alternating_content=True,
        acceptance_mode=acceptance,
        proposal_samples=proposal_samples,
    )


def initialize_runner(
    length: int,
    seed: int,
    energy_mode: str,
    seed_policy: str,
    proposal_samples: int,
    candidate_count: int,
    elite_count: int,
    fkm_pool_size: int,
    target_profile_index: int = 0,
    stagnation_iterations: Optional[int] = 100_000,
) -> SearchRunner:
    """Initialize through the exact production API, including selector cost."""
    parameters = parameters_for(
        length,
        energy_mode,
        seed_policy,
        proposal_samples,
        candidate_count,
        elite_count,
        fkm_pool_size,
        target_profile_index,
        stagnation_iterations,
    )
    return SearchRunner.new(length, seed, parameters)


def energy_key(
    profile: Sequence[int], target: TargetContentProfile, mode: str,
) -> Tuple[int, ...]:
    """Return the exact navigation key used by one named method."""
    factors = (2, 4) if len(profile) % 4 == 0 else (2,)
    breakdown = structured_energy_breakdown(
        profile, target.k, target.eta, factors
    )
    full = breakdown.full
    if mode == "full":
        return (full,)
    e2 = breakdown.component(2)
    if mode == "compressed_tiebreak":
        return (full,) + tuple(
            breakdown.component(factor) for factor in factors
        )
    if mode == "full_e2":
        return (full + e2,)
    total = full + 2 * e2
    if len(profile) % 4 == 0:
        total += 4 * breakdown.component(4)
    return (total,)


def _blank_thresholds() -> Dict[str, Dict[str, object]]:
    return {
        str(threshold): {
            "hit": False, "time": None,
            "iteration": None, "evaluations": None,
        }
        for threshold in THRESHOLDS
    }


def _record_thresholds(
    thresholds: Dict[str, Dict[str, object]],
    score: int,
    elapsed: float,
    iteration: int,
    proposal_samples: int,
) -> None:
    for threshold in THRESHOLDS:
        entry = thresholds[str(threshold)]
        if not entry["hit"] and score <= threshold:
            entry.update({
                "hit": True,
                "time": elapsed,
                "iteration": iteration,
                "evaluations": iteration * proposal_samples,
            })


def run_case(
    config: BenchmarkConfig,
    length: int,
    seed: int,
    energy_mode: str,
    seed_policy: str,
    target_profile_index: int = 0,
) -> Dict[str, object]:
    """Run one exact independently checked factorial cell."""
    started = perf_counter()
    runner = initialize_runner(
        length, seed, energy_mode, seed_policy,
        config.proposal_samples, config.candidate_count,
        config.elite_count, config.fkm_pool_size,
        target_profile_index, config.stagnation_iterations,
    )
    initialization_elapsed = perf_counter() - started
    target = runner._current_target_content_profile()
    if target is None:  # pragma: no cover - guarded initialization
        raise RuntimeError("initialized runner has no target profile")
    initial_profile = tuple(full_correlation_profile(
        runner.state.current_a, runner.state.current_b
    ))
    initial_a = runner.state.current_a
    initial_b = runner.state.current_b
    initial_score = pqcp_objective(initial_profile)
    if initial_score != runner.state.current_score:
        raise RuntimeError("initial score disagrees with full recomputation")
    thresholds = _blank_thresholds()
    _record_thresholds(
        thresholds, initial_score, initialization_elapsed, 0,
        config.proposal_samples,
    )
    improvements = [{
        "time": initialization_elapsed,
        "iteration": 0,
        "score": initial_score,
    }]
    initial_verification = verify_pqcp(
        runner.state.current_a, runner.state.current_b
    )
    if initial_score == 0 and not initial_verification.is_valid:
        raise RuntimeError("score-zero initial candidate failed independent verifier")
    verified = initial_verification.is_valid
    while not verified:
        elapsed = perf_counter() - started
        if config.iterations is not None:
            if runner.state.iteration >= config.iterations:
                break
        elif elapsed >= config.seconds_per_run:
            break
        old_best = runner.state.best_score
        outcome = runner.step()
        elapsed = perf_counter() - started
        if runner.state.best_score < old_best:
            improvements.append({
                "time": elapsed,
                "iteration": runner.state.iteration,
                "score": runner.state.best_score,
            })
            _record_thresholds(
                thresholds, runner.state.best_score, elapsed,
                runner.state.iteration, config.proposal_samples,
            )
        if outcome.verified_solution is not None:
            verified = verify_pqcp(*outcome.verified_solution).is_valid
            if not verified:
                raise RuntimeError("reported solution failed independent verifier")
            break

    elapsed = perf_counter() - started
    best_profile = tuple(full_correlation_profile(
        runner.state.best_a, runner.state.best_b
    ))
    best_score = pqcp_objective(best_profile)
    if best_score != runner.state.best_score:
        raise RuntimeError("final best score disagrees with full recomputation")
    final_verification = verify_pqcp(
        runner.state.best_a, runner.state.best_b
    )
    if best_score == 0 and not final_verification.is_valid:
        raise RuntimeError("score-zero final best failed independent verifier")
    verified = verified or final_verification.is_valid
    evaluations = runner.state.iteration * config.proposal_samples
    initial_breakdown = structured_energy_breakdown(
        initial_profile, target.k, target.eta,
        (2, 4) if length % 4 == 0 else (2,),
    )
    return {
        "L": length,
        "seed": seed,
        "energy_mode": energy_mode,
        "seed_policy": seed_policy,
        "budget_type": "iterations" if config.iterations is not None else "seconds",
        "budget": config.iterations if config.iterations is not None else config.seconds_per_run,
        "proposal_samples": config.proposal_samples,
        "candidate_count": config.candidate_count,
        "elite_count": config.elite_count,
        "target_profile_index": target_profile_index,
        "target_k": target.k,
        "target_eta": target.eta,
        "target_content": [
            target.a_even_ones, target.a_odd_ones,
            target.b_even_ones, target.b_odd_ones,
        ],
        "initialization_elapsed": initialization_elapsed,
        "initial_score": initial_score,
        "initial_full_energy": initial_breakdown.full,
        "initial_energy_key": list(energy_key(initial_profile, target, energy_mode)),
        # Persist both sides so paired runs can be audited after the fact;
        # these are captured before the first SA step.
        "initial_A": list(initial_a),
        "initial_B": list(initial_b),
        "initial_profile": list(initial_profile),
        "best_score": best_score,
        "best_A": list(runner.state.best_a),
        "best_B": list(runner.state.best_b),
        "best_profile": list(best_profile),
        "elapsed": elapsed,
        "iterations": runner.state.iteration,
        # SearchRunner samples with replacement and caches duplicate proposals,
        # so this is an upper bound on actual unique energy computations.
        "sampled_proposals": evaluations,
        "sampled_proposals_per_second": evaluations / elapsed if elapsed else 0.0,
        # Backward-compatible aliases for older aggregation consumers.
        "proposal_evaluations": evaluations,
        "evaluations_per_second": evaluations / elapsed if elapsed else 0.0,
        "restarts": runner.state.restart_index,
        "thresholds": thresholds,
        "improvements": improvements,
        "verified": verified,
    }


def paired_wtl(
    records: Sequence[Dict[str, object]],
    reference_energy: str,
    reference_policy: str,
    energy_mode: str,
    seed_policy: str,
    metric: str = "best_score",
) -> Dict[str, int]:
    """Compare one lower-is-better metric on common (L, seed) cells."""
    reference = {
        (int(row["L"]), int(row["seed"])): float(row[metric])
        for row in records
        if row["energy_mode"] == reference_energy
        and row["seed_policy"] == reference_policy
    }
    candidate = {
        (int(row["L"]), int(row["seed"])): float(row[metric])
        for row in records
        if row["energy_mode"] == energy_mode
        and row["seed_policy"] == seed_policy
    }
    keys = sorted(set(reference) & set(candidate))
    wins = sum(candidate[key] < reference[key] for key in keys)
    ties = sum(candidate[key] == reference[key] for key in keys)
    losses = sum(candidate[key] > reference[key] for key in keys)
    return {"wins": wins, "ties": ties, "losses": losses, "pairs": len(keys)}


def paired_initialization_audit(
    records: Sequence[Dict[str, object]],
) -> Dict[str, int]:
    """Prove every energy mode saw identical A/B for each L/seed/policy."""
    signatures: Dict[Tuple[int, int, str], Tuple[Tuple[int, ...], Tuple[int, ...]]] = {}
    comparisons = 0
    for row in records:
        key = (int(row["L"]), int(row["seed"]), str(row["seed_policy"]))
        signature = (tuple(row["initial_A"]), tuple(row["initial_B"]))
        previous = signatures.setdefault(key, signature)
        if previous != signature:
            raise AssertionError(
                "paired energy modes received different initial A/B for {}".format(key)
            )
        comparisons += 1
    return {"groups": len(signatures), "records_checked": comparisons, "mismatches": 0}


def threshold_summary(
    records: Sequence[Dict[str, object]], threshold: int,
) -> Dict[str, object]:
    """Aggregate hit time and conservative restricted means including misses."""
    if not records:
        raise ValueError("threshold summary requires records")
    entries = [row["thresholds"][str(threshold)] for row in records]
    hit_times = [float(entry["time"]) for entry in entries if entry["hit"]]
    hit_evaluations = [
        int(entry["evaluations"]) for entry in entries if entry["hit"]
    ]
    censored_times = [
        float(entry["time"]) if entry["hit"] else float(row["elapsed"])
        for row, entry in zip(records, entries)
    ]
    censored_evaluations = [
        int(entry["evaluations"])
        if entry["hit"] else int(row["proposal_evaluations"])
        for row, entry in zip(records, entries)
    ]
    return {
        "hits": len(hit_times),
        "runs": len(records),
        "hit_rate": len(hit_times) / len(records),
        "median_time_when_hit": median(hit_times) if hit_times else None,
        "median_evaluations_when_hit": (
            median(hit_evaluations) if hit_evaluations else None
        ),
        "restricted_mean_time": mean(censored_times),
        "restricted_mean_evaluations": mean(censored_evaluations),
    }


def summarize(records: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    """Aggregate factorial cells with paired energy and selector comparisons."""
    groups: Dict[Tuple[int, str, str], List[Dict[str, object]]] = {}
    for row in records:
        groups.setdefault(
            (int(row["L"]), str(row["energy_mode"]), str(row["seed_policy"])),
            [],
        ).append(row)
    output = []
    for (length, energy_mode, seed_policy), rows in sorted(groups.items()):
        scores = [int(row["best_score"]) for row in rows]
        same_length = [row for row in records if int(row["L"]) == length]
        output.append({
            "L": length,
            "energy_mode": energy_mode,
            "seed_policy": seed_policy,
            "runs": len(rows),
            "minimum": min(scores),
            "median": median(scores),
            "mean": mean(scores),
            "maximum": max(scores),
            "mean_initial_score": mean(int(row["initial_score"]) for row in rows),
            "median_initial_score": median(
                int(row["initial_score"]) for row in rows
            ),
            "mean_initial_full_energy": mean(
                int(row["initial_full_energy"]) for row in rows
            ),
            "median_initial_full_energy": median(
                int(row["initial_full_energy"]) for row in rows
            ),
            "mean_initialization_elapsed": mean(
                float(row["initialization_elapsed"]) for row in rows
            ),
            "mean_iterations": mean(int(row["iterations"]) for row in rows),
            "mean_evaluations_per_second": mean(
                float(row["evaluations_per_second"]) for row in rows
            ),
            "verified": sum(bool(row["verified"]) for row in rows),
            "paired_energy_vs_full": paired_wtl(
                same_length, "full", seed_policy, energy_mode, seed_policy
            ),
            "paired_policy_vs_legacy": paired_wtl(
                same_length, energy_mode, "legacy", energy_mode, seed_policy
            ),
            "paired_initial_energy_vs_legacy": paired_wtl(
                same_length, energy_mode, "legacy", energy_mode, seed_policy,
                metric="initial_full_energy",
            ),
            "thresholds": {
                str(threshold): threshold_summary(rows, threshold)
                for threshold in THRESHOLDS
            },
        })
    return output


def counterbalanced_order(
    cells: Sequence[Tuple[str, str]], design_index: int,
) -> Tuple[Tuple[str, str], ...]:
    """Return a deterministic forward/reverse rotated benchmark order.

    Consecutive design rows use opposite traversal directions at the same
    rotation.  Across ``2 * len(cells)`` rows, every cell therefore occupies
    every serial position twice, reducing systematic warm-up and thermal-order
    bias without changing any search RNG.
    """
    values = tuple(cells)
    if not values:
        raise ValueError("counterbalance requires at least one cell")
    if not isinstance(design_index, int) or isinstance(design_index, bool):
        raise ValueError("design_index must be an integer")
    rotation = (design_index // 2) % len(values)
    oriented = values if design_index % 2 == 0 else tuple(reversed(values))
    return oriented[rotation:] + oriented[:rotation]


def benchmark(config: BenchmarkConfig) -> Dict[str, object]:
    """Run a counterbalanced paired factorial experiment."""
    cells = [
        (energy_mode, seed_policy)
        for energy_mode in config.energy_modes
        for seed_policy in config.seed_policies
    ]
    records = []
    for length_index, length in enumerate(config.lengths):
        # Validate once before beginning potentially expensive cells.
        if not encoded_profiles(length):
            raise ValueError("L={} has no canonical target/content profile".format(length))
        for seed_index, seed in enumerate(config.seeds):
            order = counterbalanced_order(
                cells, seed_index + length_index * len(config.seeds)
            )
            target_profile_index = seed_index % len(encoded_profiles(length))
            for energy_mode, seed_policy in order:
                record = run_case(
                    config, length, seed, energy_mode, seed_policy,
                    target_profile_index=target_profile_index,
                )
                records.append(record)
                print(
                    "L={} seed={} energy={} policy={} score={} iterations={} "
                    "elapsed={:.3f}s".format(
                        length, seed, energy_mode, seed_policy,
                        record["best_score"], record["iterations"],
                        record["elapsed"],
                    ),
                    flush=True,
                )
    initialization_audit = paired_initialization_audit(records)
    return {
        "config": {
            "lengths": list(config.lengths),
            "seeds": list(config.seeds),
            "iterations": config.iterations,
            "seconds_per_run": config.seconds_per_run,
            "proposal_samples": config.proposal_samples,
            "candidate_count": config.candidate_count,
            "elite_count": config.elite_count,
            "fkm_pool_size": config.fkm_pool_size,
            "stagnation_iterations": config.stagnation_iterations,
            "energy_modes": list(config.energy_modes),
            "seed_policies": list(config.seed_policies),
        },
        "records": records,
        "paired_initialization_audit": initialization_audit,
        "summary": summarize(records),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--L", type=int, nargs="+", default=(28, 44, 46))
    parser.add_argument("--seeds", type=int, nargs="+", default=(123, 456, 789, 1024, 2026))
    budget = parser.add_mutually_exclusive_group(required=True)
    budget.add_argument("--iterations", type=int)
    budget.add_argument("--seconds-per-run", type=float)
    parser.add_argument("--proposal-samples", type=int, default=10)
    parser.add_argument("--candidate-count", type=int, default=16)
    parser.add_argument("--elite-count", type=int, default=4)
    parser.add_argument("--fkm-pool-size", type=int, default=128)
    parser.add_argument("--stagnation-iterations", type=int, default=100_000)
    parser.add_argument("--energy-modes", nargs="+", choices=ENERGY_MODES, default=ENERGY_MODES)
    parser.add_argument("--seed-policies", nargs="+", choices=SEED_POLICIES, default=SEED_POLICIES)
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/benchmark_compressed_search.json"),
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    config = BenchmarkConfig(
        lengths=tuple(args.L), seeds=tuple(args.seeds),
        iterations=args.iterations, seconds_per_run=args.seconds_per_run,
        proposal_samples=args.proposal_samples,
        candidate_count=args.candidate_count, elite_count=args.elite_count,
        fkm_pool_size=args.fkm_pool_size,
        stagnation_iterations=args.stagnation_iterations,
        energy_modes=tuple(args.energy_modes),
        seed_policies=tuple(args.seed_policies),
    )
    payload = benchmark(config)
    atomic_write_json(args.output, payload)
    print("Compressed search benchmark summary")
    for row in payload["summary"]:
        print(
            "L={L} {energy_mode}/{seed_policy}: min={minimum} median={median} "
            "mean={mean:.2f} max={maximum} verified={verified}/{runs} "
            "rate={mean_evaluations_per_second:.0f}/s".format(**row)
        )
    print("saved {}".format(args.output))


if __name__ == "__main__":
    main()
