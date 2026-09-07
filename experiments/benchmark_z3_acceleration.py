"""Paired end-to-end benchmark for legacy versus adaptive Z3 completion.

Each timed run starts the same deterministic FKM/fixed-weight-SA trajectory and
stops at its first independently verified PQCP.  The reference path preserves
the former score<=10 trigger, nested radius 0/1/2/3 multi-elite portfolio, and
Int-bit encoding.  The optimized path uses the production adaptive trigger,
newest elite only, one radius-three task, and Bool-flip/PB encoding.
"""

import argparse
from pathlib import Path
import statistics
import sys
from time import perf_counter
from typing import Sequence, Tuple

import z3


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from solver.checkpoint import SearchParameters
from solver.hybrid import PortfolioConfig, SAElite, run_hybrid_portfolio
from solver.search_runner import SearchRunner
from solver.verifier import verify_pqcp
from solver.z3_solver import Z3CompletionResult, solve_with_z3


DEFAULT_SEEDS = (6, 24, 43, 100, 183)


def legacy_solve_with_z3(L, initial_a, initial_b, radius, timeout_ms):
    """Reproduce the pre-optimization Int-bit bounded-neighborhood encoding."""
    started = perf_counter()
    solver = z3.Solver()
    if timeout_ms is not None:
        solver.set(timeout=timeout_ms)
    a_vars = [z3.Int("legacy_a_{}".format(index)) for index in range(L)]
    b_vars = [z3.Int("legacy_b_{}".format(index)) for index in range(L)]
    for variable in a_vars + b_vars:
        solver.add(z3.Or(variable == 0, variable == 1))
    solver.add(z3.Sum(
        [z3.If(variable != bit, 1, 0) for variable, bit in zip(a_vars, initial_a)]
        + [z3.If(variable != bit, 1, 0) for variable, bit in zip(b_vars, initial_b)]
    ) <= radius)
    solver.add(_legacy_pair_correlation(a_vars, b_vars, 0, L) == 2 * L)
    indicators = []
    for shift in range(1, L // 2 + 1):
        correlation = _legacy_pair_correlation(a_vars, b_vars, shift, L)
        solver.add(z3.Or(correlation == -4, correlation == 0, correlation == 4))
        nonzero = z3.Bool("legacy_nonzero_{}".format(shift))
        solver.add(nonzero == (correlation != 0))
        weight = 1 if L % 2 == 0 and shift == L // 2 else 2
        indicators.append(z3.If(nonzero, weight, 0))
    solver.add(z3.Sum(indicators) == 2)
    outcome = solver.check()
    elapsed = perf_counter() - started
    if outcome == z3.sat:
        model = solver.model()
        a = tuple(model.evaluate(variable).as_long() for variable in a_vars)
        b = tuple(model.evaluate(variable).as_long() for variable in b_vars)
        verification = verify_pqcp(a, b)
        return Z3CompletionResult("SAT", L, radius, elapsed, a, b, verification.profile,
                                  verification.is_valid, None)
    if outcome == z3.unsat:
        return Z3CompletionResult("UNSAT", L, radius, elapsed, None, None, None, False, None)
    return Z3CompletionResult("UNKNOWN", L, radius, elapsed, None, None, None, False,
                              solver.reason_unknown())


def run_to_first_solution(length: int, seed: int, max_steps: int, timeout_ms: int,
                          optimized: bool) -> Tuple[float, Tuple[int, ...], Tuple[int, ...]]:
    """Time identical SA plus the selected Z3 policy to its first verified pair."""
    started = perf_counter()
    runner = SearchRunner.new(length, seed, SearchParameters(stagnation_iterations=None))
    elites = []
    last_z3_score = None
    for _ in range(max_steps):
        outcome = runner.step()
        if outcome.improved_best:
            state = runner.state
            elites.append(SAElite(state.best_a, state.best_b, state.best_score, seed, len(elites)))
            threshold = 7 if optimized and length <= 20 else (4 if optimized else 10)
            if 0 < state.best_score <= threshold and (
                    not optimized or last_z3_score is None or state.best_score < last_z3_score):
                last_z3_score = state.best_score
                if optimized:
                    config = PortfolioConfig(
                        timeout_radius_0_ms=timeout_ms, timeout_radius_1_ms=timeout_ms,
                        timeout_radius_2_ms=timeout_ms, timeout_radius_3_ms=timeout_ms,
                        radius_0_elite_count=0, radius_1_elite_count=0,
                        radius_2_elite_count=0, radius_3_elite_count=1,
                        total_timeout_seconds=timeout_ms / 1000.0,
                    )
                    portfolio = run_hybrid_portfolio(
                        length, elites[-1:], 1, config, solve_function=solve_with_z3
                    )
                else:
                    count = min(5, len(elites))
                    config = PortfolioConfig(
                        timeout_radius_0_ms=timeout_ms, timeout_radius_1_ms=timeout_ms,
                        timeout_radius_2_ms=timeout_ms, timeout_radius_3_ms=timeout_ms,
                        total_timeout_seconds=(2 * count + 3) * timeout_ms / 1000.0,
                    )
                    portfolio = run_hybrid_portfolio(
                        length, elites, count, config, solve_function=legacy_solve_with_z3
                    )
                if portfolio.solved:
                    solution = portfolio.solution
                    assert solution is not None and solution.a is not None and solution.b is not None
                    if not verify_pqcp(solution.a, solution.b).is_valid:
                        raise AssertionError("portfolio returned a verifier-failing solution")
                    return perf_counter() - started, solution.a, solution.b
        if outcome.verified_solution is not None:
            a, b = outcome.verified_solution
            if not verify_pqcp(a, b).is_valid:
                raise AssertionError("SA returned a verifier-failing solution")
            return perf_counter() - started, a, b
    raise RuntimeError("no verified solution within {} SA steps for seed {}".format(max_steps, seed))


def benchmark(length: int, seeds: Sequence[int], repeats: int, max_steps: int, timeout_ms: int):
    """Return interleaved arithmetic-mean time-to-first-solution statistics."""
    if not seeds or repeats <= 0 or max_steps <= 0 or timeout_ms <= 0:
        raise ValueError("seeds, repeats, max_steps, and timeout_ms must be positive")
    legacy_times = []
    optimized_times = []
    verified_pairs = set()
    # Warm both solver stacks once before timed samples.
    run_to_first_solution(length, seeds[0], max_steps, timeout_ms, False)
    run_to_first_solution(length, seeds[0], max_steps, timeout_ms, True)
    for repeat in range(repeats):
        for seed_index, seed in enumerate(seeds):
            for optimized in ((False, True) if (repeat + seed_index) % 2 == 0 else (True, False)):
                elapsed, a, b = run_to_first_solution(length, seed, max_steps, timeout_ms, optimized)
                (optimized_times if optimized else legacy_times).append(elapsed)
                verified_pairs.add(min((a, b), (b, a)))
    legacy_mean = statistics.mean(legacy_times)
    optimized_mean = statistics.mean(optimized_times)
    return {
        "legacy_times": legacy_times,
        "optimized_times": optimized_times,
        "legacy_mean": legacy_mean,
        "optimized_mean": optimized_mean,
        "reduction_percent": 100.0 * (legacy_mean - optimized_mean) / legacy_mean,
        "verified_pair_count": len(verified_pairs),
    }


def _legacy_pair_correlation(a_vars, b_vars, shift, length):
    terms = []
    for index in range(length):
        terms.append(z3.If(a_vars[index] == a_vars[(index + shift) % length], 1, -1))
        terms.append(z3.If(b_vars[index] == b_vars[(index + shift) % length], 1, -1))
    return z3.Sum(terms)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--L", type=int, default=20)
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--timeout-ms", type=int, default=1000)
    args = parser.parse_args()
    result = benchmark(
        args.L, tuple(int(seed) for seed in args.seeds.split(",") if seed.strip()),
        args.repeats, args.max_steps, args.timeout_ms,
    )
    print("Paired Z3 hybrid time-to-first-verified-PQCP benchmark")
    print("L={} seeds={} repeats={}".format(args.L, len(args.seeds.split(",")), args.repeats))
    print("legacy mean = {:.6f} sec/solution".format(result["legacy_mean"]))
    print("optimized mean = {:.6f} sec/solution".format(result["optimized_mean"]))
    print("additional mean reduction = {:.2f}%".format(result["reduction_percent"]))
    print("independently verified distinct pairs = {}".format(result["verified_pair_count"]))


if __name__ == "__main__":
    main()
