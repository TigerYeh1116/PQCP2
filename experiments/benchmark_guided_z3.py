"""Paired completion benchmark on withheld perturbations of verified PQCPs."""

import argparse
import json
from pathlib import Path
import re
import statistics
import sys
from time import perf_counter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from solver.compression import CorrelationState
from solver.moves import (
    WeightPreservingSwap,
    apply_weight_preserving_swap,
    rollback_weight_preserving_swap,
    trial_weight_preserving_swap,
)
from solver.verifier import verify_pqcp
from solver.z3_guidance import guided_completion
from solver.z3_solver import solve_with_z3


def eligible_centers(path: Path, length: int, trigger_score: int = 8):
    """Build all fixed-weight radius-two centers accepted by the old trigger."""
    records = re.findall(r"^a=([01]+)\nb=([01]+)$", path.read_text(encoding="utf-8"), re.MULTILINE)
    primary_centers = []
    additional_centers = []
    seen = set()
    for raw_a, raw_b in records:
        a, b = tuple(map(int, raw_a)), tuple(map(int, raw_b))
        if len(a) != length or not verify_pqcp(a, b).is_valid:
            continue
        state = CorrelationState(a, b)
        source_centers = []
        for name, bits in (("a", state.a), ("b", state.b)):
            zeros = [index for index, bit in enumerate(bits) if bit == 0]
            ones = [index for index, bit in enumerate(bits) if bit == 1]
            for zero in zeros:
                for one in ones:
                    move = WeightPreservingSwap(name, zero, one)
                    evaluation = trial_weight_preserving_swap(state, move)
                    if evaluation.score <= trigger_score:
                        apply_weight_preserving_swap(state, move)
                        key = (state.a, state.b)
                        if key not in seen:
                            seen.add(key)
                            source_centers.append((state.a, state.b, evaluation.score))
                        rollback_weight_preserving_swap(state, move)
        if source_centers:
            # Exercise distinct known PQCP basins before taking a second
            # perturbation from any one source solution.
            primary_centers.append(source_centers[0])
            additional_centers.extend(source_centers[1:])
    return tuple(primary_centers + additional_centers)


def run_broad(length, a, b, timeout_ms):
    """Time the previous broad radius-three Z3 operation."""
    started = perf_counter()
    result = solve_with_z3(length, a, b, radius=3, timeout_ms=timeout_ms)
    elapsed = perf_counter() - started
    if result.status != "SAT" or not result.verified:
        raise RuntimeError("broad Z3 did not verify the controlled near-solution: {}".format(result.status))
    return elapsed


def run_guided(length, a, b, timeout_ms, top_k):
    """Time guidance plus the production radius-zero Z3 confirmation."""
    started = perf_counter()
    guidance = guided_completion(a, b, top_k=top_k)
    if not guidance.solved:
        raise RuntimeError("guidance missed a controlled radius-two solution")
    result = solve_with_z3(length, guidance.a, guidance.b, radius=0, timeout_ms=timeout_ms)
    elapsed = perf_counter() - started
    if result.status != "SAT" or not result.verified:
        raise RuntimeError("guided Z3 confirmation failed: {}".format(result.status))
    return elapsed


def aggregate(records):
    """Calculate arithmetic mean time and literal speed improvement."""
    broad = [record["elapsed"] for record in records if record["method"] == "broad"]
    guided = [record["elapsed"] for record in records if record["method"] == "guided"]
    if not broad or len(broad) != len(guided):
        raise ValueError("records must contain complete broad/guided pairs")
    broad_mean, guided_mean = statistics.mean(broad), statistics.mean(guided)
    factor = broad_mean / guided_mean
    return {
        "broad_mean_seconds": broad_mean,
        "guided_mean_seconds": guided_mean,
        "speed_factor": factor,
        "speed_increase_percent": 100.0 * (factor - 1.0),
    }


def benchmark(path: Path, length: int, case_count: int, timeout_ms: int, top_k: int):
    """Interleave paired completion methods over distinct controlled centers."""
    centers = eligible_centers(path, length)
    if len(centers) < case_count:
        raise ValueError("requested {} cases but only {} are eligible".format(case_count, len(centers)))
    records = []
    for index, (a, b, score) in enumerate(centers[:case_count]):
        order = ("broad", "guided") if index % 2 == 0 else ("guided", "broad")
        for method in order:
            elapsed = (
                run_broad(length, a, b, timeout_ms)
                if method == "broad"
                else run_guided(length, a, b, timeout_ms, top_k)
            )
            records.append({"case": index, "score": score, "method": method, "elapsed": elapsed})
    return {"L": length, "case_count": case_count, "top_k": top_k, "records": records, **aggregate(records)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--L", type=int, default=44)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--cases", type=int, default=5)
    parser.add_argument("--timeout-ms", type=int, default=60_000)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("results/benchmark_guided_z3.json"))
    args = parser.parse_args()
    result = benchmark(args.source or Path("{}.txt".format(args.L)), args.L, args.cases, args.timeout_ms, args.top_k)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("Paired broad-Z3 vs correlation-guided Z3 benchmark")
    print("L={} cases={} top_k={}".format(args.L, args.cases, args.top_k))
    print("broad mean = {:.6f} sec".format(result["broad_mean_seconds"]))
    print("guided mean = {:.6f} sec".format(result["guided_mean_seconds"]))
    print("speed factor = {:.3f}x".format(result["speed_factor"]))
    print("speed increase = {:.2f}%".format(result["speed_increase_percent"]))


if __name__ == "__main__":
    main()
