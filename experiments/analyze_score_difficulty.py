"""Validate whether PQCP scores track exact and empirical repair difficulty."""

import argparse
from collections import defaultdict
from itertools import combinations
import json
from math import sqrt
from pathlib import Path
import random
import re
import statistics
import sys
from typing import Dict, Iterable, List, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from solver.correlation import full_correlation_profile
from solver.difficulty import lookahead_repair_difficulty
from solver.objective import (
    pqcp_objective,
    target_pair_squared_energy,
    target_profile_l1_distance,
)
from solver.verifier import verify_pqcp
from solver.weight_constraints import admissible_weight_pairs
from solver.z3_guidance import correlation_hamming_lower_bound


Pair = Tuple[Tuple[int, ...], Tuple[int, ...]]


def spearman_rank_correlation(left: Sequence[float], right: Sequence[float]) -> float:
    """Return Spearman correlation with average ranks for ties."""
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("correlation inputs must have equal length at least two")
    rank_left, rank_right = _average_ranks(left), _average_ranks(right)
    mean_left = sum(rank_left) / len(rank_left)
    mean_right = sum(rank_right) / len(rank_right)
    covariance = sum(
        (x - mean_left) * (y - mean_right) for x, y in zip(rank_left, rank_right)
    )
    variance_left = sum((x - mean_left) ** 2 for x in rank_left)
    variance_right = sum((y - mean_right) ** 2 for y in rank_right)
    if variance_left == 0 or variance_right == 0:
        return 0.0
    return covariance / sqrt(variance_left * variance_right)


def exhaustive_length(length: int) -> Dict[str, object]:
    """Compare profile scores with exact nearest-solution swap distance."""
    rows = []
    solution_count = 0
    for weights in admissible_weight_pairs(length):
        sequences_a = tuple(_sequences(length, weights[0]))
        sequences_b = tuple(_sequences(length, weights[1]))
        metrics = {}
        solutions = []
        for a in sequences_a:
            for b in sequences_b:
                profile = tuple(full_correlation_profile(a, b))
                old = pqcp_objective(profile)
                metrics[(a, b)] = {
                    "old": old,
                    "target_l1": target_profile_l1_distance(profile),
                    "target_l2": target_pair_squared_energy(profile),
                    "profile_lower_bound": correlation_hamming_lower_bound(profile),
                }
                if old == 0:
                    if not verify_pqcp(a, b).is_valid:
                        raise RuntimeError("score-zero exhaustive state failed verifier")
                    solutions.append((a, b))
        if not solutions:
            continue
        solution_count += len(solutions)
        for pair, values in metrics.items():
            exact_distance = min(_swap_distance(pair, solution) for solution in solutions)
            values = dict(values)
            values["exact_swap_distance"] = exact_distance
            values["lookahead"] = _cached_lookahead(pair, metrics, max_depth=2)
            rows.append(values)
    correlations = {
        name: spearman_rank_correlation(
            [row[name] for row in rows], [row["exact_swap_distance"] for row in rows]
        )
        for name in ("old", "target_l1", "target_l2", "profile_lower_bound", "lookahead")
    }
    return {
        "L": length,
        "states": len(rows),
        "solutions": solution_count,
        "maximum_exact_swap_distance": max(row["exact_swap_distance"] for row in rows),
        "spearman_vs_exact_swap_distance": correlations,
    }


def l44_discrimination(root: Path, samples: int, seed: int) -> Dict[str, object]:
    """Contrast known-near perturbations with genuine persisted low-score elites."""
    pairs = _load_verified_pairs(root / "44.txt", 44)
    near_rows = []
    for swaps in (1, 2):
        for trial in range(samples):
            pair = _perturb_by_swaps(
                pairs[(trial + swaps * samples) % len(pairs)], swaps,
                random.Random(seed + swaps * 10_000 + trial),
            )
            near_rows.append(_candidate_row(pair, "known_radius_{}".format(swaps)))
    real_rows = []
    for path in sorted((root / "results" / "best").glob("L44_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        a, b = payload.get("A", payload.get("a")), payload.get("B", payload.get("b"))
        if a is not None and b is not None and len(a) == len(b) == 44:
            real_rows.append(_candidate_row((tuple(a), tuple(b)), path.name))
    return {
        "known_near": near_rows,
        "persisted_real_elites": real_rows,
        "summary": {
            "known_near_count": len(near_rows),
            "known_near_old_score_median": statistics.median(row["old"] for row in near_rows),
            "known_near_lookahead_median": statistics.median(row["lookahead"] for row in near_rows),
            "known_near_solution_within_beam": sum(row["solution_depth"] is not None for row in near_rows),
            "real_elite_count": len(real_rows),
            "real_elite_old_score_median": statistics.median((row["old"] for row in real_rows)) if real_rows else None,
            "real_elite_lookahead_median": statistics.median((row["lookahead"] for row in real_rows)) if real_rows else None,
            "real_elite_solution_within_beam": sum(row["solution_depth"] is not None for row in real_rows),
        },
    }


def controlled_project_length(root: Path, length: int, samples: int, seed: int) -> Dict[str, object]:
    """Holdout check that known one/two-swap cost is recognized at another L."""
    pairs = _load_verified_pairs(root / "{}.txt".format(length), length)
    rows = []
    for swaps in (1, 2):
        for trial in range(samples):
            pair = _perturb_by_swaps(
                pairs[(trial + swaps * samples) % len(pairs)], swaps,
                random.Random(seed + length * 100_000 + swaps * 10_000 + trial),
            )
            rows.append(_candidate_row(pair, "L{}_known_radius_{}".format(length, swaps)))
    return {
        "L": length,
        "count": len(rows),
        "old_score_median": statistics.median(row["old"] for row in rows),
        "lookahead_median": statistics.median(row["lookahead"] for row in rows),
        "solution_within_beam": sum(row["solution_depth"] is not None for row in rows),
        "rows": rows,
    }


def _candidate_row(pair: Pair, source: str) -> Dict[str, object]:
    profile = tuple(full_correlation_profile(*pair))
    lookahead = lookahead_repair_difficulty(*pair, max_depth=2, beam_width=10)
    return {
        "source": source,
        "old": pqcp_objective(profile),
        "target_l1": target_profile_l1_distance(profile),
        "lookahead": lookahead.total,
        "solution_depth": lookahead.solution_depth,
        "states_examined": lookahead.states_examined,
    }


def _cached_lookahead(pair: Pair, metrics: Dict[Pair, Dict[str, int]], max_depth: int) -> int:
    """Exhaustive small-L counterpart of the production beam metric."""
    best = metrics[pair]["target_l1"]
    frontier = {pair}
    seen = {pair}
    for depth in range(1, max_depth + 1):
        following = set()
        for current in frontier:
            following.update(neighbor for neighbor in _neighbors(current) if neighbor not in seen)
        if not following:
            break
        best = min(best, min(depth + metrics[item]["target_l1"] for item in following))
        seen.update(following)
        frontier = following
    return best


def _neighbors(pair: Pair) -> Iterable[Pair]:
    for sequence_index, sequence in enumerate(pair):
        zeros = (index for index, bit in enumerate(sequence) if bit == 0)
        ones = tuple(index for index, bit in enumerate(sequence) if bit == 1)
        for zero in zeros:
            for one in ones:
                changed = list(sequence)
                changed[zero], changed[one] = 1, 0
                yield (tuple(changed), pair[1]) if sequence_index == 0 else (pair[0], tuple(changed))


def _sequences(length: int, weight: int) -> Iterable[Tuple[int, ...]]:
    for positions in combinations(range(length), weight):
        values = [0] * length
        for position in positions:
            values[position] = 1
        yield tuple(values)


def _swap_distance(left: Pair, right: Pair) -> int:
    return (
        sum(x != y for x, y in zip(left[0], right[0]))
        + sum(x != y for x, y in zip(left[1], right[1]))
    ) // 2


def _average_ranks(values: Sequence[float]) -> List[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + end - 1) / 2.0
        for index in order[start:end]:
            ranks[index] = rank
        start = end
    return ranks


def _load_verified_pairs(path: Path, length: int) -> Tuple[Pair, ...]:
    records = re.findall(r"^a=([01]+)\nb=([01]+)$", path.read_text(encoding="utf-8"), re.MULTILINE)
    pairs = []
    for raw_a, raw_b in records:
        pair = (tuple(map(int, raw_a)), tuple(map(int, raw_b)))
        if len(pair[0]) == len(pair[1]) == length and verify_pqcp(*pair).is_valid:
            pairs.append(pair)
    if not pairs:
        raise ValueError("no verified pairs in {}".format(path))
    return tuple(pairs)


def _perturb_by_swaps(pair: Pair, swaps: int, rng: random.Random) -> Pair:
    values = [list(pair[0]), list(pair[1])]
    for _ in range(swaps):
        sequence = values[rng.randrange(2)]
        zero = rng.choice([i for i, bit in enumerate(sequence) if bit == 0])
        one = rng.choice([i for i, bit in enumerate(sequence) if bit == 1])
        sequence[zero], sequence[one] = 1, 0
    return tuple(values[0]), tuple(values[1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--output", type=Path, default=Path("results/score_difficulty_analysis.json"))
    args = parser.parse_args()
    payload = {
        "definition": "difficulty = exact fixed-weight swap distance / local repairability",
        "small_exhaustive": [exhaustive_length(length) for length in (4, 6, 8)],
        "L44": l44_discrimination(PROJECT_ROOT, args.samples, args.seed),
        "L46_holdout": controlled_project_length(PROJECT_ROOT, 46, args.samples, args.seed),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for row in payload["small_exhaustive"]:
        print("L={} states={} correlations={}".format(row["L"], row["states"], row["spearman_vs_exact_swap_distance"]))
    print("L44 {}".format(payload["L44"]["summary"]))
    print("L46 holdout {}".format({key: value for key, value in payload["L46_holdout"].items() if key != "rows"}))
    print("saved {}".format(args.output))


if __name__ == "__main__":
    main()
