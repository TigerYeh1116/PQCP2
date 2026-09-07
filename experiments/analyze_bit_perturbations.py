"""Measure how known PQCP profiles change under one, two, and three bit flips."""

import argparse
import json
from pathlib import Path
import random
import re
import statistics
import sys
from typing import Dict, Iterable, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from solver.compression import CorrelationState
from solver.verifier import verify_pqcp
from solver.z3_guidance import correlation_hamming_lower_bound


def load_verified_pairs(path: Path, length: int):
    """Load only independently verified A/B records from an existing L.txt."""
    records = re.findall(r"^a=([01]+)\nb=([01]+)$", path.read_text(encoding="utf-8"), re.MULTILINE)
    pairs = []
    for raw_a, raw_b in records:
        a, b = tuple(map(int, raw_a)), tuple(map(int, raw_b))
        if len(a) != length or len(b) != length:
            continue
        if not verify_pqcp(a, b).is_valid:
            raise RuntimeError("{} contains a verifier-failing record".format(path))
        pairs.append((a, b))
    if not pairs:
        raise ValueError("no verified length-{} pair found in {}".format(length, path))
    return tuple(pairs)


def perturbation_rows(
    pairs,
    radius: int,
    samples_per_pair: int,
    seed: int,
):
    """Return exact score/lower-bound measurements for deterministic flips."""
    if radius <= 0 or samples_per_pair <= 0:
        raise ValueError("radius and samples_per_pair must be positive")
    rng = random.Random(seed)
    rows = []
    length = len(pairs[0][0])
    for pair_index, (a, b) in enumerate(pairs):
        state = CorrelationState(a, b)
        if radius == 1:
            flip_sets = [(index,) for index in range(2 * length)]
        else:
            flip_sets = [tuple(rng.sample(range(2 * length), radius)) for _ in range(samples_per_pair)]
        for flips in flip_sets:
            for index in flips:
                state.flip_a(index) if index < length else state.flip_b(index - length)
            rows.append({
                "pair_index": pair_index,
                "radius": radius,
                "score": state.score,
                "correlation_lower_bound": correlation_hamming_lower_bound(state.profile),
            })
            for index in reversed(flips):
                state.flip_a(index) if index < length else state.flip_b(index - length)
    return rows


def summarize(
    rows: Sequence[Dict[str, int]],
    score_trigger: int = 8,
    guided_score_trigger: int = 170,
    guided_profile_radius: int = 4,
) -> Dict[str, object]:
    """Summarize score distribution and trigger recall on known-near states."""
    scores = [row["score"] for row in rows]
    bounds = [row["correlation_lower_bound"] for row in rows]
    radius = rows[0]["radius"]
    if any(bound > radius for bound in bounds):
        raise AssertionError("correlation lower bound exceeded a known Hamming upper bound")
    deciles = statistics.quantiles(scores, n=10, method="inclusive") if len(scores) > 1 else scores * 9
    return {
        "radius": radius,
        "count": len(rows),
        "score_min": min(scores),
        "score_p10": deciles[0],
        "score_median": statistics.median(scores),
        "score_p90": deciles[8],
        "score_max": max(scores),
        "score_trigger": score_trigger,
        "score_trigger_hits": sum(score <= score_trigger for score in scores),
        "score_trigger_recall": sum(score <= score_trigger for score in scores) / len(scores),
        "guided_trigger_score": guided_score_trigger,
        "guided_profile_radius": guided_profile_radius,
        "guided_trigger_hits": sum(
            score <= guided_score_trigger and bound <= guided_profile_radius
            for score, bound in zip(scores, bounds)
        ),
        "guided_trigger_recall": sum(
            score <= guided_score_trigger and bound <= guided_profile_radius
            for score, bound in zip(scores, bounds)
        ) / len(scores),
        "lower_bound_safe": True,
        "lower_bound_counts": {
            str(bound): bounds.count(bound) for bound in sorted(set(bounds))
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--L", type=int, default=44)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--samples-per-pair", type=int, default=400)
    parser.add_argument("--seed", type=int, default=90210)
    parser.add_argument("--output", type=Path, default=Path("results/bit_perturbation_analysis.json"))
    args = parser.parse_args()
    source = args.source or Path("{}.txt".format(args.L))
    pairs = load_verified_pairs(source, args.L)
    summaries = []
    for radius in (1, 2, 3):
        rows = perturbation_rows(
            pairs, radius, args.samples_per_pair,
            args.seed + radius,
        )
        summaries.append(summarize(rows))
    payload = {
        "L": args.L,
        "verified_centers": len(pairs),
        "source": str(source),
        "summaries": summaries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("Known-PQCP bit perturbation analysis")
    print("L={} verified centers={}".format(args.L, len(pairs)))
    for item in summaries:
        print(
            "radius={} n={} score[min/p10/median/p90/max]={}/{}/{}/{}/{} "
            "score<=8 recall={:.4%} guided recall={:.4%} lower-bound-safe={}".format(
                item["radius"], item["count"], item["score_min"], item["score_p10"],
                item["score_median"], item["score_p90"], item["score_max"],
                item["score_trigger_recall"], item["guided_trigger_recall"],
                item["lower_bound_safe"],
            )
        )


if __name__ == "__main__":
    main()
