"""Small, bounded timing benchmark for the FKM candidate generator."""

from pathlib import Path
import sys
from time import perf_counter
from typing import Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from solver.fkm import generate_fkm_sequences


CASES: Tuple[Tuple[int, Optional[int]], ...] = (
    (8, None),
    (8, 4),
    (10, None),
    (10, 5),
    (12, None),
    (12, 6),
    (16, None),
    (16, 8),
)


def benchmark_case(length: int, weight: Optional[int]) -> Tuple[int, float, float]:
    """Consume one bounded FKM stream and return count, seconds, and rate."""
    started = perf_counter()
    count = sum(1 for _ in generate_fkm_sequences(length, weight=weight))
    elapsed = perf_counter() - started
    rate = count / elapsed if elapsed else float("inf")
    return count, elapsed, rate


def main() -> None:
    """Run the documented small-length FKM benchmark cases."""
    print("FKM benchmark")
    print("=" * 60)
    for length, weight in CASES:
        count, elapsed, rate = benchmark_case(length, weight)
        print("L = {}".format(length))
        print("weight = {}".format(weight))
        print("count = {}".format(count))
        print("time = {:.6f} sec".format(elapsed))
        print("rate = {:.0f} candidates/sec".format(rate))
        print("-" * 60)


if __name__ == "__main__":
    main()
