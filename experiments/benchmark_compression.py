"""Compare exact full and incremental pair-correlation updates."""

from pathlib import Path
import random
import sys
from time import perf_counter
from typing import List, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from solver.compression import CorrelationState
from solver.correlation import full_correlation_profile


LENGTHS = (44, 46, 58, 68)
FLIPS_PER_CASE = 1_000
Flip = Tuple[str, int]


def make_case(length: int, flips: int) -> Tuple[List[int], List[int], List[Flip]]:
    """Create deterministic initial sequences and A/B bit-flip events."""
    rng = random.Random(20_260_823 + length)
    a = [rng.randrange(2) for _ in range(length)]
    b = [rng.randrange(2) for _ in range(length)]
    events = [("a" if rng.randrange(2) == 0 else "b", rng.randrange(length))
              for _ in range(flips)]
    return a, b, events


def time_full_recomputation(a: Sequence[int], b: Sequence[int], events: Sequence[Flip]):
    """Apply flips and recompute the complete baseline profile after each one."""
    current_a = list(a)
    current_b = list(b)
    started = perf_counter()
    for sequence_name, position in events:
        target = current_a if sequence_name == "a" else current_b
        target[position] ^= 1
        profile = tuple(full_correlation_profile(current_a, current_b))
    elapsed = perf_counter() - started
    return profile, elapsed


def time_incremental_updates(a: Sequence[int], b: Sequence[int], events: Sequence[Flip]):
    """Apply the same flips using one maintained CorrelationState."""
    state = CorrelationState(a, b)
    started = perf_counter()
    for sequence_name, position in events:
        if sequence_name == "a":
            state.flip_a(position)
        else:
            state.flip_b(position)
    elapsed = perf_counter() - started
    return state.profile, elapsed


def main() -> None:
    """Run exact baseline-versus-incremental comparisons at Project 2 lengths."""
    print("Compression / incremental correlation benchmark")
    print("=" * 60)
    for length in LENGTHS:
        a, b, events = make_case(length, FLIPS_PER_CASE)
        full_profile, full_time = time_full_recomputation(a, b, events)
        incremental_profile, incremental_time = time_incremental_updates(a, b, events)
        speedup = full_time / incremental_time if incremental_time else float("inf")
        print("L = {}".format(length))
        print("number of flips = {}".format(FLIPS_PER_CASE))
        print("full recomputation time = {:.6f} sec".format(full_time))
        print("incremental time = {:.6f} sec".format(incremental_time))
        print("speedup = {:.2f}x".format(speedup))
        print("final profile equality = {}".format(full_profile == incremental_profile))
        print("-" * 60)


if __name__ == "__main__":
    main()
