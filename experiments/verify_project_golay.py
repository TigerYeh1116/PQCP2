"""Quick exact verification of every registered Project 2 Golay seed.

This is deliberately not a search benchmark.  It constructs each seed in
milliseconds, recomputes its full periodic correlation profile with the
project baseline, and reports whether it is an exact periodic Golay pair or
a truthfully-labelled Turyn near-periodic construction.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from solver.correlation import full_correlation_profile
from solver.golay import PROJECT_GOLAY_LENGTHS, is_periodic_golay_pair, project_length_golay_seed
from solver.objective import pqcp_objective


def main() -> int:
    """Print one compact, independently recomputed row per requested length."""
    print("L   kind                  score  nonzero  periodic-GCP")
    print("--  --------------------  -----  -------  ------------")
    for length in sorted(PROJECT_GOLAY_LENGTHS):
        seed = project_length_golay_seed(length, seed=123)
        profile = full_correlation_profile(seed.a, seed.b)
        score = pqcp_objective(profile)
        nonzero_count = sum(value != 0 for value in profile[1:])
        periodic = is_periodic_golay_pair(seed.a, seed.b)
        print("{:<3} {:<21} {:>5} {:>8}  {}".format(
            length, seed.kind, score, nonzero_count, "yes" if periodic else "no"
        ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
