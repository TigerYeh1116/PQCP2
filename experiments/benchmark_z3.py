"""Bounded Z3 neighborhood-completion benchmark from reproducible SA centers."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from solver.annealing import AnnealingParameters, simulated_annealing
from solver.z3_solver import save_verified_solution, solve_with_z3


LENGTHS = (44, 46, 58, 68)
RADII = (0, 1, 2, 3)
TIMEOUT_MS = 1_000
SA_PARAMETERS = AnnealingParameters(
    max_iterations=25_000,
    initial_temperature=8.0,
    cooling_rate=0.9995,
    min_temperature=0.05,
    seed=20_260_825,
    restart_count=2,
    fkm_pool_size=128,
)


def main() -> None:
    """Benchmark exact Z3 completion only in radii 0 through 3 around SA output."""
    print("Hybrid Z3 completion benchmark")
    print("=" * 60)
    for length in LENGTHS:
        sa_result = simulated_annealing(length, SA_PARAMETERS)
        print("L = {}".format(length))
        print("SA score = {}".format(sa_result.best_score))
        for radius in RADII:
            result = solve_with_z3(
                length,
                sa_result.best_a,
                sa_result.best_b,
                radius=radius,
                timeout_ms=TIMEOUT_MS,
            )
            print("radius = {}".format(radius))
            print("timeout = {} ms".format(TIMEOUT_MS))
            print("status = {}".format(result.status))
            print("elapsed time = {:.6f} sec".format(result.elapsed_time))
            print("verified solution = {}".format(result.verified))
            if result.status == "UNKNOWN":
                print("reason = {}".format(result.reason))
            if result.verified:
                saved = save_verified_solution(result)
                print("saved result = {}".format(saved))
        print("-" * 60)


if __name__ == "__main__":
    main()
