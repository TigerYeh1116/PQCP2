"""Bounded, reproducible simulated-annealing benchmark for Project 2 lengths."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from solver.annealing import AnnealingParameters, simulated_annealing
from solver.verifier import verify_pqcp


LENGTHS = (44, 46, 58, 68)
ITERATIONS = 25_000
RESTARTS = 2
SEED = 20_260_825


def main() -> None:
    """Run the baseline SA configuration without printing individual moves."""
    print("Simulated annealing benchmark")
    print("=" * 60)
    for length in LENGTHS:
        parameters = AnnealingParameters(
            max_iterations=ITERATIONS,
            initial_temperature=8.0,
            cooling_rate=0.9995,
            min_temperature=0.05,
            seed=SEED,
            restart_count=RESTARTS,
            fkm_pool_size=128,
        )
        result = simulated_annealing(length, parameters)
        print("L = {}".format(length))
        print("iterations = {}".format(result.iterations))
        print("restarts = {}".format(result.restarts))
        print("seed = {}".format(result.seed))
        print("initial score = {}".format(result.initial_scores[0]))
        print("best score = {}".format(result.best_score))
        print("elapsed time = {:.6f} sec".format(result.elapsed_time))
        print("solution found = {}".format(result.solved))
        if result.solved:
            verification = verify_pqcp(result.best_a, result.best_b)
            print("independent verifier = {}".format(verification.is_valid))
        print("-" * 60)


if __name__ == "__main__":
    main()
