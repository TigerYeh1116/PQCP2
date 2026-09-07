"""Bounded sequential multiple-elite SA -> Z3 portfolio benchmark."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from solver.annealing import AnnealingParameters
from solver.hybrid import PortfolioConfig, run_hybrid


LENGTHS = (44, 46, 58, 68)
SA_RUNS = 20
SELECTED_ELITES = 5
BASE_SEED = 20_260_900
SA_PARAMETERS = AnnealingParameters(
    max_iterations=750,
    initial_temperature=8.0,
    cooling_rate=0.999,
    min_temperature=0.05,
    restart_count=1,
    fkm_pool_size=128,
)
PORTFOLIO = PortfolioConfig(
    timeout_radius_0_ms=300,
    timeout_radius_1_ms=400,
    timeout_radius_2_ms=600,
    timeout_radius_3_ms=600,
    radius_2_elite_count=2,
    radius_3_elite_count=1,
    total_timeout_seconds=4.0,
)


def main() -> None:
    """Run reproducible small-radius exact portfolios without claiming global coverage."""
    print("Multiple-SA-elite Z3 portfolio benchmark")
    print("=" * 72)
    for length in LENGTHS:
        result = run_hybrid(
            length,
            num_sa_runs=SA_RUNS,
            elite_count=SELECTED_ELITES,
            parameters=SA_PARAMETERS,
            base_seed=BASE_SEED + length * 100,
            config=PORTFOLIO,
            result_directory=Path("results"),
        )
        print("L = {}".format(length))
        print("SA runs = {}".format(SA_RUNS))
        print("elite pool = {}".format(len(result.collected_elites)))
        print("selected elites = {}".format(len(result.selected_elites)))
        for task in result.tasks:
            print("seed={} score={} radius={} timeout={}ms status={} runtime={:.6f}s verified={}".format(
                task.elite.source_seed, task.elite.score, task.radius, task.timeout_ms,
                task.status, task.elapsed_time, task.verified,
            ))
            if task.reason is not None:
                print("reason={}".format(task.reason))
        print("verified solution = {}".format(result.solved))
        print("Z3 calls = {}".format(len(result.tasks)))
        print("status counts = {}".format(result.status_counts))
        print("SA collection runtime = {:.6f}s".format(result.sa_elapsed_time))
        print("selection + Z3 portfolio runtime = {:.6f}s".format(result.portfolio_elapsed_time))
        print("total end-to-end runtime = {:.6f}s".format(result.elapsed_time))
        print("-" * 72)


if __name__ == "__main__":
    main()
