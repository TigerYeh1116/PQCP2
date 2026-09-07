"""Fast orchestration tests using tiny cases and fake bounded Z3 outcomes."""

from pathlib import Path

import pytest

from solver.annealing import AnnealingParameters
from solver.hybrid import (
    PortfolioConfig,
    SAElite,
    build_portfolio_schedule,
    collect_sa_elites,
    cyclic_aligned_pair_distance,
    deduplicate_elites,
    run_hybrid_portfolio,
    select_diverse_elites,
)
from solver.z3_solver import Z3CompletionResult


KNOWN_A = (0, 0, 0, 0)
KNOWN_B = (0, 0, 1, 1)


def _elite(a, b, score, seed=1, run_id=0):
    return SAElite(tuple(a), tuple(b), score, seed, run_id)


def _result(status, a=None, b=None, verified=False, reason=None):
    profile = (8, 4, 0, 4) if a is not None and b is not None else None
    return Z3CompletionResult(status, 4, 0, 0.001, a, b, profile, verified, reason)


def test_elite_collection_is_deterministic_for_tiny_case():
    parameters = AnnealingParameters(max_iterations=10, seed=0, restart_count=1, fkm_pool_size=8)
    first = collect_sa_elites(4, 3, parameters, base_seed=70)
    second = collect_sa_elites(4, 3, parameters, base_seed=70)
    assert first == second
    assert [elite.source_seed for elite in first] == [70, 71, 72]


def test_deduplication_removes_cyclic_equivalent_pair_without_swapping_ab():
    original = _elite((0, 0, 0, 1), (0, 0, 1, 1), 5, seed=2)
    rotated = _elite((0, 0, 1, 0), (0, 1, 1, 0), 6, seed=3)
    preserved_pairing = _elite((0, 0, 1, 1), (0, 0, 0, 1), 5, seed=4)
    deduplicated = deduplicate_elites((original, rotated, preserved_pairing))
    assert original in deduplicated
    assert preserved_pairing in deduplicated
    assert len(deduplicated) == 2


def test_cyclic_aligned_distance_removes_independent_rotation_effect():
    first = _elite((0, 0, 0, 1), (0, 0, 1, 1), 2)
    second = _elite((0, 0, 1, 0), (0, 1, 1, 0), 2, seed=2)
    assert cyclic_aligned_pair_distance(first, second) == 0


def test_diverse_selection_keeps_best_then_a_distant_elite():
    best = _elite((0, 0, 0, 0), (0, 0, 0, 0), 1, seed=1)
    nearby = _elite((0, 0, 0, 1), (0, 0, 0, 0), 2, seed=2)
    distant = _elite((0, 1, 0, 1), (1, 0, 1, 0), 4, seed=3)
    selected = select_diverse_elites((best, nearby, distant), 2)
    assert selected[0] == best
    assert selected[1] == distant


def test_radius_schedule_is_staged_and_late_radii_are_limited():
    elites = tuple(_elite((0, 0, 0, index % 2), (0, 0, 1, index % 2), index, seed=index)
                   for index in range(5))
    config = PortfolioConfig(radius_2_elite_count=2, radius_3_elite_count=1)
    schedule = build_portfolio_schedule(elites, config)
    assert [radius for _, radius, _ in schedule] == [0] * 5 + [1] * 5 + [2] * 2 + [3]
    assert len({(elite.source_seed, radius) for elite, radius, _ in schedule}) == len(schedule)


def test_nonredundant_production_schedule_can_run_radius_three_only():
    elites = tuple(_elite(KNOWN_A, KNOWN_B, score) for score in range(3))
    config = PortfolioConfig(
        radius_0_elite_count=0, radius_1_elite_count=0,
        radius_2_elite_count=0, radius_3_elite_count=1,
    )
    assert [radius for _, radius, _ in build_portfolio_schedule(elites, config)] == [3]


def test_portfolio_sat_is_independently_verified_and_stops():
    calls = []

    def fake_solver(_L, _a, _b, radius, _timeout):
        calls.append(radius)
        return _result("SAT", KNOWN_A, KNOWN_B, verified=False)

    result = run_hybrid_portfolio(4, (_elite(KNOWN_A, KNOWN_B, 0),), 1, PortfolioConfig(), fake_solver)
    assert result.solved
    assert result.tasks[0].verified
    assert calls == [0]


def test_unknown_is_counted_separately_from_unsat():
    outcomes = iter((
        _result("UNSAT"),
        _result("UNKNOWN", reason="timeout"),
        _result("UNSAT"),
        _result("UNSAT"),
    ))

    def fake_solver(*_args):
        return next(outcomes)

    result = run_hybrid_portfolio(
        4,
        (_elite((0, 0, 0, 0), (0, 0, 0, 0), 1),),
        1,
        PortfolioConfig(radius_2_elite_count=1, radius_3_elite_count=1),
        fake_solver,
    )
    assert result.status_counts == {"SAT": 0, "UNSAT": 3, "UNKNOWN": 1}
    assert not result.solved


def test_budget_prevents_any_task_when_exhausted():
    def should_not_run(*_args):
        raise AssertionError("budget-exhausted portfolio must not call Z3")

    result = run_hybrid_portfolio(
        4, (_elite((0, 0, 0, 0), (0, 0, 0, 0), 1),), 1,
        PortfolioConfig(total_timeout_seconds=0), should_not_run,
    )
    assert not result.tasks


def test_solution_storage_refuses_to_overwrite(tmp_path):
    existing = Path(tmp_path) / "4.txt"
    existing.write_text("existing", encoding="utf-8")

    def fake_solver(*_args):
        return _result("SAT", KNOWN_A, KNOWN_B)

    with pytest.raises(FileExistsError):
        run_hybrid_portfolio(
            4, (_elite(KNOWN_A, KNOWN_B, 0),), 1, PortfolioConfig(), fake_solver, tmp_path
        )


@pytest.mark.parametrize("elite_count", [0, True])
def test_invalid_elite_count_is_rejected(elite_count):
    with pytest.raises(ValueError):
        select_diverse_elites((_elite(KNOWN_A, KNOWN_B, 0),), elite_count)
