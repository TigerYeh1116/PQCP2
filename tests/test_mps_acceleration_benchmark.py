"""Fast aggregation tests only; no timed GPU experiment is started here."""

import pytest

pytest.importorskip("torch")

from experiments.benchmark_mps_acceleration import exhaustive_swap_draws, recorded_pairs, summarize


def test_benchmark_equivalence_matches_production_writer():
    assert recorded_pairs("a=0011\nb=0101\na=0101\nb=0011\n") == {("0011", "0101")}
    assert len(recorded_pairs("a=0011\nb=0101\na=0110\nb=0101\n")) == 2


def test_zero_baseline_does_not_claim_infinite_or_threefold_speedup():
    rows = [{"L": 44, "method": m, "wall_seconds": 10, "new_solutions": n,
             "best_score": s} for m, n, s in [("direct", 0, 8), ("dft", 2, 0)]]
    assert all(row["rate_ratio_to_direct"] is None for row in summarize(rows))


def test_aggregate_actual_exposure_not_mean_of_per_run_rates():
    rows = [{"L": 44, "method": m, "wall_seconds": t, "new_solutions": n,
             "best_score": 0} for m, t, n in [("direct", 10, 1), ("direct", 30, 1), ("dft", 40, 6)]]
    result = {row["method"]: row for row in summarize(rows)}
    assert result["direct"]["new_per_second"] == 0.05
    assert result["dft"]["rate_ratio_to_direct"] == pytest.approx(3)


def test_exhaustive_draws_cover_all_unordered_same_parity_pairs():
    import torch
    draws = exhaustive_swap_draws(8, 2, 2, torch.Generator().manual_seed(13))
    expected = {(g, i, j) for g in range(4) for i in range(4) for j in range(i + 1, 4)}
    for step in draws.tolist():
        for lane in step:
            assert {(int(g * 4), int(i * 4), int(j * 4)) for g, i, j, _ in lane} == expected


def test_exhaustive_probe_memory_is_bounded_before_allocation():
    import torch
    with pytest.raises(ValueError, match="memory bound"):
        exhaustive_swap_draws(94, 512, 8000, torch.Generator())
