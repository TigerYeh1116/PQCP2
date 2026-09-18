"""Lifecycle tests for periodic MPS seed expansion/completion."""

from pathlib import Path
from types import SimpleNamespace

import pytest

import solver.torch_hybrid_runner as hybrid_runner
from solver.torch_hybrid_runner import (
    PreparedSeedBank, TorchHybridConfig, prepare_continuous_seed_bank,
    prepare_mps_seed_bank,
)


@pytest.mark.parametrize("kwargs", [
    {"L": 3}, {"L": 5}, {"L": 44, "seconds": 0},
    {"L": 44, "cycle_seconds": float("inf")}, {"L": 44, "threads": 0},
])
def test_hybrid_config_rejects_invalid_controls(kwargs):
    with pytest.raises(ValueError):
        TorchHybridConfig(**kwargs)


def test_hybrid_config_keeps_explicit_root_and_unbounded_budget(tmp_path):
    config = TorchHybridConfig(44, root=tmp_path)
    assert config.root == tmp_path
    assert config.seconds == float("inf")


def test_failed_mps_initialization_does_not_orphan_c(tmp_path, monkeypatch):
    import threading
    started, stopped = threading.Event(), threading.Event()

    def fake_c(_config, stop_event=None, **_kwargs):
        started.set()
        assert stop_event.wait(2)
        stopped.set()
        return SimpleNamespace(restarts=0, moves=0, swap_evaluations=0,
                               verified_candidates=0, new_solutions=0)

    def failed_init(_config):
        assert started.wait(1)
        raise RuntimeError("MPS initialization failed")

    monkeypatch.setattr(hybrid_runner, "run_c_search", fake_c)
    monkeypatch.setattr(hybrid_runner, "_new_continuous_search", failed_init)
    with pytest.raises(RuntimeError, match="initialization failed"):
        hybrid_runner.run_torch_hybrid_search(TorchHybridConfig(44, root=tmp_path))
    assert stopped.is_set()


def test_initial_handoff_budget_does_not_change_regular_formation_setting(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(hybrid_runner, "_new_continuous_search", lambda _: None)

    def prepare(config, *args, seconds=None, stop_event=None, **kwargs):
        seen.append((config.formation_seconds, seconds))
        stop_event.set()
        return PreparedSeedBank(None, 0, 0, "test", 100)

    def fake_c(_config, stop_event=None, **_kwargs):
        stop_event.wait(1)
        return SimpleNamespace(restarts=0, moves=0, swap_evaluations=0,
                               verified_candidates=0, new_solutions=0)

    monkeypatch.setattr(hybrid_runner, "prepare_continuous_seed_bank", prepare)
    monkeypatch.setattr(hybrid_runner, "run_c_search", fake_c)
    hybrid_runner.run_torch_hybrid_search(TorchHybridConfig(
        44, root=tmp_path, seconds=20, initial_formation_seconds=8))
    assert seen == [(45.0, 8)]


def test_mixed_handoff_keeps_existing_fkm_fallback_available(tmp_path):
    config = TorchHybridConfig(12, root=tmp_path, device="cpu", formation_seconds=0.01,
                               continuous_batch_size=8, continuous_archive_size=16,
                               polish_steps=1, polish_elites=4, polish_candidates=2,
                               completion_pair_seed_percent=50)
    result = prepare_continuous_seed_bank(config)
    assert result.count > 0 and result.path.is_file()
    assert result.pair_seed_percent == 50


@pytest.mark.parametrize("value", [-1, 101, True, 0.5])
def test_invalid_mix_percent_is_rejected(value):
    with pytest.raises(ValueError, match="completion_pair_seed_percent"):
        TorchHybridConfig(44, completion_pair_seed_percent=value)


def test_no_known_solution_uses_fixed_content_bootstrap_not_adam(tmp_path):
    prepared = prepare_mps_seed_bank(TorchHybridConfig(
        12, root=tmp_path, device="cpu", target_seed_count=12,
        bootstrap_islands_per_profile=1, bootstrap_population_size=8,
        bootstrap_pair_seed_percent=25,
        generations_per_cycle=2,
    ))
    assert prepared.strategy == "unseeded-refinement"
    assert prepared.pair_seed_percent == 25
    assert prepared.count > 0
    assert prepared.path.is_file()
    text = prepared.path.read_text(encoding="ascii")
    assert "L=12" in text and "\na=" in text and "\nb=" in text


def test_live_counters_remain_cumulative_across_refresh_cycles(tmp_path, monkeypatch):
    import threading

    bank = tmp_path / "bank.txt"
    bank.write_text("seed", encoding="ascii")
    c_started = threading.Event()
    mps_finished = threading.Event()
    monkeypatch.setattr(hybrid_runner, "_new_continuous_search", lambda _config: None)

    def fake_prepare(*_args, stop_event=None, **_kwargs):
        assert c_started.wait(timeout=1), "MPS formation began before C worker"
        mps_finished.set()
        stop_event.set()
        return PreparedSeedBank(bank, 1, 0.0, "test", 100)

    monkeypatch.setattr(hybrid_runner, "prepare_continuous_seed_bank", fake_prepare)
    calls = {"count": 0}

    def fake_c(config, line_callback=None, **_kwargs):
        calls["count"] += 1
        assert config.pair_seed_path is None  # first C run does not wait for MPS
        c_started.set()
        assert mps_finished.wait(timeout=1), "C and MPS did not overlap"
        line_callback("搜尋進度：已測試 2 組序列對")
        line_callback(
            "搜尋統計：restart=2, moves=10, swap evaluations=20, valid results=0"
        )
        return SimpleNamespace(
            restarts=2, moves=10, swap_evaluations=20,
            verified_candidates=0, new_solutions=0, interrupted=False,
        )

    monkeypatch.setattr(hybrid_runner, "run_c_search", fake_c)
    lines = []
    result = hybrid_runner.run_torch_hybrid_search(
        TorchHybridConfig(12, seconds=10, cycle_seconds=1, root=tmp_path),
        line_callback=lines.append,
    )
    assert calls["count"] == 1
    stats = [line for line in lines if line.startswith("搜尋統計：")]
    assert "restart=2, moves=10, swap evaluations=20" in stats[0]
    assert result.restarts == 2 and result.moves == 10
    assert result.last_seed_bank == bank


def test_solution_inventory_is_never_read_for_seed_formation(tmp_path, monkeypatch):
    """Even a corrupt existing L.txt must not influence the seed trajectory."""
    path = tmp_path / "44.txt"
    path.write_text("This is intentionally not a PQCP seed bank.")
    read_text = Path.read_text

    def guarded_read(self, *args, **kwargs):
        if self == path:
            pytest.fail("seed generation tried to read the solution inventory")
        return read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read)
    config = TorchHybridConfig(44, root=tmp_path, device="cpu", target_seed_count=8,
                               bootstrap_population_size=8, bootstrap_islands_per_profile=1,
                               generations_per_cycle=2)
    result = prepare_mps_seed_bank(config)
    assert result.count > 0 and result.generations == 2


def test_continuous_matrix_bank_exports_only_exact_close_target_elites(tmp_path):
    import json
    from solver.structured_energy import structured_energy_breakdown

    config = TorchHybridConfig(
        12, root=tmp_path, device="cpu", formation_seconds=0.02,
        continuous_batch_size=16, continuous_steps_per_restart=20,
        continuous_observation_interval=2, continuous_archive_size=32,
        target_seed_count=16, polish_elites=8, polish_steps=2,
        polish_candidates=2,
    )
    result = prepare_continuous_seed_bank(config)
    assert result.strategy == "continuous-matrix-dft+gpu-polish"
    assert result.pair_seed_percent == 100
    assert result.path is not None and result.path.is_file()
    records = json.loads(result.path.with_suffix(".txt.json").read_text())["records"]
    assert records
    energies = []
    for record in records:
        exact = structured_energy_breakdown(
            record["profile"], record["k"], record["sign"], (2, 4)
        ).weighted_total({2: 2, 4: 4})
        assert exact == record["target_energy"]
        energies.append(exact)
    assert energies == sorted(energies)
    assert len(energies) <= config.target_seed_count


def test_population_learning_survives_cycle_boundary(tmp_path):
    config = TorchHybridConfig(44, root=tmp_path, device="cpu", target_seed_count=8,
                               bootstrap_population_size=8, bootstrap_islands_per_profile=1,
                               generations_per_cycle=2)
    search = hybrid_runner._new_population(config)
    uninterrupted = hybrid_runner._new_population(config)
    prepare_mps_seed_bank(config, search=search)
    result = prepare_mps_seed_bank(config, cycle=1, search=search)
    for _ in range(4):
        uninterrupted.step()
    import torch
    assert result.generations == 4
    torch.testing.assert_close(search.logits, uninterrupted.logits, rtol=0, atol=0)
    torch.testing.assert_close(search.elites.bits, uninterrupted.elites.bits, rtol=0, atol=0)


def test_mps_discoveries_are_verified_saved_and_excluded_from_seed_bank(tmp_path):
    import json
    from solver.torch_seed_expansion import load_verified_solution_seeds

    config = TorchHybridConfig(4, root=tmp_path, device="cpu", bootstrap_population_size=16,
                               bootstrap_islands_per_profile=1, generations_per_cycle=2)
    result = prepare_mps_seed_bank(config)
    assert result.verified > 0 and result.new_solutions > 0
    # Reading the inventory here is independent verification, not initialization.
    assert len(load_verified_solution_seeds(tmp_path / "4.txt", 4)) == result.new_solutions
    if result.path is not None:
        records = json.loads(result.path.with_suffix(".txt.json").read_text())["records"]
        assert all(record["score"] > 0 for record in records)


def test_failed_verifier_cannot_publish_a_population_zero(tmp_path, monkeypatch):
    monkeypatch.setattr(hybrid_runner, "verify_pqcp", lambda *_: SimpleNamespace(is_valid=False))
    config = TorchHybridConfig(4, root=tmp_path, device="cpu", bootstrap_population_size=16,
                               bootstrap_islands_per_profile=1, generations_per_cycle=1)
    with pytest.raises(RuntimeError, match="independent exact verifier"):
        prepare_mps_seed_bank(config)
    assert not (tmp_path / "4.txt").exists()
