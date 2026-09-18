"""Portable CUDA/MPS/CPU routing without changing search mathematics."""

import json

import pytest

torch = pytest.importorskip("torch")

import main
from solver.torch_device import resolve_torch_device
from solver.torch_hybrid_runner import TorchHybridConfig
from solver.torch_population import PopulationSearchConfig
from solver.torch_search import TorchSearch, TorchSearchConfig
from solver.torch_seed_expansion import SeedExpansionConfig


def test_auto_prefers_cuda_then_mps_then_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert resolve_torch_device("auto").type == "cuda"

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_torch_device("auto").type == "mps"

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert resolve_torch_device("auto").type == "cpu"


def test_explicit_accelerator_never_silently_falls_back(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        resolve_torch_device("cuda")
    with pytest.raises(RuntimeError, match="MPS"):
        resolve_torch_device("mps")
    assert resolve_torch_device("cpu").type == "cpu"


@pytest.mark.parametrize(
    "factory",
    (TorchSearchConfig, TorchHybridConfig, PopulationSearchConfig, SeedExpansionConfig),
)
@pytest.mark.parametrize("device", ("auto", "mps", "cuda", "cpu"))
def test_all_torch_configs_accept_portable_device_names(factory, device):
    assert factory(44, device=device).device == device


@pytest.mark.parametrize("factory", (TorchSearchConfig, TorchHybridConfig))
def test_custom_metal_observation_rejects_cuda(factory):
    with pytest.raises(ValueError, match="MPS"):
        factory(44, device="cuda", observation_backend="metal")


def test_auto_metal_rejects_cuda_after_hardware_resolution(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    with pytest.raises(RuntimeError, match="MPS"):
        TorchSearch(TorchSearchConfig(
            4, device="auto", observation_backend="metal", batch_size=1,
        ), tmp_path)


def test_cli_separates_method_from_device_and_preserves_shortcuts():
    args = main.parse_args(["--L", "44"])
    assert args.backend == "torch" and args.torch_device == "auto"
    assert main._requested_torch_device_name(args) == "auto"
    assert main._requested_torch_device_name(
        main.parse_args(["--L", "44", "--backend", "mps"])
    ) == "mps"
    assert main._requested_torch_device_name(
        main.parse_args(["--L", "44", "--backend", "cuda"])
    ) == "cuda"
    with pytest.raises(ValueError, match="衝突"):
        main._requested_torch_device_name(main.parse_args([
            "--L", "44", "--backend", "cuda", "--torch-device", "mps",
        ]))


def test_checkpoint_records_and_locks_resolved_device(tmp_path):
    search = TorchSearch(
        TorchSearchConfig(4, device="cpu", batch_size=2, observation_interval=1),
        tmp_path,
    )
    path = tmp_path / "checkpoint.json"
    search.checkpoint(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["resolved_device"] == "cpu"
    resumed = TorchSearch.resume(path, tmp_path / "resumed")
    assert resumed.device.type == resumed.config.device == "cpu"
