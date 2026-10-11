"""Hardware sizing must preserve overrides and the legacy portable path."""

import pytest

import main
from solver.hardware_resources import HostResources, default_batch_size, default_c_threads
from solver.torch_hybrid_runner import TorchHybridConfig, run_torch_hybrid_search


@pytest.mark.parametrize("device,host,expected", [
    ("mps", HostResources(12, 32 * 1024**3, True), (8192, 16384, 6)),
    ("mps", HostResources(8, 16 * 1024**3, True), (2048, 1024, 4)),
    ("mps", HostResources(12, 0, True), (2048, 1024, 4)),
    ("cpu", HostResources(12, 32 * 1024**3, True), (2048, 1024, 4)),
    ("cuda", HostResources(32, 64 * 1024**3, False), (2048, 1024, 4)),
])
def test_hardware_policy(device, host, expected):
    assert (default_batch_size(44, device, host), default_batch_size(68, device, host),
            default_c_threads(device, host)) == expected


def test_explicit_batch_bypasses_hardware_probe(monkeypatch):
    def forbidden():
        raise AssertionError("explicit override must not need host detection")
    monkeypatch.setattr("solver.hardware_resources.host_resources", forbidden)
    assert main._effective_torch_batch_size(44, 128, "mps") == 128
    with pytest.raises(ValueError):
        main._effective_torch_batch_size(44, 0, "mps")


def test_larger_experimental_lengths_keep_previous_resource_bound():
    host = HostResources(12, 32 * 1024**3, True)
    assert default_batch_size(94, "mps", host) == 16384
    assert default_batch_size(96, "mps", host) == 2048


def test_restricted_host_detection_keeps_original_limits(monkeypatch):
    import solver.hardware_resources as resources
    monkeypatch.setattr(resources.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(resources.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(resources.os, "cpu_count", lambda: 12)

    def denied(*args, **kwargs):
        raise PermissionError("sandbox")
    monkeypatch.setattr(resources.subprocess, "check_output", denied)
    resources.host_resources.cache_clear()
    try:
        host = resources.host_resources()
        assert default_batch_size(44, "mps", host) == 2048
        assert default_c_threads("mps", host) == 4
    finally:
        resources.host_resources.cache_clear()


@pytest.mark.parametrize("requested,workers,expected", [(6, 8, 6), (8, 2, 2), (None, 8, 4)])
def test_c_limit_reaches_worker_and_respects_workers(tmp_path, monkeypatch, requested, workers, expected):
    import threading
    from types import SimpleNamespace
    import solver.torch_hybrid_runner as hybrid
    called = threading.Event()
    seen = []

    def fake_c(config, stop_event, **kwargs):
        seen.append(config.threads)
        called.set()
        stop_event.set()
        return SimpleNamespace(restarts=0, moves=0, swap_evaluations=0,
                               verified_candidates=0, new_solutions=0)

    def init(config):
        assert called.wait(5)

    monkeypatch.setattr(hybrid, "run_c_search", fake_c)
    monkeypatch.setattr(hybrid, "_new_continuous_search", init)
    run_torch_hybrid_search(TorchHybridConfig(44, threads=workers, c_threads=requested, root=tmp_path))
    assert seen == [expected]


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_c_limit(value):
    with pytest.raises(ValueError, match="c_threads"):
        TorchHybridConfig(44, c_threads=value)
