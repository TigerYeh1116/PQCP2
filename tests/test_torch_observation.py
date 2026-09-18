"""Observation acceleration must not change candidates or trust GPU scores."""

from dataclasses import replace
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")

from solver.correlation import full_correlation_profile
from solver.objective import pqcp_objective
from solver.target_profiles import target_content_profiles
from solver.torch_search import BatchedPQCP, TorchSearch, TorchSearchConfig
from solver.verifier import verify_pqcp


def test_cache_reuses_exact_proof_and_refreshes_metadata(tmp_path):
    search = TorchSearch(TorchSearchConfig(44, batch_size=1, device="cpu", fast_observation=True), tmp_path)
    a, b = ((1 - search.model.project(search.theta)[0]) / 2).to(torch.int32).tolist()
    a, b = tuple(a), tuple(b)
    score = pqcp_objective(full_correlation_profile(a, b))
    with patch("solver.torch_search.verify_pqcp", wraps=verify_pqcp) as verifier:
        old = search._payload(a, b, 0, score)
        search.epoch, search.elapsed = 100, 1.5
        new = search._payload(a, b, 0, score)
    assert verifier.call_count == 1
    assert new["iteration"] == 100 and new["elapsed"] == 1.5
    assert new["profile"] == old["profile"] == full_correlation_profile(a, b)
    with pytest.raises(RuntimeError, match="disagrees"):
        search._payload(a, b, 0, score + 1)


def test_cached_observation_preserves_trajectory_and_resume(tmp_path):
    cfg = TorchSearchConfig(8, batch_size=8, device="cpu", observation_interval=1)
    slow = TorchSearch(cfg, tmp_path / "slow")
    fast = TorchSearch(replace(cfg, fast_observation=True), tmp_path / "fast")
    for _ in range(5):
        slow.step(); fast.step()
        slow.observe(); fast.observe()
    assert torch.equal(slow.theta, fast.theta)
    assert slow.lane_best == fast.lane_best
    assert slow.last_improved == fast.last_improved
    assert set(slow.archive) == set(fast.archive)
    assert slow.best["profile"] == fast.best["profile"]
    fast.checkpoint(tmp_path / "fast.json")
    resumed = TorchSearch.resume(tmp_path / "fast.json", tmp_path / "resumed")
    assert not resumed._exact_payload_cache
    for _ in range(3):
        fast.step(); resumed.step()
        fast.observe(); resumed.observe()
    assert torch.equal(fast.theta, resumed.theta)


def test_cached_zero_still_calls_independent_verifier(tmp_path):
    search = TorchSearch(TorchSearchConfig(4, batch_size=8, device="cpu", fast_observation=True), tmp_path)
    with patch("solver.torch_search.verify_pqcp", wraps=verify_pqcp) as verifier:
        search.observe()
        count = verifier.call_count
        assert search.best["score"] == 0
        search.observe()
        assert verifier.call_count > count
    assert all(not item["verified"] for item in search._exact_payload_cache.values())


def test_proof_cache_has_a_hard_capacity(tmp_path):
    search = TorchSearch(TorchSearchConfig(44, batch_size=1, device="cpu",
                                         fast_observation=True, archive_size=1), tmp_path)
    generator = torch.Generator().manual_seed(481)
    for _ in range(12):
        x = torch.rand(1, 2, 44, generator=generator)
        a, b = ((1-search.model.project(x)[0])/2).to(torch.int32).tolist()
        profile = full_correlation_profile(a, b)
        search._payload(tuple(a), tuple(b), 0, pqcp_objective(profile))
        assert len(search._exact_payload_cache) <= 4


def test_metal_and_cached_observation_preserve_complete_search_state(tmp_path):
    if not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    cfg = TorchSearchConfig(44, batch_size=16, continuous_kernel="dft", loss_mode="balanced")
    old = TorchSearch(cfg, tmp_path / "old")
    new = TorchSearch(replace(cfg, fast_observation=True, observation_backend="metal"), tmp_path / "new")
    for _ in range(4):
        old.step(); new.step()
        old.observe(); new.observe()
    assert torch.equal(old.theta, new.theta)
    assert old.lane_best == new.lane_best
    assert old.last_improved == new.last_improved
    assert old.archive.keys() == new.archive.keys()
    for key in old.archive:
        assert old.archive[key] == new.archive[key]
    new.checkpoint(tmp_path / "metal.json")
    resumed = TorchSearch.resume(tmp_path / "metal.json", tmp_path / "resumed")
    assert resumed.config.observation_backend == "metal"
    new.step(); resumed.step()
    new.observe(); resumed.observe()
    assert torch.equal(new.theta, resumed.theta)


@pytest.mark.parametrize("length", [4, 8, 44, 46, 68, 86, 94])
def test_metal_matches_stable_projection_and_full_integer_pacf(length):
    if not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    from solver.torch_metal import project_and_correlate
    profiles = target_content_profiles(length)[:16]
    model = BatchedPQCP(profiles, torch.device("mps"))
    # Many ties plus unequal odd/even contents exercise the stable index rule.
    x = (torch.arange(len(profiles)*2*length, device="mps", dtype=torch.float32) % 7).reshape(-1, 2, length)
    signs, profile = project_and_correlate(x, model.weights)
    assert torch.equal(signs, model.project(x))
    assert torch.equal(profile, model.correlation(signs))
    bits = ((1-signs)/2).to(torch.int32).cpu().tolist()
    for row, (a, b) in zip(profile.cpu().tolist(), bits):
        assert row == full_correlation_profile(a, b)


def test_invalid_metal_configuration_and_no_cpu_fallback():
    from solver.torch_metal import project_and_correlate
    with pytest.raises(ValueError, match="MPS"):
        project_and_correlate(torch.zeros(1, 2, 8), torch.zeros(1, 2, 2, dtype=torch.int64))
    with pytest.raises(ValueError, match="MPS"):
        TorchSearchConfig(44, device="cpu", observation_backend="metal")
