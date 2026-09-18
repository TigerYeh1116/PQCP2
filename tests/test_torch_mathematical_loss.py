"""Mathematical loss terms must be necessary, differentiable guidance only."""

from dataclasses import replace
from itertools import product

import pytest

torch = pytest.importorskip("torch")

from solver.correlation import full_correlation_profile
from solver.structured_energy import target_profile
from solver.target_profiles import TargetContentProfile, pair_content, target_content_profiles
from solver.torch_mathematical_loss import (
    bernoulli_pacf_variance_loss, mathematical_loss_components, mathematical_structure_loss,
    new_compression_factors, proper_compression_factors,
)
from solver.torch_search import BatchedPQCP, TorchSearch, TorchSearchConfig
from solver.verifier import verify_pqcp


def _small_assigned_solution():
    length = 4
    for a in product((0, 1), repeat=length):
        for b in product((0, 1), repeat=length):
            check = verify_pqcp(a, b)
            if not check.is_valid:
                continue
            shifts = check.nonzero_shifts
            k = min(shifts)
            ae, ao, be, bo = pair_content(a, b)
            profile = TargetContentProfile(length, k, check.profile[k] // 4,
                                           ae, ao, be, bo)
            return a, b, profile
    raise AssertionError("L=4 should have a small exhaustive test solution")


def test_factor_sets_cover_every_mathematical_compression():
    assert proper_compression_factors(44) == (2, 4, 11, 22)
    assert new_compression_factors(44) == (11, 22)
    assert proper_compression_factors(46) == (2, 23)
    assert new_compression_factors(46) == (23,)
    assert proper_compression_factors(68) == (2, 4, 17, 34)
    assert proper_compression_factors(90) == (2, 3, 5, 6, 9, 10, 15, 18, 30, 45)
    with pytest.raises(ValueError):
        proper_compression_factors(True)


def test_all_new_components_vanish_on_an_independently_verified_solution():
    a, b, profile = _small_assigned_solution()
    model = BatchedPQCP((profile,), torch.device("cpu"), continuous_kernel="dft").double()
    signs = torch.tensor([[[1 - 2*x for x in a], [1 - 2*x for x in b]]],
                         dtype=torch.float64)
    assert tuple(model.correlation(signs)[0].to(torch.int64).tolist()) == tuple(
        target_profile(4, profile.k, profile.eta)
    )
    parts = mathematical_loss_components(model, signs)
    assert set(parts) == {"psd_cap", "divisor_fold", "liftability", "pacf_lattice",
                          "bernoulli_variance"}
    for value in parts.values():
        assert value.item() == pytest.approx(0, abs=1e-20)
    assert mathematical_structure_loss(model, signs, "combined").item() == pytest.approx(0, abs=1e-20)


@pytest.mark.parametrize("length", [4, 8, 12, 44, 46, 68, 90, 94])
def test_every_binary_pair_is_exactly_liftable(length):
    generator = torch.Generator().manual_seed(91000 + length)
    signs = 1 - 2 * torch.randint(0, 2, (5, 2, length), generator=generator).to(torch.float64)
    from solver.torch_mathematical_loss import compression_liftability_loss
    assert torch.equal(compression_liftability_loss(signs), torch.zeros(5, dtype=torch.float64))
    profile = target_content_profiles(length, decimation_reduced=True)
    if profile:
        model = BatchedPQCP((profile[0],) * 5, torch.device("cpu"), continuous_kernel="dft").double()
        assert torch.equal(
            bernoulli_pacf_variance_loss(model, signs),
            torch.zeros(5, dtype=torch.float64),
        )


@pytest.mark.parametrize("mode", ["psd_cap", "divisor_lift", "lattice", "variance", "combined"])
def test_new_loss_has_finite_nonzero_gradient(mode):
    _, _, profile = _small_assigned_solution()
    model = BatchedPQCP((profile,) * 3, torch.device("cpu"), continuous_kernel="dft").double()
    values = torch.randn(3, 2, 4, generator=torch.Generator().manual_seed(2048),
                         dtype=torch.float64, requires_grad=True)
    loss = mathematical_structure_loss(model, values, mode).sum()
    loss.backward()
    assert torch.isfinite(loss)
    assert values.grad is not None and torch.isfinite(values.grad).all()
    assert values.grad.abs().sum().item() > 0


def test_none_path_is_bit_identical_and_new_modes_are_additive(tmp_path):
    base = TorchSearchConfig(8, batch_size=8, device="cpu", continuous_kernel="dft",
                             loss_mode="balanced")
    old = TorchSearch(base, tmp_path / "old")
    explicit = TorchSearch(replace(base, mathematical_loss="none"), tmp_path / "none")
    assert torch.equal(old.theta, explicit.theta)
    old_loss = old.model.loss(torch.tanh(old.theta), old.config)
    none_loss = explicit.model.loss(torch.tanh(explicit.theta), explicit.config)
    assert torch.equal(old_loss, none_loss)
    combined = replace(base, mathematical_loss="combined")
    expected = old_loss + mathematical_structure_loss(old.model, torch.tanh(old.theta), "combined")
    assert torch.allclose(old.model.loss(torch.tanh(old.theta), combined), expected)


def test_mathematical_loss_resume_is_deterministic(tmp_path):
    config = TorchSearchConfig(8, batch_size=8, device="cpu", continuous_kernel="dft",
                               mathematical_loss="combined")
    search = TorchSearch(config, tmp_path)
    for _ in range(3):
        search.step()
    path = tmp_path / "combined.json"
    search.checkpoint(path)
    resumed = TorchSearch.resume(path, tmp_path)
    assert resumed.config.mathematical_loss == "combined"
    for _ in range(3):
        search.step(); resumed.step()
    assert torch.equal(search.theta, resumed.theta)


def test_psd_cap_rejects_direct_kernel_and_unknown_mode():
    _, _, profile = _small_assigned_solution()
    model = BatchedPQCP((profile,), torch.device("cpu"), continuous_kernel="direct")
    values = torch.zeros(1, 2, 4)
    with pytest.raises(ValueError, match="DFT"):
        mathematical_structure_loss(model, values, "psd_cap")
    with pytest.raises(ValueError, match="unknown"):
        mathematical_structure_loss(model, values, "invented")
    with pytest.raises(ValueError, match="dft"):
        TorchSearchConfig(8, mathematical_loss="combined")


def test_divisor_lift_is_available_with_direct_kernel():
    """Compression identities do not require the Fourier implementation."""
    _, _, profile = _small_assigned_solution()
    model = BatchedPQCP((profile,), torch.device("cpu"), continuous_kernel="direct")
    values = torch.randn(1, 2, 4, generator=torch.Generator().manual_seed(91011),
                         requires_grad=True)
    loss = mathematical_structure_loss(model, values, "divisor_lift").sum()
    loss.backward()
    assert torch.isfinite(loss)
    assert values.grad is not None and torch.isfinite(values.grad).all()


def test_lattice_bootstrap_uses_one_structural_cosine_cycle(tmp_path):
    config = TorchSearchConfig(
        8, batch_size=4, device="cpu", continuous_kernel="dft",
        steps_per_restart=8, mathematical_loss="lattice_bootstrap",
    )
    search = TorchSearch(config, tmp_path)
    original = search.model.loss
    observed_modes = []

    def record(signs, active_config):
        observed_modes.append(active_config.mathematical_loss)
        return original(signs, active_config)

    search.model.loss = record
    for _ in range(4):
        search.step()
    assert observed_modes == ["lattice", "lattice", "none", "none"]


@pytest.mark.parametrize("length", [4, 6])
@pytest.mark.parametrize("kernel", ["direct", "dft"])
def test_bernoulli_variance_matches_exhaustive_randomized_binary_loss(length, kernel):
    """Cross-check the covariance derivation against every binary outcome."""
    profile = target_content_profiles(length, decimation_reduced=True)[0]
    model = BatchedPQCP((profile,), torch.device("cpu"), continuous_kernel=kernel).double()
    means = torch.linspace(-0.72, 0.67, 2 * length, dtype=torch.float64).reshape(1, 2, length)
    expected = 0.0
    target = target_profile(length, profile.k, profile.eta)
    for flat in product((-1, 1), repeat=2 * length):
        probability = 1.0
        for sign, mean in zip(flat, means.flatten().tolist()):
            probability *= (1 + sign * mean) / 2
        a = tuple(0 if value == 1 else 1 for value in flat[:length])
        b = tuple(0 if value == 1 else 1 for value in flat[length:])
        observed = full_correlation_profile(a, b)
        expected += probability * sum(
            (observed[u] - target[u]) ** 2 for u in range(1, length // 2 + 1)
        ) / 16
    mean_profile = model.continuous_correlation(means)
    squared_mean = (
        (mean_profile[:, 1:length // 2 + 1] - model.targets[:, 1:length // 2 + 1])
        .square().sum(-1) / 16
    )
    actual = squared_mean + bernoulli_pacf_variance_loss(model, means)
    # DFT constants are intentionally stored as production float32 before the
    # test model is promoted to double; direct is the high-precision oracle.
    tolerance = 1e-7 if kernel == "dft" else 1e-12
    assert actual.item() == pytest.approx(expected, rel=tolerance, abs=tolerance)
