"""Parseval loss equivalence, gradients, CUDA execution and resume."""

from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from solver.target_profiles import target_content_profiles
from solver.torch_search import BatchedPQCP, TorchSearch, TorchSearchConfig


@pytest.mark.parametrize("length", [4, 8, 44, 46, 68, 94])
@pytest.mark.parametrize("mode", ["legacy", "balanced", "projected"])
@pytest.mark.parametrize("compression", [False, True])
def test_parseval_matches_existing_loss_and_gradient(length, mode, compression):
    model = BatchedPQCP(target_content_profiles(length)[:3], torch.device("cpu"), continuous_kernel="dft")
    x = torch.linspace(-0.94, 0.89, len(model.profiles)*2*length).reshape(-1, 2, length).requires_grad_()
    cfg = TorchSearchConfig(length, continuous_kernel="dft", loss_mode=mode, compression=compression)
    old, new = model.loss(x, cfg), model.loss(x, replace(cfg, loss_backend="spectral"))
    torch.testing.assert_close(old, new, rtol=3e-5, atol=1e-3)
    old_grad = torch.autograd.grad(old.sum(), x)[0]
    new_grad = torch.autograd.grad(new.sum(), x)[0]
    torch.testing.assert_close(old_grad, new_grad, rtol=4e-4, atol=1e-3)


def test_parseval_gradient_finite_differences():
    model = BatchedPQCP(target_content_profiles(8)[:2], torch.device("cpu"), continuous_kernel="dft").double()
    x = torch.linspace(-0.9, 0.7, 32, dtype=torch.float64).reshape(2, 2, 8).requires_grad_()
    cfg = TorchSearchConfig(8, continuous_kernel="dft", loss_backend="spectral", loss_mode="balanced")
    assert torch.autograd.gradcheck(lambda y: model.loss(y, cfg), (x,))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_spectral_resume_keeps_trajectory(tmp_path, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    cfg = TorchSearchConfig(8, batch_size=4, device=device, continuous_kernel="dft", loss_backend="spectral", loss_mode="balanced")
    search = TorchSearch(cfg, tmp_path)
    for _ in range(3): search.step()
    path = tmp_path / "spectral.json"
    search.checkpoint(path)
    resumed = TorchSearch.resume(path, tmp_path)
    assert resumed.config.loss_backend == "spectral"
    for _ in range(3):
        search.step()
        resumed.step()
    assert torch.equal(search.theta, resumed.theta)
    assert search.theta.device.type == device


def test_invalid_spectral_configuration():
    with pytest.raises(ValueError, match="requires"):
        TorchSearchConfig(44, loss_backend="spectral")
    with pytest.raises(ValueError, match="loss_backend"):
        TorchSearchConfig(44, loss_backend="unknown")
