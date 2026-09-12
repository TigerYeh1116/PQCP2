"""Feasibility projection identities, not assumptions about convergence."""

import pytest

torch = pytest.importorskip("torch")

from solver.target_profiles import target_content_profiles
from solver.torch_feasibility import douglas_rachford_step, project_target_spectrum
from solver.torch_search import BatchedPQCP, TorchSearch, TorchSearchConfig
from solver.verifier import verify_pqcp


@pytest.mark.parametrize("length", [4, 8, 44, 46, 68, 94])
@pytest.mark.parametrize("zero", [False, True])
def test_projection_has_exact_target_in_real_arithmetic_and_fixed_sums(length, zero):
    model = BatchedPQCP(target_content_profiles(length)[:4], torch.device("cpu"), continuous_kernel="dft").double()
    x = torch.zeros(len(model.profiles), 2, length, dtype=torch.float64)
    if not zero:
        x = torch.randn(x.shape, generator=torch.Generator().manual_seed(length), dtype=torch.float64)
    projected = project_target_spectrum(model, x)
    assert torch.isfinite(projected).all()
    torch.testing.assert_close(model.correlation(projected), model.targets, rtol=1e-5, atol=6e-5)
    expected = (length//2 - 2*model.weights).to(torch.float64)
    actual = projected.reshape(-1, 2, length//2, 2).sum(-2)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-5)
    torch.testing.assert_close(project_target_spectrum(model, projected), projected, rtol=1e-5, atol=2e-5)


def test_dr_step_is_the_stated_two_set_iteration():
    model = BatchedPQCP(target_content_profiles(8)[:2], torch.device("cpu"), continuous_kernel="dft")
    x = torch.linspace(-1.2, 1.3, 32).reshape(2, 2, 8)
    binary = model.project(x)
    expected = x + project_target_spectrum(model, 2*binary-x) - binary
    assert torch.equal(douglas_rachford_step(model, x), expected)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_feasibility_resume_and_independent_observation(tmp_path, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    search = TorchSearch(TorchSearchConfig(8, batch_size=8, device=device, continuous_kernel="dft",
                                          optimization_mode="douglas_rachford",
                                          observation_backend="torch"), tmp_path)
    for _ in range(3): search.step()
    search.observe()
    path = tmp_path / "dr.json"
    search.checkpoint(path)
    resumed = TorchSearch.resume(path, tmp_path)
    for _ in range(3):
        search.step(); resumed.step()
    assert torch.equal(search.theta, resumed.theta)
    assert not search.optimizer.state
    assert resumed.config.optimization_mode == "douglas_rachford"


def test_feasibility_requires_dft():
    with pytest.raises(ValueError, match="DFT"):
        TorchSearchConfig(44, optimization_mode="douglas_rachford")


def test_exact_small_binary_solution_is_a_projection_fixed_point(tmp_path):
    # A tiny test fixture from independent observation, never a production seed.
    search = TorchSearch(TorchSearchConfig(4, batch_size=8, device="cpu",
                                          continuous_kernel="dft"), tmp_path)
    binary = search.model.project(search.theta)
    profile = search.model.correlation(binary)
    assigned_solutions = (profile == search.model.targets).all(-1)
    assert assigned_solutions.any()
    for a, b in ((1 - binary[assigned_solutions]) / 2).to(torch.int32).tolist():
        assert verify_pqcp(a, b).is_valid
    updated = douglas_rachford_step(search.model, binary)
    torch.testing.assert_close(updated[assigned_solutions], binary[assigned_solutions],
                               rtol=0, atol=1e-6)
    assert torch.equal(search.model.project(updated)[assigned_solutions],
                       binary[assigned_solutions])


@pytest.mark.parametrize("length", [4, 8, 44, 46, 68, 94])
def test_cuda_projection_preserves_pacf_target_and_all_parity_sums(length):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    profiles = target_content_profiles(length)
    model = BatchedPQCP(profiles, torch.device("cuda"), continuous_kernel="dft")
    values = torch.randn((len(profiles), 2, length),
                         generator=torch.Generator().manual_seed(length)).to("cuda")
    projected = project_target_spectrum(model, values)
    # This is a floating projection identity, NOT a PQCP acceptance tolerance.
    torch.testing.assert_close(model.correlation(projected), model.targets,
                               rtol=1e-4, atol=5e-4)
    sums = projected.reshape(-1, 2, length//2, 2).sum(-2)
    expected = (length//2 - 2*model.weights).to(torch.float32)
    torch.testing.assert_close(sums, expected, rtol=1e-4, atol=5e-4)
