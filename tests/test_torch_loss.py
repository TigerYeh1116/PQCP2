"""Loss algebra, true derivatives and safe checkpoint compatibility."""

from dataclasses import replace
import itertools
import json

import pytest

torch = pytest.importorskip("torch")

from solver.target_profiles import TargetContentProfile, pair_content, target_content_profiles
from solver.torch_search import BatchedPQCP, TorchSearch, TorchSearchConfig
from solver.verifier import verify_pqcp


@pytest.mark.parametrize("mode", ["legacy", "balanced", "projected"])
def test_loss_gradient_is_actual_derivative(mode):
    model = BatchedPQCP(target_content_profiles(8)[:2], torch.device("cpu")).double()
    x = torch.linspace(-0.9, 0.8, 32, dtype=torch.float64).reshape(2, 2, 8).requires_grad_()
    cfg = TorchSearchConfig(8, loss_mode=mode)
    assert torch.autograd.gradcheck(lambda y: model.loss(y, cfg), (x,))


def test_legacy_loss_is_unchanged_and_balanced_uses_normalized_folds():
    model = BatchedPQCP(target_content_profiles(8)[:2], torch.device("cpu"))
    x = torch.linspace(-0.9, 0.8, 32).reshape(2, 2, 8)
    parts = model.loss_components(x)
    base = parts["full"] + parts["origin"]
    penalty = 2 * parts["content"] + parts["binary"]
    expected_old = base + 2 * parts["fold2"] + 4 * parts["fold4"] + penalty
    expected_new = base + parts["fold2"] / 2 + parts["fold4"] / 4 + penalty
    torch.testing.assert_close(model.loss(x, TorchSearchConfig(8, loss_mode="legacy")), expected_old)
    torch.testing.assert_close(model.loss(x, TorchSearchConfig(8, loss_mode="balanced")), expected_new)
    for mode in ("legacy", "balanced"):
        torch.testing.assert_close(model.loss(x, TorchSearchConfig(8, loss_mode=mode, compression=False)),
                                   base + penalty)


def test_projection_distance_bounds_quantized_correlation_error():
    model = BatchedPQCP(target_content_profiles(8)[:4], torch.device("cpu"))
    x = torch.linspace(-0.97, 0.99, 64).reshape(4, 2, 8)
    quantized = model.project(x)
    gap = (model.correlation(quantized) - model.correlation(x)).abs()
    bound = 2 * torch.sqrt(2 * model.L * model.projection_distance(x))
    assert (gap <= bound[:, None] + 1e-5).all()


def test_rank_projection_is_nearest_fixed_content_vertex_exhaustively():
    for p in target_content_profiles(4):
        model = BatchedPQCP([p], torch.device("cpu"))
        x = torch.tensor([[[-0.8, 0.3, 0.7, -0.4], [0.9, 0.1, -0.1, -0.7]]])
        content = (p.a_even_ones, p.a_odd_ones, p.b_even_ones, p.b_odd_ones)
        distances = []
        for bits in itertools.product((0, 1), repeat=8):
            if pair_content(bits[:4], bits[4:]) == content:
                vertex = 1 - 2 * torch.tensor(bits).reshape(1, 2, 4)
                distances.append((x - vertex).square().sum().item())
        assert (x - model.project(x)).square().sum().item() == pytest.approx(min(distances))
        assert model.projection_distance(x).item() == pytest.approx(min(distances))


@pytest.mark.parametrize("length", [4, 8, 44, 46, 68, 94, 96])
def test_parallel_rank_distance_matches_stable_sort_including_ties(length):
    model = BatchedPQCP(target_content_profiles(length)[:4], torch.device("cpu"))
    x = torch.arange(len(model.profiles) * 2 * length, dtype=torch.float32).reshape(-1, 2, length) % 5
    torch.testing.assert_close(model.projection_distance(x),
                               (x - model.project(x)).square().sum((-1, -2)))


@pytest.mark.parametrize("length", [4, 8, 44, 46, 68])
def test_fold_normalization_is_orthogonal_projection_energy(length):
    e = torch.linspace(-2, 3, length, dtype=torch.float64)
    for factor in (2, 4):
        if length % factor:
            continue
        folded = e.reshape(factor, -1).sum(0)
        projection = (folded / factor).repeat(factor)
        assert torch.dot(e - projection, projection).abs() < 1e-10
        torch.testing.assert_close(folded.square().sum() / factor, projection.square().sum())
        assert projection.square().sum() <= e.square().sum() + 1e-10


@pytest.mark.parametrize("length", [8, 44, 46, 68])
def test_symmetric_residual_quadratic_condition_bounds(length):
    half = length // 2
    basis = torch.zeros(length, half + 1, dtype=torch.float64)
    basis[0, 0] = basis[half, half] = 1
    for shift in range(1, half):
        basis[shift, shift] = basis[length - shift, shift] = 2 ** -0.5
    base = torch.diag(torch.tensor([1.] * (half + 1) + [0.] * (half - 1), dtype=torch.float64))
    for mode in ("legacy", "balanced"):
        matrix = base.clone()
        for factor in (2, 4):
            if length % factor == 0:
                fold = torch.eye(length // factor, dtype=torch.float64).repeat(1, factor)
                coefficient = factor if mode == "legacy" else 1 / factor
                matrix += coefficient * fold.T @ fold
        eigenvalues = torch.linalg.eigvalsh(basis.T @ matrix @ basis)
        bound = (42 if length % 4 == 0 else 10) if mode == "legacy" else (6 if length % 4 == 0 else 4)
        assert eigenvalues.min() >= 0.5 - 1e-12
        assert eigenvalues.max() / eigenvalues.min() <= bound + 1e-10


def test_all_actual_small_solutions_have_zero_loss_and_nonsolutions_do_not():
    checked = 0
    for bits in itertools.product((0, 1), repeat=8):
        a, b = bits[:4], bits[4:]
        check = verify_pqcp(a, b)
        if not check.is_valid:
            continue
        k = min(check.nonzero_shifts)
        eta = check.profile[k] // 4
        p = TargetContentProfile(4, k, eta, *pair_content(a, b))
        model = BatchedPQCP([p], torch.device("cpu"))
        signs = (1 - 2 * torch.tensor(bits, dtype=torch.float32)).reshape(1, 2, 4)
        for mode in ("legacy", "balanced", "projected"):
            cfg = TorchSearchConfig(4, loss_mode=mode)
            assert model.loss(signs, cfg).item() == 0
            assert model.loss(signs * 0.9, cfg).item() > 0
        checked += 1
    assert checked > 0


@pytest.mark.parametrize("mode", ["legacy", "balanced", "projected"])
def test_new_loss_checkpoint_continuation_and_identical_initialization(mode, tmp_path):
    cfg = TorchSearchConfig(8, device="cpu", batch_size=4, loss_mode=mode)
    search = TorchSearch(cfg, tmp_path)
    control = TorchSearch(replace(cfg, loss_mode="legacy"), tmp_path)
    assert torch.equal(search.theta, control.theta)
    search.step()
    path = tmp_path / "state.json"
    search.checkpoint(path)
    resumed = TorchSearch.resume(path, tmp_path)
    assert resumed.config.loss_mode == mode
    for _ in range(3):
        search.step()
        resumed.step()
    assert torch.equal(search.theta, resumed.theta)
    assert torch.equal(search.optimizer.state[search.theta]["exp_avg"],
                       resumed.optimizer.state[resumed.theta]["exp_avg"])


def test_legacy_checkpoint_without_loss_mode_preserves_legacy(tmp_path):
    search = TorchSearch(TorchSearchConfig(8, device="cpu", batch_size=4), tmp_path)
    path = tmp_path / "state.json"
    search.checkpoint(path)
    data = json.loads(path.read_text())
    del data["config"]["loss_mode"]
    path.write_text(json.dumps(data))
    resumed = TorchSearch.resume(path, tmp_path)
    assert resumed.config.loss_mode == "legacy"
    search.step()
    resumed.step()
    assert torch.equal(search.theta, resumed.theta)


def test_invalid_loss_mode():
    with pytest.raises(ValueError, match="loss_mode"):
        TorchSearchConfig(44, loss_mode="invented")


@pytest.mark.parametrize("mode", ["legacy", "balanced", "projected"])
def test_production_loss_selection_and_cli(mode, tmp_path):
    import main
    from solver.torch_hybrid_runner import TorchHybridConfig, _new_continuous_search
    assert main.parse_args(["--torch-loss", mode]).torch_loss == mode
    config = TorchHybridConfig(8, device="cpu", root=tmp_path, continuous_batch_size=4,
                               continuous_loss_mode=mode)
    assert _new_continuous_search(config).config.loss_mode == mode


def test_production_default_and_rollback_configuration():
    import main
    from solver.torch_hybrid_runner import TorchHybridConfig
    assert main.parse_args([]).torch_loss == "balanced"
    assert TorchHybridConfig(44).continuous_loss_mode == "balanced"
    with pytest.raises(ValueError, match="continuous_loss_mode"):
        TorchHybridConfig(44, continuous_loss_mode="invalid")


@pytest.mark.parametrize("length", [44, 46, 68])
@pytest.mark.parametrize("mode", ["balanced", "projected"])
def test_new_losses_and_gradients_execute_on_real_cuda(length, mode):
    if not torch.cuda.is_available():
        pytest.skip("real CUDA unavailable")
    profiles = target_content_profiles(length)[:4]
    cpu = BatchedPQCP(profiles, torch.device("cpu"), continuous_kernel="dft")
    gpu = BatchedPQCP(profiles, torch.device("cuda"), continuous_kernel="dft")
    signs = torch.linspace(-0.93, 0.91, len(profiles) * 2 * length).reshape(-1, 2, length)
    x, y = signs.clone().requires_grad_(), signs.to("cuda").requires_grad_()
    cfg = TorchSearchConfig(length, loss_mode=mode)
    left, right = cpu.loss(x, cfg), gpu.loss(y, cfg)
    torch.testing.assert_close(left, right.cpu(), rtol=3e-5, atol=5e-4)
    left.sum().backward()
    right.sum().backward()
    assert y.grad.device.type == "cuda"
    torch.testing.assert_close(x.grad, y.grad.cpu(), rtol=3e-4, atol=5e-4)
    torch.testing.assert_close(gpu.projection_distance(y),
                               (y - gpu.project(y)).square().sum((-1, -2)))


def test_loss_benchmark_small_smoke_and_aggregation(tmp_path):
    from experiments.benchmark_torch_loss import run_case, summarize
    from solver.torch_hybrid_runner import TorchHybridConfig
    config = TorchHybridConfig(4, device="cpu", continuous_batch_size=4, root=tmp_path,
                               continuous_observation_interval=1)
    report = run_case(config, "projected", 0.005, polish=False)
    assert report["epochs"] > 0
    assert report["formation_elapsed"] >= 0.005
    assert (tmp_path / "report.json").is_file()
    summary = summarize([report])
    assert summary[0]["projected_median"] == report["projected_best_score"]
    assert summary[0]["runs"] == 1
