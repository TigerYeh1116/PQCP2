"""Independent PQCP math, device, persistence and Adam continuation checks."""

import json
import random

import pytest

torch = pytest.importorskip("torch")

from solver.correlation import full_correlation_profile
from solver.objective import pqcp_objective
from solver.structured_energy import structured_energy_breakdown
from solver.target_profiles import TargetContentProfile, pair_content, target_content_profiles
from solver.torch_search import BatchedPQCP, TorchSearch, TorchSearchConfig, require_device
from solver.verifier import verify_pqcp


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("real CUDA is unavailable in this execution environment")
    return torch.device(request.param)


@pytest.mark.parametrize("length", [4, 8, 44, 46, 68, 86, 94])
def test_batched_correlation_projection_loss_match_integer_baselines(length, device):
    profiles = target_content_profiles(length, decimation_reduced=True)
    selected = tuple(profiles[i % len(profiles)] for i in range(7))
    model = BatchedPQCP(selected, device)
    rng = random.Random(length)
    values = [[[rng.uniform(-1, 1) for _ in range(length)] for _ in range(2)] for _ in selected]
    theta = torch.tensor(values, device=device, dtype=torch.float32)
    signs = model.project(theta)
    actual = model.correlation(signs).cpu().tolist()
    scores = model.discrete_scores(model.correlation(signs)).cpu().tolist()
    parts = {key: val.cpu().tolist() for key, val in model.loss_components(signs).items()}
    pairs = ((1 - signs) / 2).to(torch.int32).cpu().tolist()
    for i, ((a, b), p) in enumerate(zip(pairs, selected)):
        exact = full_correlation_profile(a, b)
        assert actual[i] == exact
        assert scores[i] == pqcp_objective(exact)
        assert pair_content(a, b) == (p.a_even_ones, p.a_odd_ones, p.b_even_ones, p.b_odd_ones)
        factors = (2, 4) if length % 4 == 0 else (2,)
        expected = structured_energy_breakdown(exact, p.k, p.eta, factors)
        assert parts["full"][i] == expected.full
        for factor in factors:
            assert parts["fold{}".format(factor)][i] == expected.component(factor)
        assert parts["origin"][i] == parts["binary"][i] == parts["content"][i] == 0


def test_float_loss_has_finite_device_gradients(device):
    profiles = target_content_profiles(8, decimation_reduced=True)[:4]
    model = BatchedPQCP(profiles, device)
    theta = torch.linspace(-1.5, 1.5, len(profiles) * 16, device=device).reshape(-1, 2, 8).requires_grad_()
    model.loss(torch.tanh(theta), TorchSearchConfig(8)).sum().backward()
    assert theta.grad.device.type == device.type
    assert torch.isfinite(theta.grad).all().item()
    assert theta.grad.abs().sum().item() > 0


def test_continuous_gradient_matches_finite_differences():
    p = target_content_profiles(8, decimation_reduced=True)[0]
    model = BatchedPQCP([p], torch.device("cpu")).double()
    theta = torch.linspace(-1.2, 0.9, 16, dtype=torch.float64).reshape(1, 2, 8).requires_grad_()
    cfg = TorchSearchConfig(8)
    assert torch.autograd.gradcheck(lambda x: model.loss(torch.tanh(x), cfg), (theta,))


@pytest.mark.parametrize("length", [4, 8, 44, 46, 68, 86, 94])
def test_dft_relaxation_matches_direct_value_gradient_and_keeps_exact_gate(length, device):
    profiles = target_content_profiles(length)[:3]
    direct = BatchedPQCP(profiles, device)
    spectral = BatchedPQCP(profiles, device, continuous_kernel="dft")
    generator = torch.Generator().manual_seed(length)
    signs = (torch.rand(len(profiles), 2, length, generator=generator) * 2 - 1).to(device).requires_grad_()
    left, right = direct.continuous_correlation(signs), spectral.continuous_correlation(signs)
    torch.testing.assert_close(left, right, rtol=2e-5, atol=3e-5)
    cfg = TorchSearchConfig(length)
    left, right = direct.loss(signs, cfg), spectral.loss(signs, cfg)
    torch.testing.assert_close(left, right, rtol=3e-5, atol=3e-4)
    grad_left = torch.autograd.grad(left.sum(), signs)[0]
    grad_right = torch.autograd.grad(right.sum(), signs)[0]
    torch.testing.assert_close(grad_left, grad_right, rtol=3e-4, atol=3e-4)
    binary = spectral.project(signs.detach())
    assert torch.equal(spectral.correlation(binary), direct.correlation(binary))
    assert spectral.dft_real.device == signs.device


def test_dft_checkpoint_preserves_kernel_and_trajectory(tmp_path, device):
    cfg = TorchSearchConfig(8, device=device.type, batch_size=4, continuous_kernel="dft")
    continuous = TorchSearch(cfg, tmp_path)
    for _ in range(3):
        continuous.step()
    path = tmp_path / "dft.json"
    continuous.checkpoint(path)
    resumed = TorchSearch.resume(path, tmp_path)
    assert resumed.model.continuous_kernel == "dft"
    for _ in range(4):
        continuous.step()
        resumed.step()
    assert torch.equal(continuous.theta, resumed.theta)


def test_invalid_continuous_kernel_is_rejected():
    with pytest.raises(ValueError, match="continuous_kernel"):
        TorchSearchConfig(44, continuous_kernel="approximate_discrete")


def test_invalid_optimization_mode_is_rejected():
    with pytest.raises(ValueError, match="optimization_mode"):
        TorchSearchConfig(44, optimization_mode="wishful")


def test_straight_through_step_scores_binary_forward(monkeypatch):
    search = TorchSearch(TorchSearchConfig(
        8, device="cpu", batch_size=4, optimization_mode="straight_through",
    ))
    original = search.model.loss
    observed = []

    def audit(signs, config):
        observed.append(signs.detach().clone())
        return original(signs, config)

    monkeypatch.setattr(search.model, "loss", audit)
    search.step()
    assert len(observed) == 1
    assert bool(((observed[0] == -1) | (observed[0] == 1)).all().item())
    assert search.theta.grad is not None
    assert search.theta.grad.abs().sum().item() > 0


def test_kernel_comparison_uses_identical_fkm_start_and_old_checkpoint_defaults(tmp_path):
    direct = TorchSearch(TorchSearchConfig(44, device="cpu", batch_size=4), tmp_path)
    dft = TorchSearch(TorchSearchConfig(44, device="cpu", batch_size=4, continuous_kernel="dft"), tmp_path)
    assert torch.equal(direct.theta, dft.theta)
    assert direct.model.profiles == dft.model.profiles
    direct.step()
    path = tmp_path / "old.json"
    direct.checkpoint(path)
    state = json.loads(path.read_text())
    del state["config"]["continuous_kernel"]
    del state["config"]["optimization_mode"]
    path.write_text(json.dumps(state))
    resumed = TorchSearch.resume(path, tmp_path)
    assert resumed.model.continuous_kernel == "direct"
    assert resumed.config.optimization_mode == "relaxed"
    assert torch.equal(direct.theta, resumed.theta)


def test_pacp_target_is_not_mistaken_for_pqcp():
    model = BatchedPQCP([target_content_profiles(8)[0]], torch.device("cpu"))
    # Reference slides put the sole nonzero sidelobe at L/2, unlike Project 2.
    half_shift_only = torch.tensor([[16, 0, 0, 0, 4, 0, 0, 0]], dtype=torch.float32)
    assert model.discrete_scores(half_shift_only).item() == 1


def test_cuda_unavailable_does_not_silently_use_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        require_device("cuda")


def test_removed_mps_device_is_rejected():
    with pytest.raises(ValueError, match="unsupported"):
        require_device("mps")


def test_solution_at_another_shift_is_still_saved(tmp_path, device):
    # A recorded L44 elite: fixed k=2 energy was 32, but k=20 is a true PQCP.
    a = tuple(map(int, "10100001100011100100011010001011000010011111"))
    b = tuple(map(int, "01010100011010100100111110011010010000000100"))
    assert verify_pqcp(a, b).is_valid
    p = TargetContentProfile(44, 2, -1, *pair_content(a, b))
    cfg = TorchSearchConfig(44, device=device.type, batch_size=1)
    search = TorchSearch(cfg, tmp_path, initialize=False)
    search.model = BatchedPQCP([p], device)
    search.theta = torch.nn.Parameter(torch.tensor([[a, b]], device=device, dtype=torch.float32) * -2 + 1)
    search.lane_best, search.last_improved = [None], [0]
    search.observe()
    assert search.best["score"] == 0
    assert search.best["nonzero_shifts"] == [20, 24]
    assert search.new_solutions == 1
    original = (tmp_path / "44.txt").read_text()
    search.observe()
    assert search.new_solutions == 1
    assert (tmp_path / "44.txt").read_text() == original
    payload = json.loads(search.best_path.read_text())
    assert payload["profile"] == full_correlation_profile(a, b)


def test_resume_preserves_adam_trajectory_without_fkm_reinitialization(tmp_path, monkeypatch, device):
    cfg = TorchSearchConfig(8, device=device.type, batch_size=4, observation_interval=2)
    continuous = TorchSearch(cfg, tmp_path)
    for _ in range(3):
        continuous.step()
    path = tmp_path / "state.json"
    continuous.checkpoint(path)
    monkeypatch.setattr(TorchSearch, "_initial_pairs", lambda *_args: pytest.fail("resume reinitialized FKM"))
    resumed = TorchSearch.resume(path, tmp_path)
    for _ in range(4):
        continuous.step()
        resumed.step()
    assert torch.equal(continuous.theta, resumed.theta)
    assert continuous.epoch == resumed.epoch == 7
    for key in ("exp_avg", "exp_avg_sq"):
        assert torch.equal(continuous.optimizer.state[continuous.theta][key], resumed.optimizer.state[resumed.theta][key])


def test_budget_interrupt_and_rebirth_keep_checkpoint_and_best(tmp_path, monkeypatch):
    cfg = TorchSearchConfig(8, device="cpu", batch_size=4, steps_per_restart=2, observation_interval=1)
    search = TorchSearch(cfg, tmp_path)
    result = search.run(max_steps=5)
    assert search.restarts == 12 and search.generation == 2
    assert result["epoch"] == 5  # continues despite finding solutions
    assert search.best_path.exists()
    assert search.optimizer.state[search.theta]["step"].item() == 1  # fresh Adam after rebirth
    before = search.epoch
    search.run(seconds=0)
    assert search.epoch == before

    def interrupt():
        raise KeyboardInterrupt

    monkeypatch.setattr(search, "step", interrupt)
    result = search.run()
    assert result["interrupted"]
    payload = json.loads((tmp_path / "checkpoints/L8_torch.json").read_text())
    assert payload["epoch"] == before
    assert payload["best"]["score"] == search.best["score"]


def test_corrupt_checkpoint_is_rejected(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"format": "wrong"}')
    with pytest.raises(ValueError, match="checkpoint"):
        TorchSearch.resume(path, tmp_path)


def test_split_run_crosses_rebirth_with_same_trajectory(tmp_path, device):
    cfg = TorchSearchConfig(8, device=device.type, batch_size=4,
                            steps_per_restart=5, observation_interval=2, stagnation_steps=3)
    continuous = TorchSearch(cfg, tmp_path / "continuous")
    continuous.run(max_steps=9)
    split = TorchSearch(cfg, tmp_path / "split")
    # Stop away from the regular observation boundary.
    result = split.run(max_steps=3)
    resumed = TorchSearch.resume(result["checkpoint"], tmp_path / "split")
    resumed.run(max_steps=6)
    assert continuous.epoch == resumed.epoch
    assert continuous.generation == resumed.generation
    assert continuous.round_step == resumed.round_step
    assert torch.equal(continuous.theta, resumed.theta)


def test_tampered_checkpoint_counter_fails_clearly(tmp_path):
    s = TorchSearch(TorchSearchConfig(8, device="cpu", batch_size=2), tmp_path)
    s.step()
    path = tmp_path / "state.json"
    s.checkpoint(path)
    payload = json.loads(path.read_text())
    payload["round_step"] = 2
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Adam step"):
        TorchSearch.resume(path, tmp_path)


def test_all_zero_score_lanes_are_verified_beyond_elite_limit(tmp_path, monkeypatch):
    cfg = TorchSearchConfig(4, device="cpu", batch_size=16, elite_count=1)
    search = TorchSearch(cfg, tmp_path)
    calls = []
    original = search._payload
    def record(a, b, lane, score):
        calls.append((lane, score))
        return original(a, b, lane, score)
    monkeypatch.setattr(search, "_payload", record)
    signs = search.model.project(search.theta)
    expected = int((search.model.discrete_scores(search.model.correlation(signs)) == 0).sum().item())
    search.observe()
    assert expected > 1
    assert sum(score == 0 for _, score in calls) == expected


def test_new_solution_progress_follows_successful_write_and_best_save(tmp_path):
    search = TorchSearch(TorchSearchConfig(4, device="cpu", batch_size=4), tmp_path)
    reports = []
    def progress(state):
        reports.append(state.new_solutions)
        assert (tmp_path / "4.txt").read_text().count("\na=") == state.new_solutions
        assert json.loads(state.best_path.read_text())["score"] == 0
    search.run(seconds=0, progress=progress)
    assert reports and reports == list(range(1, search.new_solutions + 1))


def test_invalid_zero_score_never_writes_solution(tmp_path, monkeypatch):
    search = TorchSearch(TorchSearchConfig(44, device="cpu", batch_size=1), tmp_path)
    monkeypatch.setattr(search.model, "discrete_scores", lambda p: torch.zeros(p.shape[0]))
    with pytest.raises(RuntimeError, match="disagrees"):
        search.observe()
    assert not (tmp_path / "44.txt").exists()
