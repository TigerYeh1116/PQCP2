"""Exact swap math and bounded device polish, without timed search benchmarks."""

import itertools
import random

import pytest

torch = pytest.importorskip("torch")

from solver.correlation import full_correlation_profile, periodic_autocorrelation
from solver.objective import pqcp_objective
from solver.target_profiles import TargetContentProfile, pair_content, target_content_profiles
from solver.torch_polish import closest_content_target, discrete_energy, polish_batch, polish_stream, swap_correlation_delta
from solver.structured_energy import structured_energy_breakdown
from solver.torch_search import BatchedPQCP


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("real CUDA unavailable")
    return torch.device(request.param)


@pytest.mark.parametrize("length", [1, 2, 3, 4, 5, 6, 8, 44, 46, 68, 94])
def test_two_flip_delta_handles_shared_terms_half_shift_and_same_position(length, device):
    rng = random.Random(length)
    rows = []
    if length <= 4:
        for bits in itertools.product((0, 1), repeat=length):
            rows.extend((bits, p, q) for p in range(length) for q in range(length))
    else:
        for _ in range(20):
            bits = tuple(rng.randrange(2) for _ in range(length))
            rows.extend((bits, 0, q) for q in (0, 1, length // 2, length - 1))
    sequence = torch.tensor([[1 - 2 * x for x in bits] for bits, _, _ in rows], dtype=torch.float32, device=device)
    p = torch.tensor([p for _, p, _ in rows], device=device)
    q = torch.tensor([q for _, _, q in rows], device=device)
    actual = swap_correlation_delta(sequence, p, q).cpu().tolist()
    for (bits, p, q), delta in zip(rows, actual):
        after = list(bits)
        after[p] ^= 1
        after[q] ^= 1
        assert delta == [periodic_autocorrelation(after, u) - periodic_autocorrelation(bits, u) for u in range(length)]


@pytest.mark.parametrize("length", [4, 8, 44, 46])
@pytest.mark.parametrize("candidate_count,policy", [(1, "positions"), (4, "positions"), (1, "opposite"), (4, "opposite")])
def test_every_polish_step_keeps_exact_pacf_and_parity_content(length, device, candidate_count, policy):
    profiles = target_content_profiles(length)[:4]
    model = BatchedPQCP(profiles, device)
    generator = torch.Generator().manual_seed(length)
    initial = model.project(torch.rand(len(profiles), 2, length, generator=generator).to(device))
    draws = torch.rand(12, len(profiles), candidate_count, 4, generator=generator).to(device)
    trace = []
    result = polish_batch(model, initial, draws, trace=trace, proposal_policy=policy)
    for current, current_profile, best, best_profile in trace:
        for states, pacfs in ((current, current_profile), (best, best_profile)):
            for pair, profile, required in zip(states, pacfs, profiles):
                a, b = [tuple(int((1 - x) / 2) for x in seq) for seq in pair]
                assert profile == full_correlation_profile(a, b)
                assert pair_content(a, b) == (required.a_even_ones, required.a_odd_ones, required.b_even_ones, required.b_odd_ones)
    assert torch.equal(result.profile, model.correlation(result.signs))
    assert torch.equal(result.profile, result.profile.round())
    assert result.scores.cpu().tolist() == [pqcp_objective(list(map(int, p))) for p in result.profile.cpu().tolist()]
    assert 0 <= result.accepted_swaps <= result.legal_swaps <= result.proposals
    repeated = polish_batch(model, initial, draws, proposal_policy=policy)
    assert torch.equal(result.signs, repeated.signs)
    assert torch.equal(result.profile, repeated.profile)
    assert result.proposals == 12 * len(profiles) * candidate_count


def test_opposite_pool_avoids_same_sign_rejections(device):
    profiles = target_content_profiles(44)[:4]
    model = BatchedPQCP(profiles, device)
    generator = torch.Generator().manual_seed(244)
    initial = model.project(torch.rand(4, 2, 44, generator=generator).to(device))
    draws = torch.rand(5, 4, 4, 4, generator=generator).to(device)
    result = polish_batch(model, initial, draws, proposal_policy="opposite")
    assert result.scores.min().item() > 0  # no solved/frozen lanes in this fixture
    assert result.legal_swaps == result.proposals


@pytest.mark.parametrize("compression", [False, True])
def test_kicks_preserve_full_pacf_and_content(device, compression):
    profiles = target_content_profiles(44)[:4]
    model = BatchedPQCP(profiles, device)
    generator = torch.Generator().manual_seed(344)
    initial = model.project(torch.rand(4, 2, 44, generator=generator).to(device))
    draws = torch.rand(40, 4, 4, 4, generator=generator).to(device)
    trace = []
    result = polish_batch(model, initial, draws, proposal_policy="opposite", kick_interval=1,
                          compression=compression, trace=trace)
    assert result.kicks > 0 and result.forced_swaps > 0
    assert result.forced_swaps <= result.accepted_swaps
    for current, profile, best, best_profile in trace:
        for states, profiles_actual in ((current, profile), (best, best_profile)):
            for pair, actual, p in zip(states, profiles_actual, profiles):
                a, b = [tuple(int((1 - x) / 2) for x in seq) for seq in pair]
                assert actual == full_correlation_profile(a, b)
                assert pair_content(a, b) == (p.a_even_ones, p.a_odd_ones, p.b_even_ones, p.b_odd_ones)


def test_solution_at_other_target_shift_freezes_and_survives_polish(device):
    a = tuple(map(int, "10100001100011100100011010001011000010011111"))
    b = tuple(map(int, "01010100011010100100111110011010010000000100"))
    profile = TargetContentProfile(44, 2, -1, *pair_content(a, b))
    model = BatchedPQCP([profile], device)
    signs = 1 - 2 * torch.tensor([[a, b]], dtype=torch.float32, device=device)
    assert discrete_energy(model.correlation(signs), model.targets).item() > 0
    uniforms = torch.rand(8, 1, 4, generator=torch.Generator().manual_seed(5)).to(device)
    result = polish_batch(model, signs, uniforms)
    assert result.scores.item() == 0
    assert torch.equal(result.signs, signs)
    assert result.legal_swaps == result.accepted_swaps == 0


def test_polish_rejects_nonbinary_and_bad_randomness():
    model = BatchedPQCP(target_content_profiles(8)[:1], torch.device("cpu"))
    signs = torch.ones(1, 2, 8)
    with pytest.raises(ValueError, match="binary"):
        polish_batch(model, signs * 0.5, torch.zeros(1, 1, 4))
    with pytest.raises(ValueError, match="uniforms"):
        polish_batch(model, signs, torch.ones(1, 1, 4))


def test_streamed_polish_is_deterministic_and_memory_bounded():
    profiles = target_content_profiles(8)[:2]
    model = BatchedPQCP(profiles, torch.device("cpu"))
    signs = model.project(torch.zeros(2, 2, 8))
    first = polish_stream(model, signs, steps=12, candidates=2, seed=91,
                          block_steps=4, kick_interval=4)
    second = polish_stream(model, signs, steps=12, candidates=2, seed=91,
                           block_steps=4, kick_interval=4)
    assert first.proposals <= 48  # an exact solution may stop the stream early
    assert torch.equal(first.signs, second.signs)
    assert torch.equal(first.profile, second.profile)
    assert torch.equal(first.scores, second.scores)
    assert first.current_signs is not None


def test_streamed_polish_rejects_block_shorter_than_kick_interval():
    profiles = target_content_profiles(8)[:1]
    model = BatchedPQCP(profiles, torch.device("cpu"))
    signs = model.project(torch.zeros(1, 2, 8))
    with pytest.raises(ValueError, match="at least kick_interval"):
        polish_stream(model, signs, steps=10, candidates=1, seed=1,
                      block_steps=2, kick_interval=3)


def test_fixed_center_retarget_does_not_mistake_decimation_representatives_for_all_targets():
    a = tuple(map(int, "10100001100011100100011010001011000010011111"))
    b = tuple(map(int, "01010100011010100100111110011010010000000100"))
    target = closest_content_target(a, b)
    assert target.k == 20 and target.eta == -1
    assert target not in target_content_profiles(44, decimation_reduced=True)
    assert pair_content(a, b) == (target.a_even_ones, target.a_odd_ones, target.b_even_ones, target.b_odd_ones)


@pytest.mark.parametrize("compression", [False, True])
def test_retarget_minimizes_existing_energy_over_all_compatible_targets(compression):
    profiles = target_content_profiles(8)[:8]
    model = BatchedPQCP(profiles, torch.device("cpu"))
    signs = model.project(torch.rand(8, 2, 8, generator=torch.Generator().manual_seed(988)))
    for a, b in ((1 - signs) / 2).to(torch.int32).tolist():
        target = closest_content_target(a, b, compression=compression)
        actual = full_correlation_profile(a, b)
        weights = {2: 2, 4: 4} if compression else {}
        def energy(p):
            return structured_energy_breakdown(actual, p.k, p.eta, tuple(weights)).weighted_total(weights)
        compatible = [p for p in target_content_profiles(8)
                      if (p.a_even_ones, p.a_odd_ones, p.b_even_ones, p.b_odd_ones) == pair_content(a, b)]
        assert target == min(compatible, key=lambda p: (energy(p), p))
