"""Exact invariants, baseline isolation and pre-cutoff callback integration."""

from itertools import product
from types import SimpleNamespace
import threading

import pytest
import torch

from solver.correlation import full_correlation_profile
from solver.objective import pqcp_objective
from solver.target_profiles import target_content_profiles
from solver.torch_alternating import orientation_proposals, alternating_polish_stream, fold_error
from solver.torch_polish import polish_batch, discrete_energy
from solver.torch_search import BatchedPQCP
from solver.torch_hybrid_runner import TorchHybridConfig, _polish_continuous_records


def sample(length, batch=4):
    profiles = target_content_profiles(length, decimation_reduced=True)
    model = BatchedPQCP(tuple(profiles[:1]) * batch, torch.device("cpu"))
    signs = model.project(torch.randn(batch, 2, length, generator=torch.Generator().manual_seed(42)))
    return model, signs


def exact(signs):
    a, b = ((1 - signs) / 2).to(torch.int32).tolist()
    return full_correlation_profile(a, b)


@pytest.mark.parametrize("length", [6, 10, 14, 46, 86])
def test_proposals_preserve_compression_content_and_exact_delta(length):
    # Arbitrary binary words, not known solutions or handpicked favorable seeds.
    g = torch.Generator().manual_seed(333)
    signs = 1 - 2 * torch.randint(0, 2, (8, 2, length), generator=g).float()
    draws = torch.rand(8, 16, 4, generator=g)
    proposed, delta, legal = orientation_proposals(signs, draws)
    d = length // 2
    assert legal.any()
    for lane, word in enumerate(signs):
        original = torch.tensor(exact(word))
        for candidate, change, ok in zip(proposed[lane], delta[lane], legal[lane]):
            assert torch.equal(candidate[:, :d] + candidate[:, d:], word[:, :d] + word[:, d:])
            assert torch.equal(candidate[:, ::2].sum(-1), word[:, ::2].sum(-1))
            assert torch.equal(candidate[:, 1::2].sum(-1), word[:, 1::2].sum(-1))
            assert torch.equal(torch.tensor(exact(candidate)), original + change)
            assert (candidate != word).sum() == (4 if ok else 0)


def test_all_small_words_and_empty_pools():
    words = torch.tensor(list(product((-1., 1.), repeat=6)))
    pairs = torch.stack((words, words), 1)
    # All ranks for both selected sequences; covers every legal pair for L=6.
    draws = torch.tensor([[s, p / 3, q / 3, 0.] for s in (0., .5)
                          for p in range(3) for q in range(3)])
    proposals, deltas, legal = orientation_proposals(pairs, draws[None].repeat(64, 1, 1))
    for i, word in enumerate(pairs):
        for j, candidate in enumerate(proposals[i]):
            assert torch.equal(torch.tensor(exact(candidate)), torch.tensor(exact(word)) + deltas[i, j])
    assert not legal[0].any() and not legal[-1].any()
    assert torch.equal(proposals[0], pairs[0].expand_as(proposals[0]))


def test_tracking_does_not_change_baseline_trajectory():
    model, signs = sample(46)
    draws = torch.rand(10, 4, 3, 4, generator=torch.Generator().manual_seed(55))
    traces = [[], []]
    a = polish_batch(model, signs, draws, proposal_policy="opposite", trace=traces[0])
    b = polish_batch(model, signs, draws, proposal_policy="opposite", trace=traces[1], track_compression=True)
    assert traces[0] == traces[1]
    assert torch.equal(a.signs, b.signs)
    assert a.accepted_swaps == b.accepted_swaps
    assert torch.equal(model.correlation(b.compression_signs), b.compression_profile)


def test_alternating_deterministic_trace_and_phase_invariants():
    model, signs = sample(86)
    traces = [[], []]
    results = [alternating_polish_stream(model, signs, steps=12, candidates=3, seed=22,
                                         phase_steps=3, trace=t) for t in traces]
    assert traces[0] == traces[1]
    assert [phase for phase, _ in traces[0]] == ["support", "direction"] * 2
    previous = signs
    for phase, steps in traces[0]:
        for current, profile, _, _ in steps:
            current = torch.tensor(current)
            assert torch.equal(model.correlation(current), torch.tensor(profile))
            if phase == "direction":
                assert torch.equal(current[..., :43] + current[..., 43:], previous[..., :43] + previous[..., 43:])
            previous = current
    result = results[0]
    assert result.direction_proposals == 6 * 4 * 3
    for word, profile, score in zip(result.signs, result.profile, result.scores):
        assert exact(word) == profile.tolist()
        assert pqcp_objective(exact(word)) == score
    assert torch.all(fold_error(result.compression_profile, model.targets) <= fold_error(model.correlation(signs), model.targets))
    for word, profile in zip(result.compression_signs, result.compression_profile):
        assert exact(word) == profile.tolist()


@pytest.mark.parametrize("device", ["mps", "cuda"])
def test_device_direction_delta_matches_cpu(device):
    available = torch.backends.mps.is_available() if device == "mps" else torch.cuda.is_available()
    if not available:
        pytest.skip(device + " unavailable")
    g = torch.Generator().manual_seed(84)
    signs = 1 - 2 * torch.randint(0, 2, (4, 2, 86), generator=g).float()
    draws = torch.rand(4, 8, 4, generator=g)
    expected = orientation_proposals(signs, draws)
    actual = orientation_proposals(signs.to(device), draws.to(device))
    for a, b in zip(expected, actual):
        assert torch.equal(a, b.cpu())


def test_cancel_before_work_and_invalid_length():
    model, signs = sample(46)
    stop = threading.Event()
    stop.set()
    result = alternating_polish_stream(model, signs, steps=4, candidates=2, seed=4, stop_event=stop)
    assert result.proposals == 0 and torch.equal(result.signs, signs)
    with pytest.raises(ValueError, match="L=2 mod 4"):
        TorchHybridConfig(44, polish_mode="alternating")


def test_hybrid_exports_compression_candidates_before_seed_cutoff(tmp_path):
    model, signs = sample(46)
    profiles = model.correlation(signs)
    records = []
    for i, word in enumerate(signs):
        a, b = ((1 - word) / 2).int().tolist()
        target = model.profiles[i]
        records.append(dict(A=a, B=b, k=target.k, sign=target.eta,
                            score=pqcp_objective(exact(word)), iteration=0,
                            target_energy=int(discrete_energy(profiles[i:i+1], model.targets[i:i+1])[0])))
    seen = []
    output, _, _ = _polish_continuous_records(
        TorchHybridConfig(46, root=tmp_path, polish_mode="alternating", polish_steps=4,
                          polish_candidates=2, polish_phase_steps=2, target_seed_count=1),
        SimpleNamespace(device=torch.device("cpu"), epoch=0), records, 0,
        elite_callback=lambda a, b, meta: seen.append((a, b, meta)),
    )
    assert len(output) <= 1
    assert len(seen) == 8  # objective and compression best for every lane
    for a, b, meta in seen:
        assert meta["score"] == pqcp_objective(full_correlation_profile(a, b))
