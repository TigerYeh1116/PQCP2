"""Exact ternary math, exhaustive small lifts, and optional pipeline admission."""

from itertools import product
import json
from types import SimpleNamespace

import pytest
import torch

from solver.compressed_fkm import lift_factor2_compressed
from solver.correlation import full_correlation_profile
from solver.structured_energy import integer_periodic_autocorrelation_profile as pacf, fold_profile, target_profile, compress_binary_sequence
from solver.target_profiles import canonical_target_content_profiles, pair_content, target_content_profiles
from solver.torch_compressed_search import (paired_contents, representative_lift, search_compressed,
                                            ternary_profile, ternary_swap_delta)
from solver.torch_hybrid_runner import TorchHybridConfig, _direct_compression_round


def targets(length, batch):
    ts = canonical_target_content_profiles(length)
    return tuple(ts[i % len(ts)] for i in range(batch))


@pytest.mark.parametrize("d", [3, 5])
def test_every_small_ternary_swap_matches_integer_oracle(d):
    words = list(product((-1, 0, 1), repeat=d))
    tensor = torch.tensor(words, dtype=torch.float32)
    old = torch.tensor([pacf(w) for w in words])
    for p, q in product(range(d), repeat=2):
        delta = ternary_swap_delta(tensor, torch.full((len(words),), p), torch.full((len(words),), q))
        changed = []
        for word in words:
            copy = list(word)
            copy[p], copy[q] = copy[q], copy[p]
            changed.append(pacf(copy))
        assert torch.equal(old + delta, torch.tensor(changed))
        assert torch.all(delta[:, 0] == 0)


def test_small_lifts_prove_content_conditions_in_both_directions():
    # Every word/content with d=3: no genuinely feasible lift is excluded.
    all_content = set()
    for c in product((-2, 0, 2), repeat=3):
        possible = set()
        for orientation in product((0, 1), repeat=c.count(0)):
            bits = lift_factor2_compressed(c, orientation)
            possible.add((sum(bits[::2]), sum(bits[1::2])))
        for e, o in product(range(4), repeat=2):
            feasible = 0 <= e-c.count(-2) <= c.count(0) and e+o == 2*c.count(-2)+c.count(0)
            assert feasible == ((e, o) in possible)
            if feasible:
                bits = representative_lift(c, e, o)
                assert compress_binary_sequence(bits, 2) == c
                assert (sum(bits[::2]), sum(bits[1::2])) == (e, o)
                all_content.add(((e, o), (c.count(-2), c.count(0), c.count(2))))
    for t in target_content_profiles(6):
        expected = {(a, b) for (ac, a) in all_content for (bc, b) in all_content
                    if ac == (t.a_even_ones, t.a_odd_ones) and bc == (t.b_even_ones, t.b_odd_ones)
                    and a[1]+b[1] == 3}
        assert set(paired_contents(t)) == expected


@pytest.mark.parametrize("length", [6, 14, 46, 86])
def test_search_trace_exact_content_and_determinism(length):
    ts = targets(length, 8)
    trace = []
    result = search_compressed(ts, device=torch.device("cpu"), seed=123, steps=16, candidates=3, trace=trace)
    counts = None
    for words, profiles in trace:
        hist = [[tuple(w.count(v) for v in (-1, 0, 1)) for w in pair] for pair in words]
        if counts is None:
            counts = hist
        assert counts == hist
        for pair, profile in zip(words, profiles):
            assert sum(v != 0 for w in pair for v in w) == length//2
            assert all(v in (-1, 0, 1) for w in pair for v in w)
            assert list(map(sum, zip(*(pacf(tuple(map(int, w))) for w in pair)))) == profile
    repeated = search_compressed(ts, device=torch.device("cpu"), seed=123, steps=16, candidates=3)
    assert torch.equal(result.words, repeated.words)
    assert result.accepted == repeated.accepted
    for job in result.jobs:
        assert tuple(map(sum, zip(pacf(job.compressed_a), pacf(job.compressed_b)))) == fold_profile(target_profile(length, job.target.k, job.target.eta), 2)
        assert pair_content(job.source_a, job.source_b) == (job.target.a_even_ones, job.target.a_odd_ones,
                                                           job.target.b_even_ones, job.target.b_odd_ones)


@pytest.mark.parametrize("device", ["mps", "cuda"])
def test_device_math(device):
    if not (torch.backends.mps.is_available() if device == "mps" else torch.cuda.is_available()):
        pytest.skip(device + " unavailable")
    a = search_compressed(targets(46, 8), device=torch.device(device), seed=11, steps=3, candidates=2)
    # Export already independently checks EVERY lane, including non-matches.
    assert a.proposals == 48


def test_pipeline_saves_exact_jobs_before_callback(tmp_path, monkeypatch):
    # Observation handoff, no solver invocation and no official files altered.
    import solver.torch_hybrid_runner as runner
    monkeypatch.setattr(runner, "verify_pqcp", lambda a, b: SimpleNamespace(
        is_valid=False, profile=tuple(full_correlation_profile(a, b))))
    seen = []
    def callback(a, b, metadata):
        assert list((tmp_path / "results/direct_compression").rglob("*.json"))
        seen.append((a, b, metadata))
    cfg = TorchHybridConfig(14, root=tmp_path, direct_compression=True,
                           direct_compression_batch=16, direct_compression_steps=32)
    count, verified, new = _direct_compression_round(cfg, torch.device("cpu"), 0, elite_callback=callback)
    assert count == len(seen) > 0 and verified == new == 0
    assert not (tmp_path / "14.txt").exists()
    snapshot = json.loads((tmp_path / "results/direct_compression" / cfg.direct_compression_session / "L14_seed123_cycle000000.json").read_text())
    assert snapshot["exact_jobs"] == count
    assert snapshot['support_attempts'] <= cfg.direct_compression_support_top
    assert snapshot['support_new_jobs'] >= 0
    assert list((tmp_path / 'results/direct_compression').rglob('L14_seed123_support*.json'))


def test_l86_norm_default_round_saves_records_and_metrics(tmp_path):
    from solver.search_metrics import SearchMetrics
    cfg = TorchHybridConfig(86, root=tmp_path, direct_compression=True,
        direct_compression_batch=8, direct_compression_steps=1,
        direct_compression_native_top=0, direct_compression_support_top=0,
        direct_compression_repair_top=0, direct_compression_transfer_top=0,
        direct_compression_mitm_top=0)
    metrics = SearchMetrics(tmp_path, 86, 123)
    try:
        _direct_compression_round(cfg, torch.device('cpu'), 0, metrics=metrics)
        directory = tmp_path/'results/direct_compression'/cfg.direct_compression_session
        records = json.loads((directory/'L86_seed123_cyclic000000.json').read_text())['attempts']
        assert len(records) == 1
        assert records[0]['method'] == 'cyclic_norm'
        assert len(records[0]['result']['generation']) >= 2
        assert {g['fixed_side'] for g in records[0]['result']['generation'][:2]} == {0, 1}
        assert len(records[0]['result']['attempts']) <= 8
        assert records[0]['result']['completion_scope'] == 'binary'
        snapshot = json.loads((directory/'L86_seed123_cycle000000.json').read_text())
        assert snapshot['cyclic_attempts'] == 1
        assert metrics.snapshot()['counts']['direct_compression.cyclic_attempts'] == 1
        assert snapshot['cyclic_binary_exclusions'] >= 0
        assert metrics.snapshot()['counts']['direct_compression.cyclic_full_verified'] == snapshot['cyclic_full_verified']
    finally:
        metrics.close()


def test_cli_rejects_inapplicable_modes_before_start(monkeypatch):
    import main
    monkeypatch.setattr(main, "_CCompletionBridge", lambda *a, **k: pytest.fail("started a solver"))
    for argv in (["--L", "44", "--torch-direct-compression"],
                 ["--L", "86", "--backend", "c", "--torch-direct-compression"],
                 ["--L", "86", "--z3-orientation-search", "--no-z3-orientation-preprocessing"]):
        assert main.main(argv) != 0
    assert not main.parse_args(["--L", "86"]).torch_direct_compression
    assert not main.parse_args(["--L", "86"]).z3_orientation_search
    assert main.parse_args(['--L', '86']).torch_direct_compression_support_top == 4
    assert main.parse_args(['--L', '86']).torch_direct_compression_cyclic_top == 1
    assert main.parse_args(['--L', '86']).torch_direct_compression_cyclic_max == 8


@pytest.mark.parametrize('kwargs', [dict(direct_compression_support_top=-1),
    dict(direct_compression_support_top=True), dict(direct_compression_support_max=0),
    dict(direct_compression_support_max=4097), dict(direct_compression_sign_assignments=0),
    dict(direct_compression_sign_assignments=2**31+1), dict(direct_compression_cyclic_top=-1),
    dict(direct_compression_cyclic_top=True), dict(direct_compression_cyclic_max=0),
    dict(direct_compression_cyclic_max=4097)])
def test_invalid_support_config(kwargs):
    with pytest.raises(ValueError):
        TorchHybridConfig(86, **kwargs)
