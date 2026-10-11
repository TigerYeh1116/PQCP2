"""Scope/authentication gates and the independent alternating-modulation identity."""
from dataclasses import replace
import hashlib
from itertools import product
import json
from pathlib import Path

import pytest
import torch

import solver.l86_binary_content as module
from solver.l86_binary_content import binary_content_bounds, read_recorded_binary_content_proof
from solver.structured_energy import compress_binary_sequence, integer_periodic_autocorrelation_profile as pacf
from solver.target_profiles import canonical_target_content_profiles
from solver.torch_compressed_search import paired_contents, search_compressed, search_paired_contents


@pytest.fixture
def proof_root(tmp_path):
    records = json.loads((Path(__file__).parent/'fixtures/l86_binary_content_runs.json').read_text())
    for name, raw in records.items():
        path = tmp_path/'results'/name/'report.json'
        path.parent.mkdir(parents=True)
        path.write_text(raw)
    return tmp_path


@pytest.fixture
def authenticated_proof(proof_root, monkeypatch):
    report = read_recorded_binary_content_proof(proof_root)
    assert report.verified and report.cases == 5692800
    monkeypatch.setattr(module, 'l86_binary_content_proof', lambda: report)
    return report


def test_recorded_complete_coverage_is_authenticated_and_missing_or_modified_records_disable_it(proof_root):
    assert read_recorded_binary_content_proof(proof_root).verified
    name = module.RECORDS[-1][0]
    path = proof_root/'results'/name/'report.json'
    original = path.read_bytes()
    path.write_bytes(original+b' ')
    assert not read_recorded_binary_content_proof(proof_root).verified
    path.unlink()
    assert not read_recorded_binary_content_proof(proof_root).verified


def test_even_authenticated_but_wrong_scope_is_rejected(proof_root, monkeypatch):
    record = module.RECORDS[0]
    path = proof_root/'results'/record[0]/'report.json'
    data = json.loads(path.read_text())
    data['summary']['complete'] = False
    raw = json.dumps(data).encode()
    path.write_bytes(raw)
    replacement = (record[0], hashlib.sha256(raw).hexdigest())+record[2:]
    monkeypatch.setattr(module, 'RECORDS', (replacement,)+module.RECORDS[1:])
    assert not read_recorded_binary_content_proof(proof_root).verified


def test_canonical_target_ranges_keep_coarse_and_binary_proofs_separate(authenticated_proof):
    for target in canonical_target_content_profiles(86):
        raw = paired_contents(target)
        coarse = search_paired_contents(target, binary_content_pruning=False)
        full = search_paired_contents(target)
        from solver.l86_content_certificate import l86_content_certificate
        minimum = l86_content_certificate().minimum_weight
        assert len(coarse) == (14 if minimum == 8 else 15)
        assert len(full) == (13 if target.k % 2 or (minimum == 8 and target.eta == 1) else 14)
        bounds = binary_content_bounds(target)
        assert full == tuple(pair for pair in coarse if all(lo <= c[0] <= hi for c, (lo, hi) in zip(pair, bounds)))
        assert search_paired_contents(target, content_pruning=False) == raw
        # Every target in the same original-shift parity orbit is covered.
        for k in range(1, 43):
            if k % 2 == target.k % 2:
                assert binary_content_bounds(replace(target, k=k)) == bounds
    target = canonical_target_content_profiles(86)[0]
    flipped = replace(target, a_even_ones=43-target.a_even_ones, a_odd_ones=43-target.a_odd_ones)
    assert binary_content_bounds(flipped) is None


def test_no_pruning_without_authenticated_records_or_the_separate_coarse_bound(monkeypatch):
    target = canonical_target_content_profiles(86)[0]
    monkeypatch.setattr(module, 'l86_binary_content_proof', lambda: module.RecordedBinaryContentProof(False, 0, (), 'missing'))
    assert binary_content_bounds(target) is None
    assert search_paired_contents(target) == search_paired_contents(target, binary_content_pruning=False)
    monkeypatch.setattr('solver.l86_content_certificate.l86_content_certificate',
                        lambda: type('Weak', (), dict(verified=False, minimum_weight=5))())
    assert binary_content_bounds(target) is None


@pytest.mark.parametrize('L', [6, 14])
def test_all_original_rows_verify_modulation_and_content_map(L):
    d = L//2
    for bits in product((0, 1), repeat=L):
        modulated = tuple(v ^ (i % 2) for i, v in enumerate(bits))
        signs, transformed = tuple(1-2*v for v in bits), tuple(1-2*v for v in modulated)
        c = tuple(v//2 for v in compress_binary_sequence(bits, 2))
        new_c = tuple(v//2 for v in compress_binary_sequence(modulated, 2))
        g = tuple((-1 if i % 2 else 1)*(signs[i]-signs[i+d])//2 for i in range(d))
        assert new_c == g
        assert new_c.count(-1) == sum(bits[::2])-c.count(-1)
        assert tuple(pacf(transformed)) == tuple((-1 if u % 2 else 1)*v for u, v in enumerate(pacf(signs)))


def test_gpu_initialization_and_transfers_preserve_full_proven_content_ranges(authenticated_proof):
    targets = canonical_target_content_profiles(86)
    profiles = tuple(targets[i % len(targets)] for i in range(32))
    trace = []
    result = search_compressed(profiles, device=torch.device('cpu'), seed=789,
                               steps=64, transfer_interval=2, trace=trace)
    assert result.transfer_accepted > 0
    for pairs, observed in trace:
        for pair, profile, target in zip(pairs, observed, profiles):
            bounds = binary_content_bounds(target)
            assert all(lo <= w.count(-1) <= hi for w, (lo, hi) in zip(pair, bounds))
            assert tuple(profile) == tuple(x+y for x, y in zip(*(pacf(tuple(map(int, w))) for w in pair)))
            assert sum(v*v for w in pair for v in w) == 43
