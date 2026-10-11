"""Discovery accounting rejects incomplete candidates and respects symmetries."""
from dataclasses import replace
from pathlib import Path

import pytest

from main import parse_args
from solver.torch_hybrid_runner import TorchHybridConfig, _new_continuous_search
from experiments.benchmark_l86_discovery import freeze_sources, verified_inventory


def test_default_and_explicit_rollback_reach_actual_hybrid_search(tmp_path):
    args = parse_args(['--L', '86'])
    assert args.torch_half_shift_observation_projection and args.torch_half_shift_projection_energy
    args = parse_args(['--L', '86', '--no-torch-half-shift-observation-projection',
                       '--no-torch-half-shift-projection-energy'])
    config = TorchHybridConfig(86, root=tmp_path, device='cpu', continuous_batch_size=4,
        continuous_half_shift_observation_projection=args.torch_half_shift_observation_projection,
        continuous_half_shift_projection_energy=args.torch_half_shift_projection_energy)
    search = _new_continuous_search(config)
    assert not search.config.half_shift_observation_projection
    assert not search.config.half_shift_projection_energy
    with pytest.raises(ValueError):
        replace(config, continuous_half_shift_observation_projection=1)


def test_frozen_snapshot_contains_authenticated_content_evidence(tmp_path):
    from solver.l86_content_certificate import WEIGHT7_RECORDS
    from solver.l86_binary_content import read_recorded_binary_content_proof
    import hashlib
    snapshot = tmp_path/'source'
    assert len(freeze_sources(snapshot)) == 64
    for name, digest, *_ in WEIGHT7_RECORDS:
        assert hashlib.sha256((snapshot/'results'/name).read_bytes()).hexdigest() == digest
    assert read_recorded_binary_content_proof(snapshot).verified
    assert (snapshot/'main.py').exists() and (snapshot/'csrc/pqcp_search.c').exists()


def test_zero_inventory_is_valid_and_unverified_pairs_cannot_count():
    assert verified_inventory(b'') == set()
    with pytest.raises(ValueError):
        verified_inventory(('a='+'0'*86+'\nb='+'0'*86+'\n').encode())
    with pytest.raises(ValueError):
        verified_inventory(b'a=0011\n')
