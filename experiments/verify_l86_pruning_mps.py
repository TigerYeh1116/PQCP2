"""Audit real MPS traces against integer PACF and the current content proofs."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from solver.l86_binary_content import binary_content_bounds, l86_binary_content_proof
from solver.l86_content_certificate import l86_content_certificate
from solver.structured_energy import integer_periodic_autocorrelation_profile as pacf
from solver.target_profiles import canonical_target_content_profiles
from solver.torch_compressed_search import search_compressed
from solver.torch_hybrid_runner import TorchHybridConfig, _direct_compression_round
from solver.torch_search import TorchSearch, TorchSearchConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if not torch.backends.mps.is_available():
        raise RuntimeError('Real MPS required; no CPU fallback')
    certificate = l86_content_certificate()
    assert certificate.verified and certificate.minimum_weight == 8
    binary = l86_binary_content_proof()
    assert binary.verified
    targets = canonical_target_content_profiles(86)
    profiles = tuple(targets[i % len(targets)] for i in range(32))
    trace, started = [], perf_counter()
    result = search_compressed(profiles, device=torch.device('mps'), seed=789,
        steps=64, transfer_interval=2, trace=trace)
    minimum = 43
    for pairs, observed in trace:
        for pair, profile, target in zip(pairs, observed, profiles):
            pair = tuple(tuple(map(int, row)) for row in pair)
            assert all(lo <= row.count(-1) <= hi for row, (lo, hi) in zip(pair, binary_content_bounds(target)))
            weights = tuple(sum(z*z for z in row) for row in pair)
            minimum = min(minimum, *weights)
            assert min(weights) >= 8 and sum(weights) == 43
            assert tuple(profile) == tuple(a+b for a, b in zip(*(pacf(row) for row in pair)))
    assert len(trace) == 64 and result.transfer_accepted > 0
    config = TorchHybridConfig(L=86, root=args.output, direct_compression=True,
        direct_compression_batch=8, direct_compression_steps=1,
        direct_compression_repair_top=0, direct_compression_transfer_top=0,
        direct_compression_mitm_top=0, direct_compression_native_top=0,
        direct_compression_support_top=0, direct_compression_cyclic_top=1,
        direct_compression_cyclic_max=8, direct_compression_session='verification')
    direct = _direct_compression_round(config, torch.device('mps'), 0)
    observation_checks = []
    for compression in (False, True):
        cfg = TorchSearchConfig(86, device='mps', batch_size=8, elite_count=8,
            archive_size=128, symmetry_pruning=True, continuous_kernel='dft',
            fast_observation=True, compression=compression)
        old = TorchSearch(replace(cfg, half_shift_observation_projection=False),
                          args.output/('raw_'+str(compression)))
        new = TorchSearch(cfg, args.output/('projected_'+str(compression)))
        values = torch.randn(old.theta.shape, generator=torch.Generator().manual_seed(16686)).to('mps')
        with torch.no_grad():
            old.theta.copy_(values); new.theta.copy_(values)
        for step in range(16):
            old.step(); new.step()
            if (step+1) % 4 == 0:
                old.observe(); new.observe()
        assert torch.equal(old.theta, new.theta)
        assert old.lane_best == new.lane_best and old.last_improved == new.last_improved
        projected = [p for p in new.archive.values() if p.get('candidate_construction') == 'minimum_hamming_half_shift']
        assert projected and new.candidate_evaluations > old.candidate_evaluations
        assert all(p['profile'][43] == 0 for p in projected)
        observation_checks.append(dict(compression=compression, projected_archived=len(projected),
            theta_identical=True, integer_payloads_audited=True))
    report = dict(device='mps', lanes=32, steps=64, trace_frames=len(trace),
        minimum_observed_support=minimum, proven_minimum_support=8,
        transfer_accepted=result.transfer_accepted, integer_profiles_audited=True,
        content_cases=sum(case[3] for case in certificate.recorded_cases)+6880,
        binary_cases=binary.cases, direct_round_result=direct,
        continuous_observation_checks=observation_checks,
        elapsed=perf_counter()-started, discovery_rate_not_measured=True)
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
