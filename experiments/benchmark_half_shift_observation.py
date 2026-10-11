"""Paired formation replay; eligible elite throughput is only a proxy metric."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import statistics
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from solver.l86_binary_content import binary_content_bounds
from solver.target_profiles import TargetContentProfile
from solver.torch_search import TorchSearch, TorchSearchConfig
from solver.torch_hybrid_runner import _continuous_seed_records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'mps'), default='mps')
    parser.add_argument('--batch', type=int, default=512)
    parser.add_argument('--steps', type=int, default=64)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--observation-interval', type=int, default=25)
    parser.add_argument('--fast-observation', action='store_true')
    parser.add_argument('--seed-limit', type=int, default=128)
    parser.add_argument('--tie-method', choices=('plain', 'energy-aware'), default='energy-aware')
    parser.add_argument('--seed', type=int, default=20261007)
    parser.add_argument('--cache-comparison', action='store_true', help='both sides enable projection; compare exact memoization only')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = TorchSearchConfig(86, device=args.device, seed=args.seed,
        batch_size=args.batch, steps_per_restart=max(1000, args.steps),
        observation_interval=args.observation_interval, continuous_kernel='dft', loss_mode='balanced',
        loss_backend='profile', symmetry_pruning=True, frequency_pruning=True,
        fast_observation=args.fast_observation,
        half_shift_projection_energy=args.tie_method == 'energy-aware')
    def sync():
        if args.device == 'mps':
            torch.mps.synchronize()
    # Warm both code paths and mathematical proofs before paired measurements.
    for enabled in (False, True):
        variant = dict(half_shift_projection_cache=enabled) if args.cache_comparison else dict(half_shift_observation_projection=enabled)
        warm = TorchSearch(replace(config, **variant), args.output/'warm'/str(enabled))
        warm.step(); warm.observe(); sync()
    rows = []
    for repetition in range(args.repeats):
        previous_theta = None
        for enabled in ((False, True) if repetition % 2 == 0 else (True, False)):
            variant = dict(half_shift_projection_cache=enabled) if args.cache_comparison else dict(half_shift_observation_projection=enabled)
            search = TorchSearch(replace(config, **variant),
                                 args.output/('run{}_{}'.format(repetition, enabled)))
            sync()
            started, observation_seconds = perf_counter(), 0
            for step in range(args.steps):
                search.step()
                if (step+1) % config.observation_interval == 0 or step+1 == args.steps:
                    sync(); observing = perf_counter()
                    search.observe(); sync()
                    observation_seconds += perf_counter()-observing
            sync()
            elapsed = perf_counter()-started
            if previous_theta is not None and not torch.equal(previous_theta, search.theta.detach()):
                raise RuntimeError('Observation supplementation changed the Adam trajectory')
            previous_theta = search.theta.detach().clone()
            def eligible_payload(payload):
                target = TargetContentProfile(86, payload['k'], payload['sign'],
                    sum(payload['A'][::2]), sum(payload['A'][1::2]),
                    sum(payload['B'][::2]), sum(payload['B'][1::2]))
                counts = tuple(sum(w[i] == w[i+43] == 1 for i in range(43))
                               for w in (payload['A'], payload['B']))
                bounds = binary_content_bounds(target)
                return int(payload['profile'][43] == 0 and
                    (bounds is None or all(lo <= n <= hi for n, (lo, hi) in zip(counts, bounds))))
            eligible = sum(eligible_payload(payload) for payload in search.archive.values())
            projected = sum(payload.get('candidate_construction') == 'minimum_hamming_half_shift'
                            for payload in search.archive.values())
            exporting = perf_counter()
            seeds = _continuous_seed_records(search, args.seed_limit)
            export_seconds = perf_counter()-exporting
            eligible_seeds = sum(eligible_payload(payload) for payload in seeds)
            row = dict(repetition=repetition, enabled=enabled, elapsed=elapsed,
                observation_seconds=observation_seconds, archived=len(search.archive),
                eligible_archived=eligible, projected_archived=projected,
                best_score=search.best['score'], candidate_evaluations=search.candidate_evaluations,
                exported_seeds=len(seeds), eligible_exported_seeds=eligible_seeds,
                export_seconds=export_seconds,
                projection_cache_hits=search.half_shift_projection_cache_hits,
                projection_cache_misses=search.half_shift_projection_cache_misses,
                fully_verified_hits=search.verified_hits, new_solutions=search.new_solutions)
            rows.append(row)
            print(json.dumps(row), flush=True)
    summaries = {}
    for enabled in (False, True):
        subset = [r for r in rows if r['enabled'] == enabled]
        summaries[str(enabled)] = dict(median_seconds=statistics.median(r['elapsed'] for r in subset),
            median_eligible_elites=statistics.median(r['eligible_archived'] for r in subset),
            median_eligible_elites_per_second=statistics.median(r['eligible_archived']/r['elapsed'] for r in subset),
            median_eligible_exported_seeds=statistics.median(r['eligible_exported_seeds'] for r in subset),
            median_eligible_exported_seeds_per_second=statistics.median(
                r['eligible_exported_seeds']/(r['elapsed']+r['export_seconds']) for r in subset))
    report = dict(device=args.device, batch=args.batch, steps=args.steps,
        tie_method=args.tie_method, observation_interval=args.observation_interval,
        seed=args.seed, repeats=args.repeats,
        cache_comparison=args.cache_comparison,
        fast_observation=args.fast_observation, seed_limit=args.seed_limit, rows=rows,
        summary=summaries, theta_trajectory_identical=True,
        limitation='Eligible elites are a proxy, not new PQCP discoveries. No discovery speed claim.')
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(summaries), flush=True)


if __name__ == '__main__':
    main()
