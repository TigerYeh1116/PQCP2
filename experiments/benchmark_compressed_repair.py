"""Fresh ternary search -> exact coarse repair -> existing exact binary lift.

Never reads known PQCPs as seeds; saves every attempted reference and status.
No production L.txt is written. A bounded failure is not nonexistence.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--L', type=int, default=86)
    parser.add_argument('--seeds', type=int, nargs='+', default=[123, 456])
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--steps', type=int, default=2048)
    parser.add_argument('--top', type=int, default=2)
    parser.add_argument('--radii', type=int, nargs='+', default=[4, 8])
    parser.add_argument('--timeout-ms', type=int, default=5000)
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='mps')
    parser.add_argument('--radius4-backend', choices=['native', 'z3'], default='native')
    args = parser.parse_args()
    if args.L < 6 or args.L % 4 != 2:
        parser.error('--L must be twice an odd integer >=3')
    if min(args.batch, args.steps, args.top, args.timeout_ms) <= 0:
        parser.error('--batch, --steps, --top and --timeout-ms must be positive')
    if args.top > args.batch or any(r < 0 or r > args.L for r in args.radii):
        parser.error('--top must not exceed --batch; radii must lie in 0..L')
    if args.radius4_backend == 'native' and 4 in args.radii and args.L > 94:
        parser.error('native radius-four repair requires L<=94')
    args.output.mkdir(parents=True, exist_ok=False)
    from solver.torch_device import resolve_torch_device
    from solver.torch_compressed_search import search_compressed
    from solver.target_profiles import canonical_target_content_profiles
    from solver.compressed_repair import repair_compressed_pair
    from solver.compressed_lift import solve_compressed_lift
    from solver.ternary_repair_native import repair_two_swaps
    device = resolve_torch_device(args.device)
    profiles = canonical_target_content_profiles(args.L)
    if not profiles:
        raise ValueError('no feasible target/content profiles')
    targets = tuple(profiles[i % len(profiles)] for i in range(args.batch))
    rows = []
    for seed in args.seeds:
        started = perf_counter()
        generated = search_compressed(targets, device=device, seed=seed, steps=args.steps)
        generation_seconds = perf_counter()-started
        energies = generated.energies.int().cpu().tolist()
        candidates = generated.words.int().cpu().tolist()
        row = dict(seed=seed, generation_seconds=generation_seconds, proposals=generated.proposals,
                   minimum_energy=min(energies), exact_jobs=len(generated.jobs), attempts=[])
        rows.append(row)
        for lane in sorted(range(args.batch), key=lambda i: (energies[i], i))[:args.top]:
            a, b = candidates[lane]
            for radius in args.radii:
                evaluations = None
                if radius == 4 and args.radius4_backend == 'native':
                    result, evaluations = repair_two_swaps(a, b, targets[lane])
                else:
                    result = repair_compressed_pair(a, b, targets[lane], radius=radius,
                                                    timeout_ms=args.timeout_ms)
                attempt = dict(lane=lane, original_energy=energies[lane], target=asdict(targets[lane]),
                               reference=[a, b], result=asdict(result), evaluations=evaluations)
                row['attempts'].append(attempt)
                if result.words is not None:
                    ca, cb = (tuple(2*x for x in word) for word in result.words)
                    lift = solve_compressed_lift(ca, cb, targets[lane].k, targets[lane].eta,
                                                args.timeout_ms, targets[lane])
                    attempt['lift'] = asdict(lift)
                (args.output/'report.json').write_text(json.dumps(dict(config=vars(args), runs=rows),
                                                       indent=2, default=str))
                print('seed={} E={} radius={} status={} solve={:.3f}s lift={}'.format(
                    seed, energies[lane], radius, result.status, result.solve_seconds,
                    attempt.get('lift', {}).get('status', '-')), flush=True)
                if result.words is not None:
                    break
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
