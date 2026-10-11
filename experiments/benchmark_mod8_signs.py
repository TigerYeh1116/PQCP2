"""Paired exact support replay: raw, mod-8 quadratic, constant-weight kernels.

Includes native affine-plan construction. This measures fixed-support
completion, not time to a new full length-86 PQCP.
"""
import argparse
import json
from pathlib import Path
import statistics
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from solver.compressed_repair import _validated_reference
from solver.ternary_support_search import _audit_sign_words, complete_support_signs
from solver.target_profiles import TargetContentProfile
from solver.ternary_system_native import prepare_native_system, run_native_system


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=256)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--production', action='store_true',
                        help='compare the existing completion API before/after mod-8 filtering')
    parser.add_argument('--hybrid-tails', type=int, nargs='+',
                        help='compare exact quotient projection with these tail sizes, against Gray mod-8')
    parser.add_argument('--projection-baseline', action='store_true',
                        help='with --production, compare previous Gray mod-8 against quotient projection')
    args = parser.parse_args()
    if min(args.limit, args.repeats) < 1:
        parser.error('positive limits required')
    if args.hybrid_tails and (args.production or any(not 1 <= t <= 8 for t in args.hybrid_tails)):
        parser.error('hybrid tails in 1..8, without --production, required')
    if args.projection_baseline and not args.production:
        parser.error('--projection-baseline requires --production')
    refs = []
    for row in json.loads(args.input.read_text())['rows']:
        ref = row['reference']
        target = TargetContentProfile(**ref['target'])
        a, b, d, wanted = _validated_reference(*ref['words'], target)
        for attempt in row['norm']['attempts']:
            refs.append(dict(masks=tuple(attempt['masks']), d=d, wanted=wanted,
                             negative=(a.count(-1), b.count(-1)), a=a, b=b, target=target))
    refs = refs[:args.limit]
    if not refs:
        parser.error('no exact support cases')
    args.output.mkdir(parents=True, exist_ok=False)
    modes = ('legacy', 'hybrid') if args.production else ('raw', 'quadratic', 'conditioned', 'hybrid')
    if args.hybrid_tails:
        modes = ('quadratic',)+tuple('tail{}'.format(t) for t in dict.fromkeys(args.hybrid_tails))

    def solve(ref, mode):
        if args.production:
            return complete_support_signs(ref['a'], ref['b'], ref['target'], ref['masks'],
                conditioning=True, quadratic=None if mode == 'hybrid' or args.projection_baseline else False,
                hybrid_tail=0 if mode == 'legacy' else None)
        plan = prepare_native_system(ref['masks'], ref['d'], ref['wanted'], ref['negative'],
                                     conditioning=mode in ('conditioned', 'hybrid'))
        return run_native_system(plan, 2**23, quadratic=mode in ('quadratic', 'hybrid') or mode.startswith('tail'),
                                 hybrid_tail=int(mode[4:]) if mode.startswith('tail') else 0)

    for mode in modes:
        solve(refs[0], mode)
    samples, first = [], {}
    for repeat in range(args.repeats):
        order = list(modes[repeat % len(modes):]+modes[:repeat % len(modes)])
        if repeat % 2:
            order.reverse()
        sample = {}
        for mode in order:
            tick = perf_counter()
            rows = [solve(r, mode) for r in refs]
            elapsed = perf_counter()-tick
            for ref, out in zip(refs, rows):
                if out['words'] is not None:
                    _audit_sign_words(out['words'], ref['a'], ref['b'], ref['target'], ref['wanted'], ref['masks'])
            if repeat == 0:
                first[mode] = rows
            elif [r['status'] for r in rows] != [r['status'] for r in first[mode]]:
                raise RuntimeError('replay outcomes changed')
            sample[mode] = dict(seconds=elapsed, assignments=sum(r['assignments'] for r in rows),
                quadratic_matches=sum(r.get('quadratic_matches', 0) for r in rows),
                count_matches=sum(r.get('count_matches', 0) for r in rows),
                correlation_checks=sum(r.get('correlation_checks', 0) for r in rows))
        expected = [r['status'] for r in first[modes[0]]]
        if any([r['status'] for r in first[m]] != expected for m in modes):
            raise RuntimeError('exact modes disagree')
        samples.append(sample)
        print(json.dumps(dict(repeat=repeat+1, sample=sample)), flush=True)
    medians = {m: statistics.median(s[m]['seconds'] for s in samples) for m in modes}
    summary = dict(cases=len(refs), repeats=args.repeats, medians=medians,
                   outcomes_equal=True,
                   production_api=args.production,
                   projection_baseline=args.projection_baseline,
                   caveat='Fixed support completion only; not time to a new full PQCP.')
    if args.hybrid_tails:
        best = min(modes[1:], key=lambda m: medians[m])
        summary.update(best_projection_mode=best,
                       best_projection_seconds_reduction_percent=100*(1-medians[best]/medians['quadratic']))
    else:
        summary['hybrid_seconds_reduction_percent'] = 100*(1-medians['hybrid']/medians[modes[0]])
    if not args.production and not args.hybrid_tails:
        summary['quadratic_seconds_reduction_percent'] = 100*(1-medians['quadratic']/medians['raw'])
    (args.output/'report.json').write_text(json.dumps(dict(summary=summary, samples=samples, outcomes=first), indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
