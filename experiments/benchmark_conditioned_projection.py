"""Paired exact conditioned-sign kernels on uniformly sampled sparse support pairs.

No content exclusions are used. Tests negative n_A=1 and positive n_A=15,
the two coarse weight-seven shapes, and preserves all partner preimages.
Results measure fixed-support completion, not new full PQCP discovery.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
from random import Random
import statistics
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from solver.sparse_supports import sparse_bracelets
from solver.structured_energy import integer_periodic_autocorrelation_profile as pacf
from solver.ternary_system_native import prepare_native_system, run_native_system


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases', type=int, default=512)
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    if min(args.cases, args.repeats) < 1:
        parser.error('positive case/repeat limits required')
    args.output.mkdir(parents=True, exist_ok=False)
    generated = sparse_bracelets(43, 7, capacity=375364)
    if not generated['complete'] or generated['total'] != 375364:
        raise RuntimeError('bracelet generation incomplete')
    index = {}
    for mask, sig in generated['records']:
        index.setdefault(sig, []).append(mask)
    sample, count, rng = [], 0, Random(86)
    for u, sig in generated['records']:
        for w in index.get(sig ^ (((1 << 21)-1)^2), ()):
            count += 1
            if len(sample) < args.cases:
                sample.append((u, w))
            else:
                chosen = rng.randrange(count)
                if chosen < args.cases:
                    sample[chosen] = (u, w)
    if count != 130142:
        raise RuntimeError('full parity preimage coverage mismatch')
    cases, full = [], (1 << 43)-1
    for eta, negative in ((-1, (1, 16)), (1, (15, 2))):
        wanted = tuple(43 if u == 0 else eta if u in (2, 41) else 0 for u in range(43))
        for u, w in sample:
            masks = (u, full ^ w) if eta == -1 else (full ^ u, w)
            cases.append(dict(masks=masks, wanted=wanted, negative=negative, eta=eta))
    # Compile and warm both before paired measurements.
    for enabled in (False, True):
        c = cases[0]
        run_native_system(prepare_native_system(c['masks'], 43, c['wanted'], c['negative']),
                          2**23, conditioned_hybrid=enabled)
    samples, first = [], {}
    for repeat in range(args.repeats):
        measurement = {}
        for enabled in ((False, True) if repeat % 2 == 0 else (True, False)):
            tick = perf_counter()
            rows = [run_native_system(prepare_native_system(c['masks'], 43, c['wanted'], c['negative']),
                                      2**23, conditioned_hybrid=enabled) for c in cases]
            elapsed = perf_counter()-tick
            for case, out in zip(cases, rows):
                if out['words'] is not None:
                    a, b = out['words']
                    if tuple(x+y for x, y in zip(pacf(a), pacf(b))) != case['wanted'] or (a.count(-1), b.count(-1)) != case['negative']:
                        raise RuntimeError('independent coarse witness audit failed')
            if repeat == 0:
                first[enabled] = rows
            elif [r['status'] for r in rows] != [r['status'] for r in first[enabled]]:
                raise RuntimeError('nonreproducible exact statuses')
            measurement['projection' if enabled else 'legacy'] = dict(seconds=elapsed,
                assignments=sum(r['assignments'] for r in rows),
                statuses=dict(Counter(r['status'] for r in rows)),
                prefix_branches=sum(r.get('prefix_branches', 0) for r in rows),
                projected_candidates=sum(r.get('projected_candidates', 0) for r in rows))
        if [r['status'] for r in first[False]] != [r['status'] for r in first[True]]:
            raise RuntimeError('legacy/projected exact status disagreement')
        samples.append(measurement)
        print(json.dumps(dict(repeat=repeat+1, measurement=measurement)), flush=True)
    old, new = (statistics.median(s[name]['seconds'] for s in samples) for name in ('legacy', 'projection'))
    summary = dict(cases=len(cases), repeats=args.repeats, matched_pair_population=count,
                   old_seconds=old, projection_seconds=new, seconds_reduction_percent=100*(1-new/old),
                   exact_statuses_equal=True, inference_scope='fixed-support completion only')
    (args.output/'report.json').write_text(json.dumps(dict(summary=summary, samples=samples,
        references=cases, legacy=first[False], projection=first[True]), indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
