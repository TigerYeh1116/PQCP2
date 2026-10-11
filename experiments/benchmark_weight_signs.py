"""Paired replay of identical fixed-support problems, legacy vs conditioned.

This verifies exhaustive fixed-support outcomes and measures sign-completion
time. It does not measure discovery time of new full PQCPs.
"""
import argparse
import json
from pathlib import Path
import statistics
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from solver.target_profiles import TargetContentProfile
from solver.ternary_support_search import complete_support_signs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=256)
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    if min(args.limit, args.repeats) < 1:
        parser.error('positive limits required')
    refs = []
    for row in json.loads(args.input.read_text())['rows']:
        for attempt in row['norm']['attempts']:
            refs.append(dict(reference=row['reference'], masks=attempt['masks']))
    refs = refs[:args.limit]
    if not refs:
        parser.error('no fixed-support records')
    args.output.mkdir(parents=True, exist_ok=False)
    # Warm both source-hash libraries and target tables outside timed samples.
    for enabled in (False, True):
        ref = refs[0]
        complete_support_signs(*ref['reference']['words'], TargetContentProfile(**ref['reference']['target']),
                               ref['masks'], conditioning=enabled, content_pruning=enabled,
                               quadratic=None if enabled else False)
    samples = []
    outcomes = []
    for repeat in range(args.repeats):
        sample = {}
        for enabled in ((False, True) if repeat % 2 == 0 else (True, False)):
            started = perf_counter()
            rows = [complete_support_signs(*r['reference']['words'], TargetContentProfile(**r['reference']['target']),
                                          r['masks'], conditioning=enabled, content_pruning=enabled,
                                          quadratic=None if enabled else False) for r in refs]
            sample['conditioned' if enabled else 'legacy'] = dict(
                seconds=perf_counter()-started, assignments=sum(r.get('assignments', 0) for r in rows),
                conditioned_cases=sum('conditioned_side' in r for r in rows))
            if repeat == 0:
                outcomes.append(rows)
            else:
                baseline = outcomes[1 if enabled else 0]
                if [r['status'] for r in rows] != [r['status'] for r in baseline]:
                    raise RuntimeError('nonreproducible fixed-support outcomes')
        if repeat == 0 and [r['status'] for r in outcomes[0]] != [r['status'] for r in outcomes[1]]:
            raise RuntimeError('conditioned and legacy support outcomes disagree')
        samples.append(sample)
        print('repeat', repeat+1, json.dumps(sample), flush=True)
    legacy = statistics.median(s['legacy']['seconds'] for s in samples)
    conditioned = statistics.median(s['conditioned']['seconds'] for s in samples)
    summary = dict(cases=len(refs), repeats=args.repeats, legacy_seconds=legacy,
        conditioned_seconds=conditioned, seconds_reduction_percent=100*(1-conditioned/legacy),
        throughput_increase_percent=100*(legacy/conditioned-1), all_outcomes_equal=True)
    (args.output/'report.json').write_text(json.dumps(dict(summary=summary, samples=samples,
        references=refs, legacy=outcomes[0], conditioned=outcomes[1]), indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
