"""Compare local versus cyclic-norm support repair on saved candidates.

No formal sequence archive is written. Counts measure compressed search,
not full PQCP discovery speed. Use isolated output directories.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from solver.cyclic_support_search import repair_cyclic_supports
from solver.target_profiles import TargetContentProfile
from solver.ternary_support_search import repair_support_signs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=16)
    parser.add_argument('--max-supports', type=int, default=8)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error('positive candidate limit required')
    archive = json.loads(args.input.read_text())
    if isinstance(archive, dict):
        refs = [row['reference'] for row in archive['rows']]
    else:
        refs = [dict(words=row['result']['words'], target=row['reference']['target'],
                     energy=row['result']['energy']) for row in archive]
    refs = sorted(refs, key=lambda r: r['energy'])[:args.limit]
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    started = perf_counter()
    for i, ref in enumerate(refs):
        target = TargetContentProfile(**ref['target'])
        local = repair_support_signs(*ref['words'], target, max_supports=args.max_supports)
        norm = repair_cyclic_supports(*ref['words'], target, max_supports=args.max_supports, full_completion=False)
        rows.append(dict(reference=ref, local=local, norm=norm))
        if len(refs) <= 16 or (i+1) % 16 == 0 or i+1 == len(refs):
            print('completed', i+1, 'energy', ref['energy'], 'norm', norm['status'],
                  'distances', [g['distance'] for g in norm['generation']], flush=True)
    summary = dict(tested=len(rows), seconds=perf_counter()-started)
    for method in ('local', 'norm'):
        cases = [row[method] for row in rows]
        summary[method] = dict(statuses=dict(Counter(c['status'] for c in cases)),
            seconds=sum(c['elapsed'] for c in cases),
            supports_tested=sum(len(c['attempts']) for c in cases),
            sign_statuses=dict(Counter(a['result']['status'] for c in cases for a in c['attempts'])),
            assignments=sum(a['result'].get('assignments', 0) for c in cases for a in c['attempts']))
    summary['norm_weight_matches'] = sum(g['weight_matches'] for r in rows for g in r['norm']['generation'])
    summary['norm_minimum_distances'] = dict(Counter(g['distance'] for r in rows for g in r['norm']['generation']))
    (args.output/'report.json').write_text(json.dumps(dict(summary=summary, rows=rows), indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
