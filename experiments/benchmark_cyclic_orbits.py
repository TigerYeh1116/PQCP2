"""Global partner-orbit replay, both exact sign problems, isolated full writes."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from solver.checkpoint import atomic_write_json
from solver.cyclic_orbit_search import search_cyclic_orbits
from solver.search_runner import append_verified_solution_if_new
from solver.target_profiles import TargetContentProfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=16)
    parser.add_argument('--max-orbits', type=int, default=4096)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error('positive limit required')
    archive = json.loads(args.input.read_text())
    refs = ([r['reference'] for r in archive['rows']] if isinstance(archive, dict) else
            [dict(words=r['result']['words'], target=r['reference']['target'], energy=r['result']['energy'])
             for r in archive])
    refs = sorted(refs, key=lambda r: r['energy'])[:args.limit]
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for i, ref in enumerate(refs):
        target = TargetContentProfile(**ref['target'])
        out = search_cyclic_orbits(*ref['words'], target, max_orbits=args.max_orbits)
        row = dict(reference=ref, result=out)
        if out['verified']:
            row['new_isolated_solution'] = append_verified_solution_if_new(target.L, *out['words'], args.output)
        rows.append(row)
        atomic_write_json(args.output/'candidates'/('{:04d}.json'.format(i)), row)
        print('completed', i+1, 'E', ref['energy'], 'orbits', len(out['attempts']),
              'status', out['status'], 'seconds', round(out['elapsed'], 3), flush=True)
    summary = dict(tested=len(rows), statuses=dict(Counter(r['result']['status'] for r in rows)),
        orbit_cases=sum(len(r['result']['attempts']) for r in rows),
        full_solutions=sum(r['result']['verified'] for r in rows),
        seconds=sum(r['result']['elapsed'] for r in rows))
    atomic_write_json(args.output/'report.json', dict(summary=summary, rows=rows))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
