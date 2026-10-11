"""Replay saved low-energy candidates on the expanded joint norm slices.

All exact coarse witnesses are independently lifted in an isolated root.
Per-candidate reports survive interruptions; no formal project archive changes.
"""
import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from solver.checkpoint import atomic_write_json
from solver.compressed_completion import CompressedLiftJob, run_compressed_lift_job
from solver.cyclic_joint_search import repair_joint_cyclic_supports
from solver.target_profiles import TargetContentProfile
from solver.torch_compressed_search import representative_lift


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=16)
    parser.add_argument('--max-supports', type=int, default=256)
    parser.add_argument('--lift-timeout-ms', type=int, default=1000)
    parser.add_argument('--minimum-distance', type=int, default=0)
    parser.add_argument('--content-changes', action='store_true')
    args = parser.parse_args()
    if min(args.limit, args.max_supports, args.lift_timeout_ms) < 1:
        parser.error('positive limits required')
    archive = json.loads(args.input.read_text())
    refs = ([row['reference'] for row in archive['rows']] if isinstance(archive, dict) else
            [dict(words=r['result']['words'], target=r['reference']['target'], energy=r['result']['energy'])
             for r in archive])
    refs = sorted(refs, key=lambda r: r['energy'])[:args.limit]
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for i, ref in enumerate(refs):
        target = TargetContentProfile(**ref['target'])
        out = repair_joint_cyclic_supports(*ref['words'], target, max_supports=args.max_supports,
            minimum_total_distance=args.minimum_distance, content_changes=args.content_changes)
        row = dict(reference=ref, result=out)
        if out['words'] is not None:
            a, b = (tuple(2*v for v in w) for w in out['words'])
            job = CompressedLiftJob(a, b, target,
                representative_lift(a, target.a_even_ones, target.a_odd_ones),
                representative_lift(b, target.b_even_ones, target.b_odd_ones))
            atomic_write_json(args.output/'coarse_jobs'/('{}.json'.format(job.identifier)), asdict(job))
            row['lift'] = run_compressed_lift_job(job, args.lift_timeout_ms, args.output/'lift_validation', 123)
        rows.append(row)
        atomic_write_json(args.output/'candidates'/('{:04d}.json'.format(i)), row)
        print('completed', i+1, 'E', ref['energy'], 'minimum', out['generation']['minimum_distance'],
              'supports', len(out['attempts']), 'status', out['status'], 'seconds', round(out['elapsed'], 3), flush=True)
    summary = dict(tested=len(rows), statuses=dict(Counter(r['result']['status'] for r in rows)),
        minimum_distances=dict(Counter(r['result']['generation']['minimum_distance'] for r in rows)),
        supports=sum(len(r['result']['attempts']) for r in rows),
        assignments=sum(a['result'].get('assignments', 0) for r in rows for a in r['result']['attempts']),
        seconds=sum(r['result']['elapsed'] for r in rows),
        verified_full_solutions=sum(bool(r.get('lift', {}).get('verified')) for r in rows),
        exhausted_layers=sum(r['result']['layer_exhausted'] for r in rows),
        minimum_distance_floor=args.minimum_distance)
    summary['content_changes'] = args.content_changes
    atomic_write_json(args.output/'report.json', dict(summary=summary, rows=rows))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
