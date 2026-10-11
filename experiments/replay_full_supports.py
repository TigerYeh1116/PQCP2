"""Resume untested joint supports using exact two-problem binary completion."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from solver.checkpoint import atomic_write_json
from solver.pqcp_support_solver import complete_pqcp_supports
from solver.search_runner import append_verified_solution_if_new
from solver.target_profiles import TargetContentProfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    saved = json.loads(args.candidate.read_text())
    reference = saved['reference']
    target = TargetContentProfile(**reference['target'])
    tried = {tuple(a['masks']) for a in saved['result']['attempts']}
    remaining = [tuple(m) for m in saved['result']['generation']['masks'] if tuple(m) not in tried]
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for i, masks in enumerate(remaining):
        out = complete_pqcp_supports(*reference['words'], target, masks)
        row = dict(masks=masks, result=out)
        if out['verified']:
            row['new_isolated_solution'] = append_verified_solution_if_new(target.L, *out['words'], args.output)
        rows.append(row)
        if (i+1) % 64 == 0 or out['verified'] or i+1 == len(remaining):
            summary = dict(tested=len(rows), remaining=len(remaining)-len(rows),
                statuses=dict(Counter(r['result']['status'] for r in rows)),
                verified=sum(r['result']['verified'] for r in rows),
                seconds=sum(r['result']['elapsed'] for r in rows))
            atomic_write_json(args.output/'report.json', dict(summary=summary, rows=rows))
            print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
