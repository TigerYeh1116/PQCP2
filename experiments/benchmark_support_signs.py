"""Replay recorded ternary candidates; never write a formal PQCP archive."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from solver.compressed_repair import _validated_reference
from solver.target_profiles import TargetContentProfile
from solver.ternary_support_search import repair_support_signs, support_mask, support_signature


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=256)
    parser.add_argument('--max-supports', type=int, default=8)
    parser.add_argument('--max-assignments', type=int, default=2**23)
    args = parser.parse_args()
    if min(args.limit, args.max_supports, args.max_assignments) < 1: parser.error('positive limits required')
    references = []
    if args.input.is_dir():
        for path in sorted(args.input.glob('L*_repair*.json')):
            for row in json.loads(path.read_text()).get('attempts', []):
                references.append(dict(words=row['reference'], target=row['target'],
                    energy=row['original_energy'], path=str(path), lane=row['lane']))
    else:
        for i, row in enumerate(json.loads(args.input.read_text())):
            references.append(dict(words=row['result']['words'], target=row['reference']['target'],
                energy=row['result']['energy'], path=str(args.input), lane=i))
    if not references: parser.error('no supported candidate records')
    args.output.mkdir(parents=True, exist_ok=False)
    diagnostic = Counter()
    for ref in references:
        a, b, d, wanted = _validated_reference(*ref['words'], TargetContentProfile(**ref['target']))
        syndrome = support_signature(support_mask(a), d) ^ support_signature(support_mask(b), d)
        syndrome ^= sum((wanted[u] & 1) << (u-1) for u in range(1, d//2+1))
        ref['odd_residual_shifts'] = bin(syndrome).count('1')
        diagnostic[ref['odd_residual_shifts']] += 1
    selected = sorted(references, key=lambda r: (r['energy'], r['path'], r['lane']))[:args.limit]
    rows = []; started = perf_counter()
    for i, ref in enumerate(selected):
        result = repair_support_signs(*ref['words'], TargetContentProfile(**ref['target']),
            max_supports=args.max_supports, max_assignments=args.max_assignments)
        rows.append(dict(reference=ref, result=result))
        if (i+1) % 32 == 0: print('completed', i+1, 'seconds', round(perf_counter()-started, 3), flush=True)
    summary = dict(archive_references=len(references), tested=len(rows),
        original_odd_shift_histogram=dict(sorted(diagnostic.items())),
        statuses=dict(Counter(r['result']['status'] for r in rows)),
        matched_supports=sum(len(r['result']['attempts']) for r in rows),
        sign_statuses=dict(Counter(a['result']['status'] for r in rows for a in r['result']['attempts'])),
        ranks=dict(Counter(a['result']['rank'] for r in rows for a in r['result']['attempts'])),
        assignments=sum(a['result'].get('assignments', 0) for r in rows for a in r['result']['attempts']),
        seconds=perf_counter()-started)
    (args.output/'report.json').write_text(json.dumps(dict(input=str(args.input),
        max_supports=args.max_supports, max_assignments=args.max_assignments, summary=summary, rows=rows), indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__': main()
