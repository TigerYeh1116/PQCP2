"""Complete restricted construction experiment. Never updates solution archives."""
import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from solver.automorphism_construction import (positive_automorphism_target,
    projective_order, search_automorphism_target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--d', type=int, choices=(3, 11, 43), default=43)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sc, sh = {3: (2, 1), 11: (3, 2), 43: (6, 3)}[args.d]
    started, rows = perf_counter(), []
    for v in range(1, args.d//2+1):
        if projective_order(v, args.d) != (args.d-1)//2:
            continue
        for k in range(1, args.d//2+1):
            spec = positive_automorphism_target(args.d, v, k, sc, sh)
            out = search_automorphism_target(spec)
            rows.append(dict(v=v, k=k, result=out))
        print(json.dumps(dict(v=v, parameters=len(rows),
            verified=sum(r['result']['verified'] for r in rows),
            elapsed=perf_counter()-started)), flush=True)
    report = dict(length=2*args.d, parameters=len(rows), rows=rows,
        complete=all(r['result']['complete'] or r['result']['verified'] for r in rows),
        verified_parameters=sum(r['result']['verified'] for r in rows),
        elapsed=perf_counter()-started,
        scope='Only y_i=(-1)^i*x_(v*i) and the derived unary profiles; not all PQCPs.',
        speed_claim='No finding-rate claim. Equivalent models are not independent discoveries.')
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'rows'}), flush=True)


if __name__ == '__main__':
    main()
