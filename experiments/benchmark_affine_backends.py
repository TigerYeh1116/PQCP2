"""Paired exact lift throughput, NOT a benchmark of new-PQCP discovery.

Reads saved mathematical factor-2 problems only. Does not read known binary
solutions, modify production checkpoints, or append to any L.txt.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import statistics
import sys
from time import perf_counter

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from solver.affine_lift import solve_affine_lift
from solver.affine_native import _library
from solver.checkpoint import atomic_write_json
from solver.target_profiles import TargetContentProfile


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--limit',type=int,default=4)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--timeout-ms',type=int,default=10000)
    args=parser.parse_args()
    if min(args.limit,args.repeats,args.timeout_ms)<1 or args.output.exists():
        parser.error('positive controls and a new output file required')
    requests=json.loads(args.manifest.read_text())['requests'][:args.limit]
    if not requests:parser.error('empty request manifest')
    started=perf_counter();_library();startup=perf_counter()-started
    rows=[];pairs=[]
    for repeat in range(args.repeats):
        for case,request in enumerate(requests):
            p=TargetContentProfile(**request['target'])
            results={};times={}
            order=('python','native') if (repeat+case)%2==0 else ('native','python')
            for backend in order:
                tick=perf_counter()
                r=solve_affine_lift(request['middle_a'],request['middle_b'],p.k,p.eta,
                                     args.timeout_ms,p,backend=backend)
                times[backend]=perf_counter()-tick;results[backend]=r
                rows.append(dict(case=case,repeat=repeat,backend=backend,
                                 elapsed=times[backend],result=asdict(r)))
            fields=('status','a','b','profile','verified','rank','assignments','examined','next_index')
            complete=all(r.status!='UNKNOWN' for r in results.values())
            equal=all(getattr(results['python'],f)==getattr(results['native'],f) for f in fields)
            if complete and not equal:raise RuntimeError('exact backend disagreement')
            pairs.append(dict(case=case,repeat=repeat,complete=complete,equal=equal,
                              speedup=times['python']/times['native'] if complete and equal else None))
            print('case={} repeat={} Python={:.6f}s C={:.6f}s complete={} equal={}'.format(
                case,repeat,times['python'],times['native'],complete,equal),flush=True)
    valid=[p['speedup'] for p in pairs if p['speedup'] is not None]
    payload=dict(requests=requests,parameters=vars(args)|{'manifest':str(args.manifest),'output':str(args.output)},
                 native_startup_seconds=startup,rows=rows,pairs=pairs,
                 median_completed_speedup=statistics.median(valid) if valid else None,
                 caveat='Fixed-subproblem scan speed only; no claim about new PQCP time-to-solution.')
    atomic_write_json(args.output,payload)


if __name__=='__main__':main()
