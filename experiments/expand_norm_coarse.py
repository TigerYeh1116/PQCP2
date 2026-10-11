"""Exact two-swap trades around independently found C witnesses, then G lift.

Per-row signed counts stay fixed; every exported trade has exact folded C
PACF. Full completion, not C matching or a truncated neighborhood, decides
whether a new PQCP exists. Proven dihedral UNSAT orbits are optional rollback.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.benchmark_l86_discovery import discovery_key
from solver.checkpoint import atomic_write_json
from solver.pqcp_support_solver import complete_pqcp_supports
from solver.search_runner import append_verified_solution_if_new
from solver.support_orbit_cache import FULL_SUPPORT_ORBITS
from solver.target_profiles import TargetContentProfile
from solver.ternary_mitm import repair_mitm
from solver.ternary_support_search import support_mask


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=ROOT/'results/L86_independent_norm_long_20261007/coarse_witnesses.json')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--max-results',type=int,default=4096)
    parser.add_argument('--orbit-pruning',action=argparse.BooleanOptionalAction,default=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    saved=json.loads(args.input.read_text())
    tick,rows,keys,first_hit=perf_counter(),[],set(),None
    FULL_SUPPORT_ORBITS.clear()
    for source in saved['rows']:
        target=TargetContentProfile(**source['target'])
        coarse=source['result']['attempts']['coarse']['words']
        matched=repair_mitm(*coarse,target,max_results=args.max_results)
        completions=[]
        for words in matched['words']:
            masks=tuple(support_mask(w) for w in words)
            out=complete_pqcp_supports(*words,target,masks,orbit_pruning=args.orbit_pruning)
            if out['verified']:
                key=discovery_key(*out['words'])
                if key not in keys and first_hit is None:first_hit=perf_counter()-tick
                keys.add(key);append_verified_solution_if_new(86,*out['words'],args.output)
            completions.append(dict(masks=masks,result=out))
        row=dict(source_trial=source['trial'],target=source['target'],coarse_neighborhood=matched,completions=completions)
        rows.append(row)
        report=dict(rows=rows,complete=len(rows)==len(saved['rows']),
            all_neighborhoods_complete=all(r['coarse_neighborhood']['complete'] for r in rows),
            orbit_pruning=args.orbit_pruning,independently_verified_new_classes=len(keys),
            first_verified_elapsed=first_hit,elapsed=perf_counter()-tick,
            scope='Bounded N2(C_A) x N2(C_B) trades from selected independent C witnesses; not global L86 exclusion.')
        atomic_write_json(args.output/'report.json',report)
        print(json.dumps(dict(trial=source['trial'],coarse_models=len(matched['words']),
            neighborhood_complete=matched['complete'],statuses=dict(Counter(c['result']['status'] for c in completions)),
            orbit_hits=sum(c['result']['excluded_by']=='dihedral_orbit' for c in completions),
            new_classes=len(keys),elapsed=perf_counter()-tick)),flush=True)


if __name__=='__main__':main()
