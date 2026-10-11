"""IDE-runnable exact arbitrary 43-bit parity-half completion of fresh centers.

Default reads ALL audited non-solution centers in the latest root L86 log.
Each of four half-rows is completely free in turn; the other three are fixed.
All necessary counts/targets and all support rotations are retained. Finite
fiber UNSAT never excludes unrelated centers or the whole L86 space.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys
from time import perf_counter

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.benchmark_l86_discovery import discovery_key,freeze_sources,verified_inventory
from experiments.repair_unrestricted_flips import audited_log_centers
from solver.checkpoint import atomic_write_json
from solver.half_row_completion import complete_parity_half_rows
from solver.search_runner import append_verified_solution_if_new
from solver.solution_equivalence import pqcp_equivalence_key

DEFAULT_TOP=0  # 0 means all logged non-solutions; positive values bound selection.


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--input',type=Path,help='JSON with seeds/words')
    group.add_argument('--log',type=Path,help='JSONL, default newest root L86 log')
    parser.add_argument('--top',type=int,default=DEFAULT_TOP)
    parser.add_argument('--free-components',type=int,nargs='+',default=(0,1,2,3))
    parser.add_argument('--certified-psd',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--trace-backend',action=argparse.BooleanOptionalAction,default=False)
    parser.add_argument('--output',type=Path,default=ROOT/'results'/(
        'L86_half_row_completion_'+datetime.now().strftime('%Y%m%d_%H%M%S')))
    args=parser.parse_args()
    if args.top<0 or len(set(args.free_components))!=len(args.free_components) or any(v not in range(4) for v in args.free_components):
        parser.error('nonnegative top and distinct components 0..3 required')
    if args.input is None and args.log is None:
        logs=tuple((ROOT/'logs').glob('torch_L86_seed*.jsonl'))
        if not logs:raise ValueError('no L86 log; supply an input JSON')
        args.log=max(logs,key=lambda p:p.stat().st_mtime)
    source=args.log if args.log is not None else args.input
    inputs=audited_log_centers(args.log,args.top if args.top else sys.maxsize) if args.log is not None else json.loads(args.input.read_text())
    args.output.mkdir(parents=True,exist_ok=False);tick=perf_counter()
    snapshot=args.output/'source_snapshot';freeze_sources(snapshot);(snapshot/'experiments').mkdir(exist_ok=True)
    for path in (ROOT/'experiments').glob('*.py'):shutil.copy2(path,snapshot/'experiments'/path.name)
    manifest={str(p.relative_to(snapshot)):hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(snapshot.rglob('*')) if p.is_file() and p.name!='source_manifest.json'}
    atomic_write_json(snapshot/'source_manifest.json',manifest)
    shutil.copy2(source,args.output/('input_log.jsonl' if args.log is not None else 'input_source.json'))
    atomic_write_json(args.output/'input_centers.json',inputs)
    raw=(ROOT/'86.txt').read_bytes() if (ROOT/'86.txt').exists() else b''
    (args.output/'initial_inventory.txt').write_bytes(raw);known=verified_inventory(raw)
    centers={};all_components=set(args.free_components)==set(range(4))
    for item in inputs['seeds']:
        words=tuple(tuple(w) for w in item['words'])
        if tuple(map(len,words))!=(86,86):raise ValueError('L86 centers required')
        # Odd row rotations and row swap permute free components. They are
        # valid for the UNION of all four fibers, but not a selected subset.
        key=pqcp_equivalence_key(*words) if all_components else words
        centers.setdefault(key,words)
    if not centers:raise ValueError('no audited non-solution centers')
    expected=len(centers)*len(args.free_components);keys=set();first_hit=None;rows=[]
    totals=dict(norm_visited=0,weight_matches=0,intersection_checks=0,completed_support_orbits=0,
                restored_phase_candidates=0,odd_profile_checks=0,norm_trace_candidates=0,
                completed_trace_support_masks=0,integer_profile_checks=0)
    print(json.dumps(dict(input=str(source),centers=len(centers),fibers=expected)),flush=True)
    for center_index,words in enumerate(centers.values()):
        for out in complete_parity_half_rows(*words,free_components=args.free_components,
                                            certified_psd=args.certified_psd,trace_backend=args.trace_backend):
            free=out['free_component']
            if not out['complete']:raise RuntimeError('half-row fiber not completely decided')
            for hit in out['witnesses']:
                key=discovery_key(*hit['words'])
                if key not in known:
                    if key not in keys and first_hit is None:first_hit=perf_counter()-tick
                    keys.add(key);append_verified_solution_if_new(86,*hit['words'],args.output)
            with (args.output/'completion_fibers.jsonl').open('a') as handle:
                handle.write(json.dumps(dict(center_index=center_index,result=out))+'\n')
            rows.append(dict(center_index=center_index,free_component=free,status=out['status'],
                             profiles_decided=out['profiles_decided'],witnesses=len(out['witnesses'])))
            for name in totals:totals[name]+=out[name]
        report=dict(complete=len(rows)==expected,expected_fibers=expected,completed_fibers=len(rows),
                    centers=len(centers),free_components=args.free_components,certified_psd=args.certified_psd,
                    trace_backend=args.trace_backend,rows=rows,**totals,
                    raw_assignments_per_fiber=1<<43,raw_assignment_count_with_overlaps=(1<<43)*len(rows),
                    independently_verified_new_classes=len(keys),first_verified_elapsed=first_hit,
                    elapsed=perf_counter()-tick,source_manifest_sha256=hashlib.sha256(
                        json.dumps(manifest,sort_keys=True).encode()).hexdigest(),
                    scope='Complete arbitrary-bit half-row fibers of reported centers. '
                          'All-four-component quotient preserves fiber UNION; partial selections use literal centers. '
                          'Counts may change. No global exclusion or discovery-speed claim.')
        atomic_write_json(args.output/'report.json',report)
        print(json.dumps({k:report[k] for k in ('complete','completed_fibers','completed_support_orbits',
                         'independently_verified_new_classes','elapsed')},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
