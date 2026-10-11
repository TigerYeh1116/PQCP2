"""Continue unclosed centers in independent CPU processes, preserving a closed prefix.

The original partially complete report supplies ONLY its fully closed center
prefix. New children solve all remaining entire centers from one frozen
runtime copy, with exact Gram/finite-field necessary conditions. No overlap
in credited center indices, no sampled cutoff, and no restart of closed work.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime
import hashlib
import json
from math import comb
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from time import perf_counter

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.benchmark_l86_discovery import freeze_sources,verified_inventory,discovery_key
from solver.checkpoint import atomic_write_json
from solver.search_runner import append_verified_solution_if_new
from solver.verifier import verify_pqcp


def closed_parent_prefix(parent,inputs):
    count=parent['completed_centers'];rows=parent['rows'];expected=12*sum(comb(43,r) for r in range(parent['radius']+1))
    if (parent['radius']!=4 or count!=len(rows) or not 0<count<len(inputs['seeds'])
            or parent['centers']!=len(inputs['seeds'])):raise ValueError('partially complete radius-four parent with a closed nonempty prefix required')
    for index,row in enumerate(rows):
        if (row['center_index']!=index or row['generated_triples']!=expected
                or sum(row[k] for k in ('content_excluded','identical_triples','eligible_jobs'))!=expected):
            raise RuntimeError('parent center prefix was not completely decided')
    if sum(row['eligible_jobs'] for row in rows)!=parent['eligible_jobs']:raise RuntimeError('parent report mixes closed and unfinished center counts')
    if any(row['witnesses'] for row in rows):
        raise ValueError('positive parent-prefix solutions require separate inventory reconciliation before rescheduling')
    return count


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent',type=Path,default=ROOT/'results/L86_latest_two_halves_gram_radius4_top8_20261008')
    parser.add_argument('--fresh-input',type=Path,help='Start a new complete radius-four region from audited JSON centers, without crediting any old prefix.')
    parser.add_argument('--varied-halves',type=int,choices=(1,2,3),default=1)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--output',type=Path,default=ROOT/'results'/('L86_two_half_parallel_remaining_'+datetime.now().strftime('%Y%m%d_%H%M%S')))
    args=parser.parse_args()
    args.parent=args.parent.resolve();args.output=args.output.resolve()
    if not 1<=args.workers<=min(8,os.cpu_count() or 1):parser.error('workers outside the bounded available CPU count')
    if args.fresh_input is None:
        if args.varied_halves!=1:parser.error('closed-parent rescheduling applies only to the original one-varied-half region')
        parent=json.loads((args.parent/'report.json').read_text());inputs=json.loads((args.parent/'input_centers.json').read_text())
        prefix=closed_parent_prefix(parent,inputs)
        manifest=json.loads((args.parent/'source_snapshot/source_manifest.json').read_text())
        if hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()!=parent['source_manifest_sha256']:
            raise RuntimeError('parent report source manifest mismatch')
    else:
        from experiments.complete_two_l86_halves import select_json_centers
        inputs=select_json_centers(json.loads(args.fresh_input.read_text()),0);prefix=0;parent=None
        if not inputs['seeds']:raise ValueError('fresh input must contain non-solution centers')
    # Preserve the exact report at launch even if the original experiment
    # continues independently. Only this pinned prefix is ever credited.
    args.output.mkdir(parents=True,exist_ok=False);tick=perf_counter()
    atomic_write_json(args.output/'input_centers.json',inputs)
    if parent is not None:
        atomic_write_json(args.output/'parent_closed_prefix_report.json',parent)
        shutil.copy2(args.parent/'source_snapshot/source_manifest.json',args.output/'parent_source_manifest.json')
        shutil.copy2(args.parent/'initial_inventory.txt',args.output/'initial_inventory.txt')
    else:
        shutil.copy2(args.fresh_input,args.output/'input_source.json')
        (args.output/'initial_inventory.txt').write_bytes((ROOT/'86.txt').read_bytes() if (ROOT/'86.txt').exists() else b'')
    from solver.three_half_row_completion import three_half_triple_count,four_half_triple_count
    expected=(12*sum(comb(43,r) for r in range(5)) if args.varied_halves==1 else
              three_half_triple_count(43,4) if args.varied_halves==2 else four_half_triple_count(43,4))
    known=verified_inventory((args.output/'initial_inventory.txt').read_bytes())
    snapshot=args.output/'source_snapshot';freeze_sources(snapshot);(snapshot/'experiments').mkdir(exist_ok=True)
    for p in (ROOT/'experiments').glob('*.py'):shutil.copy2(p,snapshot/'experiments'/p.name)
    (snapshot/'86.txt').write_bytes((args.output/'initial_inventory.txt').read_bytes())
    sources={str(p.relative_to(snapshot)):hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(snapshot.rglob('*')) if p.is_file() and p.name!='source_manifest.json'}
    atomic_write_json(snapshot/'source_manifest.json',sources)
    # Compilation and pycache writes belong to a separate runtime checkout;
    # the authoritative source snapshot remains unchanged.
    runtime=args.output/'runtime_source';shutil.copytree(snapshot,runtime)
    job_root=args.output/'jobs';job_root.mkdir();completed={};new_keys=set();full_models=[]
    plans=[]
    for index in range(prefix,len(inputs['seeds'])):
        folder=job_root/('center_'+str(index));folder.mkdir();input_path=folder/'input.json'
        atomic_write_json(input_path,dict(seeds=[inputs['seeds'][index]],parent_center_index=index))
        plans.append((index,folder,input_path))
    def run(plan):
        index,folder,input_path=plan;result=folder/'result';env=dict(os.environ)
        env['PYTHONPYCACHEPREFIX']='/private/tmp/pqcp2_parallel_radius4_pycache'
        command=[str(ROOT/'.venv/bin/python'),str(runtime/'experiments/complete_two_l86_halves.py'),
                 '--input',str(input_path.resolve()),'--top','0','--radius','4','--joint-gram','--finite-trace',
                 '--varied-halves',str(args.varied_halves),
                 '--output',str(result.resolve())]
        with (folder/'console.log').open('w') as log:
            process=subprocess.Popen(command,cwd=str(runtime),env=env,stdout=log,stderr=subprocess.STDOUT)
            atomic_write_json(folder/'process.json',dict(pid=process.pid,parent_center_index=index,command=command))
            code=process.wait()
        if code:raise RuntimeError(f'center {index} process failed with status {code}; inspect {folder}/console.log')
        report=json.loads((result/'report.json').read_text())
        if (not report['complete'] or report['centers']!=1 or report['radius']!=4
                or report.get('varied_halves',1)!=args.varied_halves or report['generated_triples']!=expected):
            raise RuntimeError('parallel child did not close its whole declared center')
        child_input=json.loads((result/'input_centers.json').read_text())
        if tuple(map(tuple,child_input['seeds'][0]['words']))!=tuple(map(tuple,inputs['seeds'][index]['words'])):
            raise RuntimeError('parallel child changed its literal input center')
        return index,folder,report
    def save():
        credited=set(range(prefix))|set(completed)
        report=dict(complete=len(credited)==len(inputs['seeds']),radius=4,workers=args.workers,
                    varied_halves=args.varied_halves,expected_triples_per_center=expected,
                    expected_centers=len(inputs['seeds']),credited_center_indices=sorted(credited),closed_parent_prefix=prefix,
                    completed_parallel_centers=len(completed),children={str(k):v for k,v in completed.items()},
                    independently_verified_new_classes=len(new_keys),new_full_models=full_models,elapsed=perf_counter()-tick,
                    source_manifest_sha256=hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest(),
                    parent_source_manifest_sha256=parent['source_manifest_sha256'] if parent is not None else None,
                    scope=('Pinned fully closed parent center prefix plus independently complete remaining entire centers. '
                           if parent is not None else 'Fresh complete radius-four regions; no old centers credited. ')+
                          f'Each center has {args.varied_halves} nontrivially varied known halves (one-varied includes zero mask), '
                          'all free roles and the declared total radius four; the free half has all 43 bits arbitrary. '
                          'New child necessary field/Gram conditions preserve all full-word models. Original partial parent is not labeled complete.',
                    claim='No >=20% new-class-discovery claim from parallel throughput or zero discoveries.')
        atomic_write_json(args.output/'report.json',report)
        print(json.dumps({k:report[k] for k in ('complete','credited_center_indices','independently_verified_new_classes','elapsed')}),flush=True)
    save()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures=[executor.submit(run,p) for p in plans]
        for future in as_completed(futures):
            index,folder,report=future.result();completed[index]=report
            # Outputs are always independently reverified before a global
            # new-class count; this remains separate from kernel metrics.
            solution=folder/'result/86.txt'
            if solution.exists():
                raw=solution.read_bytes();verified_inventory(raw)
                for a,b in re.findall(r'^a=([01]+)\r?\nb=([01]+)\r?$',raw.decode('utf-8'),re.MULTILINE):
                    words=tuple(map(int,a)),tuple(map(int,b));check=verify_pqcp(*words)
                    if not check.is_valid:raise RuntimeError('parallel saved full model failed original verifier')
                    key=discovery_key(*words)
                    if key not in known and key not in new_keys:
                        new_keys.add(key);full_models.append(dict(center_index=index,words=words))
                        append_verified_solution_if_new(86,*words,args.output)
            save()


if __name__=='__main__':main()
