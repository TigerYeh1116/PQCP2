"""Paired exact half-row completion replay; PSD certificate switch only.

Input includes fresh long-log non-solutions AND GS non-solutions with nonempty
remaining-row completion fibers. Warm compilation and alternating order;
all literal witnesses, support preimages and restored phases must agree.
No full-discovery-speed claim from zero/zero discoveries or local wall times.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
from statistics import median
import sys
from time import perf_counter

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.benchmark_l86_discovery import discovery_key,freeze_sources,verified_inventory
from solver.checkpoint import atomic_write_json
from solver.half_row_completion import complete_parity_half_rows
from solver.search_runner import append_verified_solution_if_new
from solver.solution_equivalence import pqcp_equivalence_key


def signature(out,comparison):
    if comparison=='trace':
        return (tuple(sorted(hit['words'] for hit in out['witnesses'])),out['complete'],out['raw_half_assignments'],
                tuple((r['wanted'],r['targets']) for r in out['profile_records']))
    return (tuple(sorted(hit['words'] for hit in out['witnesses'])),
            out['profiles_decided'],out['completed_support_orbits'],out['restored_phase_candidates'],out['odd_profile_checks'],
            tuple((r['wanted'],r['targets'],r['result']['masks']) for r in out['profile_records']))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=ROOT/'results/L86_latest_log_all_half_rows_20261007/input_centers.json')
    parser.add_argument('--gs',type=Path,default=ROOT/'results/L86_GS_interleave_20261007/report.json')
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--comparison',choices=('psd','trace'),default='psd')
    parser.add_argument('--output',type=Path,default=ROOT/'results'/(
        'L86_half_row_psd_benchmark_'+datetime.now().strftime('%Y%m%d_%H%M%S')))
    args=parser.parse_args()
    if args.repeats<1:parser.error('positive repeats required')
    args.output.mkdir(parents=True,exist_ok=False);snapshot=args.output/'source_snapshot';freeze_sources(snapshot)
    (snapshot/'experiments').mkdir(exist_ok=True)
    for p in (ROOT/'experiments').glob('*.py'):shutil.copy2(p,snapshot/'experiments'/p.name)
    manifest={str(p.relative_to(snapshot)):hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(snapshot.rglob('*')) if p.is_file() and p.name!='source_manifest.json'}
    atomic_write_json(snapshot/'source_manifest.json',manifest)
    inputs=[json.loads(p.read_text()) for p in (args.input,args.gs)]
    shutil.copy2(args.input,args.output/'input_log_centers.json');shutil.copy2(args.gs,args.output/'input_gs_centers.json')
    centers={}
    for data in inputs:
        for item in data['seeds']:
            words=tuple(tuple(w) for w in item['words']);centers.setdefault(pqcp_equivalence_key(*words),words)
    raw=(ROOT/'86.txt').read_bytes() if (ROOT/'86.txt').exists() else b''
    (args.output/'initial_inventory.txt').write_bytes(raw);known=verified_inventory(raw)
    # Warm ALL input fibers once, including exact norm options and nonempty GS
    # support preimages. Neither timed variant pays initial factor/compile cost.
    for words in centers.values():
        complete_parity_half_rows(*words,certified_psd=args.comparison=='trace')
        if args.comparison=='trace':complete_parity_half_rows(*words,trace_backend=True)
    complete_parity_half_rows(*next(iter(centers.values())),certified_psd=True)
    keys={'rollback':set(),'candidate':set()};rows=[];times={'rollback':[],'candidate':[]};tick=perf_counter()
    for repeat in range(args.repeats):
        records={}
        order=('rollback','candidate') if repeat%2==0 else ('candidate','rollback')
        for variant in order:
            started=perf_counter();outputs=[];norm_visits=0;support_orbits=0;psd_certificates=0;trace_candidates=0
            for index,words in enumerate(centers.values()):
                for out in complete_parity_half_rows(*words,
                    certified_psd=args.comparison=='trace' or variant=='candidate',
                    trace_backend=args.comparison=='trace' and variant=='candidate'):
                    outputs.append(signature(out,args.comparison));norm_visits+=out['norm_visited'];support_orbits+=out['completed_support_orbits']
                    trace_candidates+=out['norm_trace_candidates']
                    psd_certificates+=sum('psd_certificate' in r['result'] for r in out['profile_records'])
                    for hit in out['witnesses']:
                        key=discovery_key(*hit['words'])
                        if key not in known:
                            keys[variant].add(key);destination=args.output/variant;destination.mkdir(exist_ok=True)
                            append_verified_solution_if_new(86,*hit['words'],destination)
            elapsed=perf_counter()-started;times[variant].append(elapsed)
            records[variant]=dict(wall_seconds=elapsed,norm_visited=norm_visits,support_orbits=support_orbits,
                                  psd_certificates=psd_certificates,trace_candidates=trace_candidates,outputs=outputs)
        if records['rollback']['outputs']!=records['candidate']['outputs']:
            raise RuntimeError('certified PSD switch changed complete support/phase/full witness results')
        for record in records.values():record.pop('outputs')
        rows.append(dict(repeat=repeat,order=order,variants=records))
        report=dict(complete=len(rows)==args.repeats,comparison=args.comparison,rows=rows,centers=len(centers),fibers_per_replay=4*len(centers),
                    independently_verified_new_classes={k:len(v) for k,v in keys.items()},
                    median_replay_seconds={k:median(v) for k,v in times.items()},
                    local_time_reduction=1-median(times['candidate'])/median(times['rollback']),
                    elapsed=perf_counter()-tick,source_manifest_sha256=hashlib.sha256(
                        json.dumps(manifest,sort_keys=True).encode()).hexdigest(),
                    scope='Same finite half-row fibers, all targets/contents/rotations; named mathematical backend switch only. '
                          'Warm fixed-work local replay, not complete pipeline new-class discovery rate.',
                    claim='No >=20% new-class-discovery claim from local timings or zero/zero discoveries.')
        atomic_write_json(args.output/'report.json',report)
        print(json.dumps(dict(repeat=repeat,variants=records,new_classes=report['independently_verified_new_classes'])),flush=True)


if __name__=='__main__':main()
