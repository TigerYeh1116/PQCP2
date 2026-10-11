"""Paired FIXED-WORK local-repair benchmark; no global discovery-speed claim.

The sole switch moves exact DC/ALT requirements into the mod-8 index key.
Both variants keep incremental row sums. Alternating order, warm compilation,
frozen inputs and audited coverage/check counts isolate the mathematical filter.
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
from solver.boundary_tail_repair import cyclic_window_center_key
from solver.checkpoint import atomic_write_json
from solver.search_runner import append_verified_solution_if_new
from solver.unrestricted_flip_join import exact_unrestricted_flip_repair


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=ROOT/'results/L86_GS_interleave_20261007/report.json')
    parser.add_argument('--output',type=Path,default=ROOT/'results'/(
        'L86_unrestricted_grouping_benchmark_'+datetime.now().strftime('%Y%m%d_%H%M%S')))
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--radius',type=int,default=4)
    args=parser.parse_args()
    if args.repeats<1 or not 0<=args.radius<=5:parser.error('positive repeats and radius 0..5 required')
    args.output.mkdir(parents=True,exist_ok=False);snapshot=args.output/'source_snapshot';freeze_sources(snapshot)
    for name in ('benchmark_unrestricted_grouping.py','benchmark_l86_discovery.py'):
        dest=snapshot/'experiments'/name;dest.parent.mkdir(exist_ok=True)
        shutil.copy2(ROOT/'experiments'/name,dest)
    manifest={str(p.relative_to(snapshot)):hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(snapshot.rglob('*')) if p.is_file() and p.name!='source_manifest.json'}
    atomic_write_json(snapshot/'source_manifest.json',manifest)
    shutil.copy2(args.input,args.output/'input.json')
    raw=(ROOT/'86.txt').read_bytes() if (ROOT/'86.txt').exists() else b''
    (args.output/'initial_inventory.txt').write_bytes(raw);known=verified_inventory(raw)
    centers={}
    for row in json.loads(args.input.read_text())['seeds']:
        centers.setdefault(cyclic_window_center_key(*row['words']),row['words'])
    if not centers:raise ValueError('non-solution centers required')
    # Deterministic tiny warm-up compiles the one native library before timing.
    for words in centers.values():exact_unrestricted_flip_repair(*words,radius=0)
    rows=[];keys={'rollback':set(),'candidate':set()};times={'rollback':[],'candidate':[]}
    tick=perf_counter()
    for repeat in range(args.repeats):
        for index,words in enumerate(centers.values()):
            records={}
            order=('rollback','candidate') if (repeat+index)%2==0 else ('candidate','rollback')
            for variant in order:
                begun=perf_counter()
                out=exact_unrestricted_flip_repair(*words,radius=args.radius,content_grouping=variant=='candidate')
                elapsed=perf_counter()-begun;times[variant].append(elapsed);records[variant]=dict(result=out,wall_seconds=elapsed)
                for hit in out['witnesses']:
                    key=discovery_key(*hit['words'])
                    if key not in known:
                        keys[variant].add(key)
                        destination=args.output/variant;destination.mkdir(exist_ok=True)
                        append_verified_solution_if_new(86,*hit['words'],destination)
            a,b=(records[v]['result'] for v in ('rollback','candidate'))
            if a['status']!=b['status'] or a['existence_decided']!=b['existence_decided']:
                raise RuntimeError('grouping changed existence decision')
            if a['complete'] and b['complete']:
                for field in ('enumerated_per_row','raw_product_configurations','indexed_configurations','full_coordinate_checks'):
                    if a[field]!=b[field]:raise RuntimeError('grouping changed complete coverage/checks: '+field)
                if a['exact_signature_lookups']!=2*b['exact_signature_lookups']:
                    raise RuntimeError('DC/ALT target-parity lookup identity failed')
            rows.append(dict(repeat=repeat,center_index=index,order=order,variants=records))
            report=dict(complete=len(rows)==len(centers)*args.repeats,rows=rows,radius_per_row=args.radius,
                        independently_verified_new_classes={k:len(v) for k,v in keys.items()},
                        median_local_call_seconds={k:median(v) for k,v in times.items()},
                        local_time_reduction=1-median(times['candidate'])/median(times['rollback']),
                        elapsed=perf_counter()-tick,source_manifest_sha256=hashlib.sha256(
                            json.dumps(manifest,sort_keys=True).encode()).hexdigest(),
                        scope='Identical finite product balls, exact DC/ALT index switch only. '
                              'Local fixed-work timing is NOT full-pipeline new-L86-class discovery rate.',
                        claim='No >=20% new-class discovery claim from local timing or zero/zero discoveries.')
            atomic_write_json(args.output/'report.json',report)
            print(json.dumps(dict(pair=len(rows),center=index,repeat=repeat,
                                  seconds={k:v['wall_seconds'] for k,v in records.items()},
                                  new_classes=report['independently_verified_new_classes'])),flush=True)


if __name__=='__main__':main()
