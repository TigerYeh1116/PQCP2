"""Exact GS repair over all cyclic windows, allowing content-family changes."""
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
from solver.boundary_tail_repair import exact_cyclic_window_repair,cyclic_window_center_key
from solver.checkpoint import atomic_write_json
from solver.search_runner import append_verified_solution_if_new


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=ROOT/'results/L86_GS_interleave_20261007/report.json')
    parser.add_argument('--output',type=Path,default=ROOT/'results'/(
        'L86_GS_cyclic_windows_'+datetime.now().strftime('%Y%m%d_%H%M%S')))
    parser.add_argument('--block-size',type=int,default=4)
    args=parser.parse_args()
    if not 1<=args.block_size<=4:parser.error('block size must be 1..4')
    args.output.mkdir(parents=True,exist_ok=False)
    tick=perf_counter()
    snapshot=args.output/'source_snapshot'
    freeze_sources(snapshot)
    for name in ('repair_gs_cyclic_windows.py','benchmark_l86_discovery.py'):
        destination=snapshot/'experiments'/name
        destination.parent.mkdir(exist_ok=True)
        shutil.copy2(ROOT/'experiments'/name,destination)
    manifest={str(p.relative_to(snapshot)):hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(snapshot.rglob('*')) if p.is_file() and p.name!='source_manifest.json'}
    atomic_write_json(snapshot/'source_manifest.json',manifest)
    shutil.copy2(args.input,args.output/'input_gs_seeds.json')
    raw=(ROOT/'86.txt').read_bytes() if (ROOT/'86.txt').exists() else b''
    (args.output/'initial_inventory.txt').write_bytes(raw)
    known=verified_inventory(raw)
    centers={}
    for row in json.loads(args.input.read_text())['seeds']:
        centers.setdefault(cyclic_window_center_key(*row['words']),row['words'])
    records,keys,first_hit=[],set(),None
    for number,center in enumerate(centers.values()):
        result=exact_cyclic_window_repair(*center,block_size=args.block_size)
        for hit in result['witnesses']:
            key=discovery_key(*hit['words'])
            if key not in known:
                if key not in keys and first_hit is None:first_hit=perf_counter()-tick
                keys.add(key);append_verified_solution_if_new(86,*hit['words'],args.output)
        records.append(dict(center=center,result=result))
        report=dict(rows=records,complete=len(records)==len(centers),
                    independently_verified_new_classes=len(keys),first_verified_elapsed=first_hit,
                    elapsed=perf_counter()-tick,block_size=args.block_size,
                    source_manifest_sha256=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest(),
                    scope='Complete cyclic-window neighborhoods of reported non-solution GS centers, '
                          'quotiented ONLY by window-preserving dihedral/complement/row-swap actions. '
                          'No center decimation quotient. Unrestricted contents and strict targets; no global exclusion.')
        atomic_write_json(args.output/'report.json',report)
        print(json.dumps(dict(center=number,queries=result['queries'],covered=result['covered_assignments'],
                              projected=result['projected_assignments'],witnesses=len(result['witnesses']),
                              new_classes=len(keys),elapsed=perf_counter()-tick)),flush=True)


if __name__=='__main__':main()
