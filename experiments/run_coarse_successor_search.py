"""IDE Run: exact new fixed points from the completed hundred-million run.

Only source row 1 is selected. Its G row and the other whole binary row
remain free. This is a new search region, not a speed comparison with both
source-row regions. Existing default entry points are unchanged.
"""
from datetime import datetime
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from time import clock_gettime,CLOCK_MONOTONIC

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.benchmark_l86_discovery import freeze_sources,verified_inventory
from solver.checkpoint import atomic_write_json

RUN_RADIUS=0
RUN_INPUT=ROOT/'results/L86_hundred_million_coarse_frontier_20261009/coarse_witnesses.json'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=RUN_INPUT)
    parser.add_argument('--radius',type=int,choices=(0,1,2),default=RUN_RADIUS)
    parser.add_argument('--resume-closed',type=Path,
                        help='Independently audited closed-anchor certificate; fresh norm caches remain required')
    args=parser.parse_args()
    started=clock_gettime(CLOCK_MONOTONIC)
    source=args.input.resolve();radius=args.radius
    if not source.exists():raise RuntimeError('audited successor anchor bank is required')
    bank=json.loads(source.read_text())
    if bank.get('planned_radius',radius)!=radius:
        raise RuntimeError('anchor exclusions were proved for a different radius')
    if not bank['rows'] or any(row['result'].get('verified') for row in bank['rows']):
        raise RuntimeError('actual compressed successor witnesses, not full solutions, required')
    output=ROOT/'results'/('L86_coarse_successor_radius'+str(radius)+'_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    output.mkdir(exist_ok=False);snapshot=output/'source_snapshot';freeze_sources(snapshot)
    for source_path in (ROOT/'experiments').glob('*.py'):
        target=snapshot/'experiments'/source_path.name;target.parent.mkdir(exist_ok=True);shutil.copy2(source_path,target)
    inventory=(ROOT/'86.txt').read_bytes() if (ROOT/'86.txt').exists() else b''
    verified_inventory(inventory);(snapshot/'86.txt').write_bytes(inventory)
    (output/'input_coarse_witnesses.json').write_bytes(source.read_bytes())
    manifest={str(p.relative_to(snapshot)):hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(snapshot.rglob('*')) if p.is_file() and p.name!='source_manifest.json'}
    atomic_write_json(snapshot/'source_manifest.json',manifest)
    command=[sys.executable,str(snapshot/'experiments/complete_l86_full_rows.py'),
             '--mode','coarse-row','--coarse-input',str(output/'input_coarse_witnesses.json'),
             '--output',str(output/'coverage'),'--anchor-side','1','--radius',str(radius),'--top','0',
             '--two-adic','--dihedral-support','--norm-unsat-cache','4096','--norm-carrier-cache','4096',
             '--unary-batch-free','12','--prebatch-dihedral','--unary-linear-certificate','--native-support-audit']
    if args.resume_closed:command.extend(('--resume-closed',str(args.resume_closed.resolve())))
    atomic_write_json(output/'execution.json',dict(actual_frozen_execution=True,command=command,
        source_manifest_sha256=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest(),
        input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        clock='POSIX CLOCK_MONOTONIC including outer archive/import/setup',start_clock=started,complete=False,
        radius=radius,anchor_side=1,
        scope='Radius-r balls around audited source row 1; all G_A signs and other whole binary row free.'))
    code=subprocess.run(command,cwd=snapshot).returncode
    state=json.loads((output/'execution.json').read_text())
    state.update(exit_code=code,runner_finished=True,elapsed=clock_gettime(CLOCK_MONOTONIC)-started)
    if (output/'coverage/report.json').exists():
        report=json.loads((output/'coverage/report.json').read_text())
        state.update(complete=bool(code==0 and report['complete']),
                     independently_verified_new_classes=report['independently_verified_new_classes'])
    atomic_write_json(output/'execution.json',state)
    if code:raise SystemExit(code)


if __name__=='__main__':main()
