"""IDE-run validation: frozen million-trial discovery, then paired real pipeline.

The CPU experiment and six MPS pipeline runs execute sequentially to avoid
contaminating timing comparisons. Only independent full L86 new-class hits
count. Neither target counts nor coarse SAT can satisfy the speed goal.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.benchmark_l86_discovery import freeze_sources,verified_inventory
from solver.checkpoint import atomic_write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'results'/(
        'L86_alternation_discovery_validation_'+datetime.now().strftime('%Y%m%d_%H%M%S')))
    parser.add_argument('--trials',type=int,default=1000000)
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--norm-seed',type=int,default=20261007)
    parser.add_argument('--seconds',type=float,default=600.)
    parser.add_argument('--batch',type=int,default=16384)
    parser.add_argument('--seeds',type=int,nargs='+',default=(123,456,789))
    args=parser.parse_args()
    if args.trials<1 or not 1<=args.workers<=16 or args.seconds<=0 or args.batch<1:
        parser.error('positive bounded experiment controls required')
    if len(set(args.seeds))!=len(args.seeds):
        parser.error('distinct paired seeds required')
    args.output=args.output.resolve()
    args.output.mkdir(parents=True,exist_ok=False)
    started=perf_counter()
    source=args.output/'source'
    freeze_sources(source)
    for name in ('run_alternation_discovery_validation.py','search_norm_supports.py','benchmark_l86_discovery.py'):
        destination=source/'experiments'/name
        destination.parent.mkdir(exist_ok=True)
        shutil.copy2(ROOT/'experiments'/name,destination)
    manifest={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(source.rglob('*')) if p.is_file() and p.name!='source_manifest.json'}
    atomic_write_json(source/'source_manifest.json',manifest)
    initial=(ROOT/'86.txt').read_bytes() if (ROOT/'86.txt').exists() else b''
    (args.output/'initial_inventory.txt').write_bytes(initial)
    if initial:(source/'86.txt').write_bytes(initial)
    report=dict(complete=False,active_stage='norm_discovery',norm_trials=args.trials,
                norm_workers=args.workers,norm_seed=args.norm_seed,paired_seconds=args.seconds,
                paired_batch=args.batch,paired_seeds=args.seeds,
                source_manifest_sha256=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest(),
                scope='Sequential frozen broad discovery and paired full-pipeline validation. '
                      'Completion of experiments alone does not establish >=20% new-class discovery improvement.')
    atomic_write_json(args.output/'report.json',report)
    print(json.dumps(dict(event='norm_start',output=str(args.output),trials=args.trials)),flush=True)
    code=subprocess.run([sys.executable,str(source/'experiments/search_norm_supports.py'),
        '--output',str(args.output/'norm_search'),'--trials',str(args.trials),
        '--workers',str(args.workers),'--seed',str(args.norm_seed),'--alternation-orbit-pruning']).returncode
    report.update(norm_returncode=code,elapsed=perf_counter()-started)
    if code:
        report['active_stage']='norm_failed'
        atomic_write_json(args.output/'report.json',report)
        return code
    norm=json.loads((args.output/'norm_search/report.json').read_text())
    if not norm['complete']:
        raise RuntimeError('norm experiment did not complete; paired timing must not overlap a live CPU job')
    report.update(norm_complete=True,norm_new_classes=norm['independently_verified_new_classes'],
                  active_stage='paired_pipeline')
    new_archive=args.output/'norm_search/86.txt'
    combined=initial
    if new_archive.exists():
        combined=(initial.rstrip(b'\n')+b'\n' if initial else b'')+new_archive.read_bytes()
    verified_inventory(combined)
    inventory=args.output/'inventory_after_norm.txt'
    inventory.write_bytes(combined)
    atomic_write_json(args.output/'report.json',report)
    print(json.dumps(dict(event='paired_start',norm_new=norm['independently_verified_new_classes'])),flush=True)
    code=subprocess.run([sys.executable,str(source/'experiments/benchmark_l86_discovery.py'),
        '--output',str(args.output/'paired_pipeline'),'--comparison','alternation',
        '--inventory',str(inventory),'--seconds',str(args.seconds),'--batch',str(args.batch),
        '--seeds',*(str(v) for v in args.seeds)]).returncode
    paired_path=args.output/'paired_pipeline/report.json'
    paired=json.loads(paired_path.read_text()) if paired_path.exists() else None
    report.update(paired_returncode=code,complete=code==0 and paired is not None and paired['complete'],
                  active_stage='finished' if code==0 else 'paired_failed',elapsed=perf_counter()-started,
                  paired_new_classes=({variant:sum(r['independently_verified_new_classes'] for r in paired['rows']
                                                  if r['variant']==variant) for variant in ('rollback','candidate')}
                                      if paired is not None else None),
                  claim='No >=20% discovery-speed claim without independently verified new-class evidence.')
    atomic_write_json(args.output/'report.json',report)
    print(json.dumps(report),flush=True)
    return code


if __name__=='__main__':raise SystemExit(main())
