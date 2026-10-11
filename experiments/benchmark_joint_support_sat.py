"""Compare exact SAT and native completion on actual compressed witnesses.

This is fixed-domain validation, not new-class discovery-speed evidence.
The default SAT runs have no timeout; UNKNOWN remains unresolved if a caller
explicitly sets a finite resource count. Sources and inputs are frozen first.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.benchmark_l86_discovery import freeze_sources
from solver.checkpoint import atomic_write_json
from solver.joint_support_sat import build_joint_support_model,check_joint_support_model
from solver.pqcp_support_solver import complete_pqcp_supports
from solver.target_profiles import TargetContentProfile
from solver.ternary_support_search import support_mask


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=ROOT/'results/L86_independent_norm_long_20261007/coarse_witnesses.json')
    p.add_argument('--output',type=Path,default=ROOT/'results'/('L86_joint_sat_fixed_domains_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')))
    p.add_argument('--top',type=int,default=9)
    p.add_argument('--resource-limit',type=int)
    p.add_argument('--reuse-coarse-witness',action=argparse.BooleanOptionalAction,default=False)
    p.add_argument('--affine-projection',action=argparse.BooleanOptionalAction,default=False)
    p.add_argument('--mod8-projection',action=argparse.BooleanOptionalAction,default=False)
    p.add_argument('--_worker',action='store_true',help=argparse.SUPPRESS)
    args=p.parse_args()
    if args.top<1:p.error('positive top required')
    output=args.output.resolve()
    if not args._worker:
        output.mkdir(parents=True,exist_ok=False);source=output/'source_snapshot';freeze_sources(source)
        (source/'experiments').mkdir()
        for name in ('benchmark_joint_support_sat.py','benchmark_l86_discovery.py'):
            shutil.copy2(ROOT/'experiments'/name,source/'experiments'/name)
        shutil.copy2(args.input,output/'input_coarse_witnesses.json')
        manifest={str(f.relative_to(source)):hashlib.sha256(f.read_bytes()).hexdigest()
                  for f in sorted(source.rglob('*')) if f.is_file() and f.name!='source_manifest.json'}
        atomic_write_json(source/'source_manifest.json',manifest)
        command=[sys.executable,str(source/'experiments/benchmark_joint_support_sat.py'),'--_worker',
                 '--input',str(output/'input_coarse_witnesses.json'),'--output',str(output),'--top',str(args.top)]
        if args.resource_limit is not None:command.extend(['--resource-limit',str(args.resource_limit)])
        if args.reuse_coarse_witness:command.append('--reuse-coarse-witness')
        if args.affine_projection:command.append('--affine-projection')
        if args.mod8_projection:command.append('--mod8-projection')
        return subprocess.run(command,cwd=source).returncode
    inputs=json.loads(args.input.read_text())['rows'][:args.top];rows=[]
    def save():
        atomic_write_json(output/'report.json',dict(rows=rows,planned=len(inputs),tested=len(rows),
            complete=len(rows)==len(inputs) and all(r['statuses_agree'] is True for r in rows),
            input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
            scope='Same literal support/content domains; actual C witnesses are not full PQCP seeds. No new-class discovery-speed claim.'))
    save()
    for index,row in enumerate(inputs):
        target=TargetContentProfile(**row['target'])
        coarse=tuple(tuple(w) for w in row['result']['attempts']['coarse']['words'])
        masks=tuple(support_mask(w) for w in coarse)
        native=complete_pqcp_supports(*coarse,target,masks,max_assignments=2**31)
        problem=build_joint_support_model(*coarse,target,fixed_supports=masks,
            known_coarse_words=coarse if args.reuse_coarse_witness else None,
            affine_projection=args.affine_projection,mod8_projection=args.mod8_projection)
        smt=output/('domain_'+str(index)+'.smt2');smt.write_text(problem['solver'].to_smt2())
        print(json.dumps(dict(event='SAT_start',index=index,source_trial=row['trial'],native_status=native['status'])),flush=True)
        sat=check_joint_support_model(problem,resource_limit=args.resource_limit)
        normalized='UNSAT_BINARY_SUPPORT' if sat['status']=='UNSAT_CONTENT_DOMAIN' else sat['status']
        agree=None if 'UNKNOWN' in (native['status'],normalized) else native['status']==normalized
        rows.append(dict(index=index,source_trial=row['trial'],masks=masks,target=row['target'],native=native,sat=sat,
                         statuses_agree=agree,reused_coarse_witness=args.reuse_coarse_witness,
                         smt2_sha256=hashlib.sha256(smt.read_bytes()).hexdigest()))
        save();print(json.dumps(dict(event='SAT_finish',index=index,status=sat['status'],statuses_agree=agree,
            native_elapsed=native['elapsed'],sat_elapsed=sat['construction_seconds']+sat['solve_seconds'])),flush=True)
        if agree is False:raise RuntimeError('exact solvers disagreed; all input/source evidence retained')
    return 0


if __name__=='__main__':raise SystemExit(main())
