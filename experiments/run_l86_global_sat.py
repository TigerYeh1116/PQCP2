"""IDE Run: jointly solve both unknown rows in exact global L86 SAT families.

The normalized k/eta families cover every strict target up to proved row
symmetries and common unit permutations. Candidate adds character and
intersection constraints; a negative-even XOR baseline runs alongside it.
All full models require independent integer verification and novelty audit.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter,sleep

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.benchmark_l86_discovery import freeze_sources,verified_inventory,discovery_key
from solver.checkpoint import atomic_write_json
from solver.global_pqcp_sat import normalized_cases,build_global_solver
from solver.search_runner import append_verified_solution_if_new
from solver.verifier import verify_pqcp

RUN_ROUNDS=6
RUN_SLICE_SECONDS=60


def worker(case_index,variant,output,rounds,slice_seconds):
    import z3
    case=normalized_cases(43)[case_index]
    known=verified_inventory((ROOT/'86.txt').read_bytes())
    tick=perf_counter();s,variables=build_global_solver(case,character_pruning=variant=='candidate',seed=20261011)
    s.set(max_memory=4096)  # Four independent native solvers remain bounded on 32 GiB.
    construction=perf_counter()-tick
    atomic_write_json(output/'formula.json',dict(case=case,variant=variant,z3_version=z3.get_version_string(),construction_seconds=construction,
        bool_inputs=172,encoding='intersection' if variant=='candidate' else 'xor',fixed_coarse_row=False))
    (output/'formula.smt2').write_text(s.to_smt2())
    for iteration in range(rounds):
        budget_seconds=slice_seconds*min(2**iteration,16)
        s.set(timeout=budget_seconds*1000);start=perf_counter();outcome=s.check()
        state=dict(complete=False,status=str(outcome),iteration=iteration+1,elapsed=perf_counter()-tick,
            check_seconds=perf_counter()-start,timeout_seconds=budget_seconds,case=case,variant=variant,independently_verified_new_classes=0,
            statistics={str(k):v for k,v in s.statistics()},reason=s.reason_unknown() if outcome==z3.unknown else None)
        if outcome==z3.sat:
            m=s.model();words=tuple(tuple(int(z3.is_true(m.eval(v,model_completion=True))) for v in row) for row in variables)
            check=verify_pqcp(*words)
            if (not check.is_valid or check.nonzero_shifts!=(case['k'],86-case['k'])
                    or check.profile[case['k']]!=4*case['eta']):
                raise RuntimeError('global SAT model failed independent original all-shift integer verifier')
            new=discovery_key(*words) not in known
            append_verified_solution_if_new(86,*words,output)
            state.update(complete=True,words=words,independently_verified_new_classes=int(new),
                         first_verified_new_seconds=perf_counter()-tick if new else None)
        elif outcome==z3.unsat:
            state.update(complete=True,scope='One normalized content family UNSAT; global nonexistence requires all families and a separate proof audit.')
        else:
            state['scope']='UNKNOWN is no exclusion. Repeated checks retain the original solver object and formula.'
        atomic_write_json(output/'report.json',state)
        if outcome!=z3.unknown:return
        if 'memory' in state['reason'].lower():return  # No repeated out-of-memory checks and no false UNSAT.


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'results'/('L86_global_sat_'+datetime.now().strftime('%Y%m%d_%H%M%S')))
    p.add_argument('--rounds',type=int,default=RUN_ROUNDS);p.add_argument('--slice-seconds',type=int,default=RUN_SLICE_SECONDS)
    p.add_argument('--_worker',type=int,help=argparse.SUPPRESS);p.add_argument('--variant',choices=('baseline','candidate'),default='candidate')
    args=p.parse_args();output=args.output.resolve()
    if args.rounds<1 or args.slice_seconds<1:p.error('positive round and time budgets required')
    if args._worker is not None:
        worker(args._worker,args.variant,output,args.rounds,args.slice_seconds);return
    output.mkdir(parents=True,exist_ok=False);snapshot=output/'source_snapshot';freeze_sources(snapshot)
    for x in (ROOT/'experiments').glob('*.py'):
        dest=snapshot/'experiments'/x.name;dest.parent.mkdir(exist_ok=True);shutil.copy2(x,dest)
    raw=(ROOT/'86.txt').read_bytes() if (ROOT/'86.txt').exists() else b'';verified_inventory(raw)
    (snapshot/'86.txt').write_bytes(raw);(output/'initial_inventory.txt').write_bytes(raw)
    manifest={str(x.relative_to(snapshot)):hashlib.sha256(x.read_bytes()).hexdigest() for x in sorted(snapshot.rglob('*'))
              if x.is_file() and x.name!='source_manifest.json'}
    atomic_write_json(snapshot/'source_manifest.json',manifest)
    jobs=[]
    for index,variant in ((0,'candidate'),(1,'candidate'),(2,'candidate'),(2,'baseline')):
        folder=output/('case{}_{}'.format(index,variant));folder.mkdir()
        command=[sys.executable,str(snapshot/'experiments'/Path(__file__).name),'--_worker',str(index),'--variant',variant,
                 '--output',str(folder),'--rounds',str(args.rounds),'--slice-seconds',str(args.slice_seconds)]
        log=(folder/'console.log').open('w');child=subprocess.Popen(command,cwd=snapshot,stdout=log,stderr=subprocess.STDOUT)
        atomic_write_json(folder/'execution.json',dict(actual_frozen_execution=True,command=command,pid=child.pid))
        jobs.append((child,log,folder))
    atomic_write_json(output/'execution.json',dict(actual_frozen_execution=True,worker_directories=[str(x[2]) for x in jobs],
        source_manifest=manifest,comparison='Case 2 baseline XOR versus candidate character/intersection encoding, same strict target and initial inventory. Other cases are independent exploratory searches.'))
    while any(child.poll() is None for child,_,_ in jobs):
        for child,_,folder in jobs:
            if child.poll() is not None and child.returncode:
                raise RuntimeError('worker failed; inspect '+str(folder/'console.log'))
        sleep(1)
    for child,log,folder in jobs:
        log.close()
        if child.returncode:raise RuntimeError('worker failed; inspect '+str(folder/'console.log'))
    states=[json.loads((folder/'report.json').read_text()) for _,_,folder in jobs]
    initial=verified_inventory((output/'initial_inventory.txt').read_bytes());new=set()
    for _,_,folder in jobs:
        path=folder/'86.txt'
        if path.exists():new.update(verified_inventory(path.read_bytes())-initial)
    atomic_write_json(output/'report.json',dict(complete=all(x['complete'] for x in states),workers=states,
        independently_verified_new_classes=len(new),
        claim='Actual global SAT outcomes; UNKNOWN is no exclusion. First-new speed requires a valid matched latency comparison.'))


if __name__=='__main__':main()
