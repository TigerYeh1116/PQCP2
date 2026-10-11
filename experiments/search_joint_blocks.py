"""Independent L86 block search with exact integer energies and full audit.

Starts are non-solutions from GS ingredient constructions and prior bounded
joint search. Full PQCP inventories are used only for discovery deduplication.
Each chosen block enumerates ALL candidates in retained necessary contents.
Positive-temperature navigation is heuristic; misses never imply global UNSAT.
"""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import shutil
import signal
import sys
from threading import Event
from time import perf_counter

PROCESS_STARTED=perf_counter()

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from experiments.benchmark_l86_discovery import discovery_key,freeze_sources,verified_inventory
from solver.checkpoint import atomic_write_json
from solver.correlation import full_correlation_profile
from solver.search_runner import append_verified_solution_if_new
from solver.solution_equivalence import pqcp_equivalence_key
from solver.target_profiles import TargetContentProfile,pair_content
from solver.torch_joint_blocks import JointBlockAuditError,exact_joint_block,joint_integer_energy
from solver.torch_joint_search import binary_joint_rows
from solver.verifier import verify_pqcp


def normalize_start(words,k,eta):
    words=tuple(map(tuple,words))
    if verify_pqcp(*words).is_valid:
        raise ValueError('refusing to initialize discovery search from an existing full PQCP')
    if type(k) is not int or not 1<=k<43 or eta not in (-1,1):
        raise ValueError('valid L86 target required')
    # P_new(u)=P_old(q*u). Odd q preserves the row parity contents.
    q=k if k%2 else k//2+(43 if (k//2)%2==0 else 0)
    normalized=tuple(tuple(row[(q*i)%86] for i in range(86)) for row in words)
    target=TargetContentProfile(86,1 if k%2 else 2,eta,*pair_content(*normalized))
    binary_joint_rows(normalized,target)
    old=full_correlation_profile(*words)
    new=full_correlation_profile(*normalized)
    if tuple(new)!=tuple(old[(q*u)%86] for u in range(86)):
        raise RuntimeError('seed target normalization failed independent full PACF identity')
    return normalized,target


def load_starts(gs_path,joint_path,per_target):
    starts=[]
    seen=set()
    for row in json.loads(gs_path.read_text())['seeds']:
        words,target=normalize_start(row['words'],row['k'],row['sign'])
        key=pqcp_equivalence_key(*words)
        if key in seen:continue
        seen.add(key)
        starts.append(dict(words=words,target=target,origin='GS-43 interleaving',initial_joint_energy=joint_integer_energy(words,target)))
    counts=Counter()
    candidates=[item for row in json.loads(joint_path.read_text())['rows'] for item in row['candidates']]
    for row in sorted(candidates,key=lambda r:r['target_energy']):
        words,target=normalize_start((row['A'],row['B']),row['k'],row['sign'])
        case=(target.k,target.eta)
        if counts[case]>=per_target:continue
        key=pqcp_equivalence_key(*words)
        if key in seen:continue
        seen.add(key);counts[case]+=1
        starts.append(dict(words=words,target=target,origin='bounded random joint search',initial_joint_energy=joint_integer_energy(words,target)))
    return starts


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gs-input',type=Path,default=ROOT/'results/L86_GS_interleave_20261007/report.json')
    parser.add_argument('--joint-input',type=Path,default=ROOT/'results/L86_joint_colors_deeper_20261007/report.json')
    parser.add_argument('--block-size',type=int,default=8)
    parser.add_argument('--sweeps',type=int,default=64)
    parser.add_argument('--seeds',type=int,nargs='+',default=(123,456,789))
    parser.add_argument('--temperatures',type=float,nargs='+',default=(0.,1.,4.))
    parser.add_argument('--per-target',type=int,default=3)
    args=parser.parse_args()
    if not 1<=args.block_size<=8 or args.sweeps<1 or args.per_target<0:
        parser.error('bounded positive work required')
    if not torch.backends.mps.is_available():raise RuntimeError('Real MPS required')
    tick=PROCESS_STARTED
    args.output.mkdir(parents=True,exist_ok=False)
    source_sha=freeze_sources(args.output/'source_snapshot')
    for name in ('search_joint_blocks.py','benchmark_l86_discovery.py'):
        destination=args.output/'source_snapshot/experiments'/name
        destination.parent.mkdir(exist_ok=True)
        shutil.copy2(ROOT/'experiments'/name,destination)
    manifest={str(p.relative_to(args.output/'source_snapshot')):hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted((args.output/'source_snapshot').rglob('*'))
              if p.is_file() and p.name!='source_manifest.json'}
    atomic_write_json(args.output/'source_snapshot/source_manifest.json',manifest)
    source_sha=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    inputs={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in (args.gs_input,args.joint_input)}
    for name,path in (('gs_input.json',args.gs_input),('joint_input.json',args.joint_input)):
        shutil.copy2(path,args.output/name)
    initial_inventory=(ROOT/'86.txt').read_bytes() if (ROOT/'86.txt').exists() else b''
    known=verified_inventory(initial_inventory)
    (args.output/'initial_inventory.txt').write_bytes(initial_inventory)
    starts=load_starts(args.gs_input,args.joint_input,args.per_target)
    if not starts:raise RuntimeError('no audited non-solution starts')
    stop=Event()
    signal.signal(signal.SIGINT,lambda *unused:stop.set())
    records,keys=[],set()
    first_hit=None
    def record_zero(words):
        nonlocal first_hit
        key=discovery_key(*words)
        if key in known:return
        if key not in keys and first_hit is None:first_hit=perf_counter()-tick
        keys.add(key);append_verified_solution_if_new(86,*words,args.output)
    total_chains=len(args.temperatures)*len(args.seeds)
    for temperature in args.temperatures:
        for seed in args.seeds:
            rng=random.Random(seed)
            current=[item['words'] for item in starts]
            best=list(current)
            best_energy=[item['initial_joint_energy'] for item in starts]
            configurations=moves=improvements=completed_blocks=content_changes=0
            begun=perf_counter()
            chain_complete=False
            for sweep in range(args.sweeps):
                for lane,item in enumerate(starts):
                    if stop.is_set():break
                    if best_energy[lane]==0:continue
                    positions=tuple(tuple(sorted(rng.sample(range(43),args.block_size))) for _ in range(2))
                    old_nc=tuple(row.count(-1) for row in binary_joint_rows(current[lane],item['target'])[0])
                    try:
                        result=exact_joint_block(current[lane],item['target'],positions,device=torch.device('mps'),
                            temperature=temperature,seed=rng.randrange(2**63),zero_callback=record_zero)
                    except JointBlockAuditError as error:
                        atomic_write_json(args.output/'block_audit_failure.json',dict(error.details,
                            seed=seed,temperature=temperature,sweep=sweep,lane=lane))
                        raise
                    configurations+=result.configurations;completed_blocks+=1
                    moves+=int(result.words!=current[lane])
                    new_nc=tuple(row.count(-1) for row in binary_joint_rows(result.words,item['target'])[0])
                    content_changes+=int(new_nc!=old_nc)
                    current[lane]=result.words
                    if result.minimum_joint_energy<best_energy[lane]:
                        improvements+=1
                        best[lane]=result.minimum_words;best_energy[lane]=result.minimum_joint_energy
                chain_complete=sweep+1==args.sweeps and not stop.is_set()
                if (sweep+1)%4==0 or chain_complete or stop.is_set():
                    row=dict(seed=seed,temperature=temperature,sweeps=sweep+1,complete=chain_complete,
                        blocks=completed_blocks,configurations=configurations,moves=moves,content_changes=content_changes,
                        improved_blocks=improvements,minimum_full_energy=min(best_energy)//2,
                        best=[dict(target=asdict(item['target']),words=pair,joint_energy=value,origin=item['origin'])
                              for item,pair,value in zip(starts,best,best_energy)],elapsed=perf_counter()-begun)
                    report=dict(source_sha256=source_sha,input_sha256=inputs,torch_version=torch.__version__,
                        block_size=args.block_size,starts=len(starts),initial_inventory_classes=len(known),
                        chains=records+[row],complete=len(records)+1==total_chains and chain_complete,
                        independently_verified_new_classes=len(keys),first_verified_elapsed=first_hit,
                        elapsed=perf_counter()-tick,
                        scope='Warm-start bounded exact paired antipodal blocks. Elapsed includes imports, loading, setup and search; excludes generation of supplied input banks. Identical input preparation required for paired discovery-speed claims. No global exclusion.')
                    atomic_write_json(args.output/'report.json',report)
                    print(json.dumps({k:v for k,v in row.items() if k!='best'}),flush=True)
                if stop.is_set():break
            records.append(row)
            if stop.is_set():return


if __name__=='__main__':main()
