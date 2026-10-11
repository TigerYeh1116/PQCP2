"""Frozen exhaustive L43 audit of all weights, blocks and literal ranks.

Four real CRT families cover zero, one, two and three zero norm factors.
The independent oracle enumerates the original full Cartesian product in
Python, groups it by integer weight and preserves its literal ordering.
This is a support-rank audit, not a full PQCP discovery or exclusion.
"""
import argparse
from array import array
import ctypes
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.benchmark_l86_discovery import freeze_sources
from solver.checkpoint import atomic_write_json
from solver.cyclic_support import iter_family,parity_support_family,rotation_representatives
from solver.cyclic_support_native import _library
from solver.cyclic_support_sampling import _native_group,_rotation_step


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'results'/(
        'L86_phase_rank_audit_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')))
    parser.add_argument('--_frozen-runner',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args();output=args.output.resolve();snapshot=output/'source_snapshot'
    if not args._frozen_runner:
        output.mkdir(parents=True,exist_ok=False);freeze_sources(snapshot)
        for name in (Path(__file__).name,'benchmark_l86_discovery.py'):
            dest=snapshot/'experiments'/name;dest.parent.mkdir(exist_ok=True);shutil.copy2(ROOT/'experiments'/name,dest)
        manifest={str(p.relative_to(snapshot)):hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in sorted(snapshot.rglob('*')) if p.is_file() and p.name!='source_manifest.json'}
        atomic_write_json(snapshot/'source_manifest.json',manifest)
        command=[sys.executable,str(snapshot/'experiments'/Path(__file__).name),'--_frozen-runner','--output',str(output)]
        atomic_write_json(output/'execution.json',dict(actual_frozen_execution=True,command=command,cwd=str(snapshot)))
        raise SystemExit(subprocess.run(command,cwd=snapshot).returncode)
    if ROOT!=snapshot:raise RuntimeError('actual archived execution required')
    manifest=json.loads((snapshot/'source_manifest.json').read_text())
    if any(hashlib.sha256((snapshot/n).read_bytes()).hexdigest()!=h for n,h in manifest.items()):
        raise RuntimeError('archived source changed')
    lib=_library();u=ctypes.c_uint64;up=ctypes.POINTER(u);integer=ctypes.c_int
    build=lib.pqcp_weighted_support_grid
    build.argtypes=[up,integer]*3+[u,integer,up,u,up];build.restype=integer
    select=lib.pqcp_weighted_grid_at
    select.argtypes=[up,integer,integer,integer,integer,integer,integer,u,up];select.restype=integer
    initial=parity_support_family(0,43,1);source=0;cases=[];total=0;tick=perf_counter()
    for cancel in range(4):
        if cancel:source^=initial.choices[cancel-1][0]
        family=parity_support_family(source,43,1);expected=[array('Q') for _ in range(44)]
        for mask in iter_family(family):expected[bin(mask).count('1')].append(mask)
        if sum(map(len,expected))!=family.size:raise RuntimeError('independent family oracle is incomplete')
        quotient=rotation_representatives(family)
        first=next((j for j,(a,b) in enumerate(zip(family.choices,quotient.choices)) if len(a)!=len(b)),None)
        groups=tuple(quotient.choices)+((0,),)*(3-len(quotient.choices));sizes=tuple(map(len,groups))
        arrays=tuple(_native_group(43,g) for g in groups)
        blocks=1 if first is None else 43;steps=tuple(_rotation_step(43,g) for g in family.choices)
        width=0 if first is None else len(quotient.choices[first])
        inverse=0 if first is None else pow(steps[first]//width,-1,43)
        capacity=sizes[0]*sizes[1]*((sizes[2]+63)//64);grid=(u*capacity)();count=u();ordinal=u()
        weight_rows=[]
        for weight,bucket in enumerate(expected):
            if build(arrays[0],sizes[0],arrays[1],sizes[1],arrays[2],sizes[2],family.base,weight,grid,capacity,ctypes.byref(count)):
                raise RuntimeError('complete grid build failed')
            per_block=count.value
            if per_block*blocks!=len(bucket):raise RuntimeError('original literal count changed')
            actual=array('Q')
            for block in range(blocks):
                rotation=block*inverse%43;shifts=[0,0,0]
                if first is not None:
                    for j in range(first+1,len(groups)):shifts[j]=steps[j]*rotation%len(groups[j])
                for rank in range(per_block):
                    if select(grid,*sizes,*shifts,rank,ctypes.byref(ordinal))!=1:
                        raise RuntimeError('complete original rank was lost')
                    a,remainder=divmod(ordinal.value,sizes[1]*sizes[2]);b,c=divmod(remainder,sizes[2])
                    mask=family.base^groups[0][a]^groups[1][b]^groups[2][c]
                    if rotation:mask=((mask<<rotation)|(mask>>(43-rotation)))&((1<<43)-1)
                    if mask!=bucket[block*per_block+rank]:raise RuntimeError('literal original support ordering changed')
                    actual.append(mask)
            expected_sha=hashlib.sha256(bucket.tobytes()).hexdigest();actual_sha=hashlib.sha256(actual.tobytes()).hexdigest()
            if expected_sha!=actual_sha:raise RuntimeError('complete support bytes changed')
            weight_rows.append(dict(weight=weight,literal_supports=len(bucket),per_block=per_block,
                                    expected_sha256=expected_sha,actual_sha256=actual_sha))
            total+=len(bucket)
        cases.append(dict(cancelled_norm_components=cancel,u=source,original_family_size=family.size,
                          first_phase_group=first,phase_steps=steps,weights=weight_rows))
        atomic_write_json(output/'report.json',dict(complete=False,actual_frozen_execution=True,
            audited_literal_ranks=total,cases=cases,elapsed=perf_counter()-tick))
    report=dict(complete=True,actual_frozen_execution=True,all_original_counts_masks_and_ranks_match=True,
        audited_literal_ranks=total,cases=cases,elapsed=perf_counter()-tick,
        native_binary_sha256=hashlib.sha256(Path(lib._name).read_bytes()).hexdigest(),
        scope='All literal ranks at all 44 weights in four real L43 CRT families, including all rotation blocks and zero factors. Not full PQCP discoveries or a global search exclusion.')
    atomic_write_json(output/'report.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='cases'}),flush=True)


if __name__=='__main__':main()
