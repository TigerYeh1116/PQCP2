"""IDE Run: exact bounded trades for a positive even-peak L86 construction.

Sum-3 rows target -2+8*delta; sum-9 rows target 2-4*delta. Their four-row
sum fixes every even sidelobe. The remaining odd cross profiles must match
exactly. This is a restricted construction, not a global PQCP exclusion.
"""
import argparse
import hashlib
from itertools import product
import json
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter
from datetime import datetime

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.benchmark_l86_discovery import discovery_key,freeze_sources,verified_inventory
from experiments.search_doptimal43_interleave import read_source
from experiments.search_perfect43_basis import perfect43_bases
from solver.checkpoint import atomic_write_json
from solver.interleaving import interleave_bits,odd_cross_vectors,periodic_sign_profile
from solver.near_norm_trades import content_trade_groups,exact_profile_pairs,signs,zero_odd_offset_pairs
from solver.search_runner import append_verified_solution_if_new
from solver.structured_energy import target_profile
from solver.verifier import verify_pqcp


def audit_pair(a,b,beta,height,k):
    rows=(signs(a['mask'],43),signs(b['mask'],43))
    actual=tuple(x+y for x,y in zip(*map(periodic_sign_profile,rows)))
    wanted=(86,)+tuple(beta+height*int(u in (k,43-k)) for u in range(1,43))
    if actual!=wanted:raise RuntimeError('packed hit failed independent all-shift integer audit')
    return rows


def normalized(rows,k):
    return tuple(tuple(row[k*i%43] for i in range(43)) for row in rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'results/L86_doptimal_source_20261007.html')
    parser.add_argument('--output',type=Path,default=ROOT/'results'/(
        'L86_near_norm_even_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')))
    parser.add_argument('--flip-limit',type=int,choices=(1,3,5),default=5)
    parser.add_argument('--_frozen-runner',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args();output=args.output.resolve();snapshot=output/'source_snapshot'
    if not args._frozen_runner:
        output.mkdir(parents=True,exist_ok=False);freeze_sources(snapshot)
        for name in (Path(__file__).name,'benchmark_l86_discovery.py','search_perfect43_basis.py','search_doptimal43_interleave.py'):
            dest=snapshot/'experiments'/name;dest.parent.mkdir(exist_ok=True);shutil.copy2(ROOT/'experiments'/name,dest)
        shutil.copy2(args.source,output/'input_source.html')
        inventory=(ROOT/'86.txt').read_bytes() if (ROOT/'86.txt').exists() else b''
        verified_inventory(inventory);(snapshot/'86.txt').write_bytes(inventory)
        (output/'initial_inventory.txt').write_bytes(inventory)
        (output/'86.txt').write_bytes(inventory)
        manifest={str(p.relative_to(snapshot)):hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in sorted(snapshot.rglob('*')) if p.is_file() and p.name!='source_manifest.json'}
        atomic_write_json(snapshot/'source_manifest.json',manifest)
        command=[sys.executable,str(snapshot/'experiments'/Path(__file__).name),'--_frozen-runner',
            '--source',str(output/'input_source.html'),'--output',str(output),'--flip-limit',str(args.flip_limit)]
        atomic_write_json(output/'execution.json',dict(actual_frozen_execution=True,command=command,cwd=str(snapshot),
            input_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest()))
        raise SystemExit(subprocess.run(command,cwd=snapshot).returncode)
    if ROOT!=snapshot:raise RuntimeError('actual archived execution required')
    manifest=json.loads((snapshot/'source_manifest.json').read_text())
    if any(hashlib.sha256((snapshot/name).read_bytes()).hexdigest()!=digest for name,digest in manifest.items()):
        raise RuntimeError('archived source changed')
    tick=perf_counter();basis=perfect43_bases();groups={};variants=0
    for source,item in enumerate(basis['rows']):
        current=content_trade_groups(item['signs'],3,args.flip_limit);variants+=current['literal_variants']
        for code,models in current['groups'].items():
            for model in models:groups.setdefault(code,[]).append(dict(model,source=source))
    legendre=dict(d=43,target_sum=3,max_flips=args.flip_limit,groups=groups,literal_variants=variants)
    leg_pairs=[];near_pairs=[];near_lookups=0;near_pair_space=0
    for k in range(1,22):
        wanted=tuple(-2+8*int(u==k) for u in range(1,22))
        for a,b in exact_profile_pairs(legendre,legendre,wanted):
            rows=audit_pair(a,b,-2,8,k)
            leg_pairs.append(dict(rows=normalized(rows,k),k=k,origins=(a,b)))
    parents,rejected=read_source(args.source)
    for parent in parents:
        if parent['weights']!=(16,18):continue
        left=content_trade_groups(parent['rows'][0],9,args.flip_limit)
        right=content_trade_groups(parent['rows'][1],9,args.flip_limit)
        near_pair_space+=left['literal_variants']*right['literal_variants']
        near_lookups+=21*len(left['groups'])
        for k in range(1,22):
            wanted=tuple(2-4*int(u==k) for u in range(1,22))
            for a,b in exact_profile_pairs(left,right,wanted):
                rows=audit_pair(a,b,2,-4,k)
                near_pairs.append(dict(rows=normalized(rows,k),k=k,parent=parent['source_index'],origins=(a,b)))
    atomic_write_json(output/'near_pairs.json',dict(sum3_pairs=leg_pairs,sum9_pairs=near_pairs,
        claim='Integer length43 pair ingredients only, not full L86 PQCP discoveries.'))
    initial_keys=set(verified_inventory((snapshot/'86.txt').read_bytes()))
    keys=set();witnesses=[];lookups=0;first=None
    for near,leg in product(near_pairs,leg_pairs):
        for swapped in (False,True):
            odd=leg['rows'][::-1] if swapped else leg['rows']
            for reflections in product((False,True),repeat=4):
                rows=tuple(tuple(row[-i%43] for i in range(43)) if reverse else tuple(row)
                           for row,reverse in zip(near['rows']+odd,reflections))
                statistics={}
                for da,db in zero_odd_offset_pairs(rows[0],rows[2],rows[1],rows[3],statistics=statistics):
                    words=(interleave_bits(rows[0],rows[2],da),interleave_bits(rows[1],rows[3],db,-1))
                    check=verify_pqcp(*words)
                    if not check.is_valid or tuple(check.profile)!=target_profile(86,2,1):
                        raise RuntimeError('even-peak construction failed independent full verifier')
                    key=discovery_key(*words)
                    if key not in initial_keys and first is None:first=perf_counter()-tick
                    keys.add(key);append_verified_solution_if_new(86,*words,output)
                    witnesses.append(dict(words=words,k=2,eta=1,offsets=(da,db),reflections=reflections))
                lookups+=statistics['profile_lookups']
    report=dict(complete=True,actual_frozen_execution=True,flip_limit=args.flip_limit,
        perfect_source_classes=len(basis['rows']),sum3_literal_variants=variants,sum3_profile_groups=len(groups),
        sum3_complete_pair_space=variants*variants,sum3_near_pairs=len(leg_pairs),sum9_near_pairs=len(near_pairs),
        sum9_complete_pair_space=near_pair_space,
        validated_doptimal_sources=len(parents),rejected_doptimal_sources=rejected,
        sum9_exact_lookups=near_lookups,odd_exact_lookups=lookups,
        independently_verified_full_classes=len(keys),independently_verified_new_classes=len(keys-initial_keys),
        initial_verified_classes=len(initial_keys),first_verified_elapsed=first,elapsed=perf_counter()-tick,
        witnesses=witnesses,scope='All content-feasible trades through the reported Hamming bound from complete perfect43 translation classes and validated published sum11/7 D-optimal pairs; all reflections and relative phases after common peak normalization. Restricted positive-even-peak construction only.')
    atomic_write_json(output/'report.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('witnesses','rejected_doptimal_sources')},sort_keys=True),flush=True)


if __name__=='__main__':main()
