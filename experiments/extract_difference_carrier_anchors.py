"""Harvest exact G sign witnesses as new radius-zero C anchor candidates.

For odd half length d and strict k<d, the modulated difference pair has
NC sum d*delta_0 + eta*(-1)^k*(delta_k+delta_-k). This is an actual coarse
witness for the transformed sign, even if its original fixed-support C
problem is UNSAT. Original C/G failures never exclude another domain.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from time import monotonic

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.benchmark_l86_discovery import freeze_sources
from experiments.extract_coarse_successor_anchors import closed_bank,file_sha
from solver.checkpoint import atomic_write_json
from solver.coarse_anchor_orbits import coarse_anchor_key
from solver.odd_lift_native import difference_sign_problem
from solver.structured_energy import fold_profile,target_profile,integer_periodic_autocorrelation_profile as pacf
from solver.target_profiles import TargetContentProfile


def difference_coarse_target(length,k,eta):
    if (type(length) is not int or not 6<=length<=94 or length%2 or not (length//2)%2
            or type(k) is not int or not 1<=k<length//2 or type(eta) is not int or eta not in (-1,1)):
        raise ValueError('strict non-half-shift target with an odd half length required')
    return dict(L=length,k=k,eta=eta*(-1 if k%2 else 1))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=ROOT/'results/L86_projection_groups_twenty_million_20261008')
    parser.add_argument('--closed',type=Path,action='append',default=[])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--_frozen',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args();args.output=args.output.resolve();args.input=args.input.resolve()
    args.closed=[p.resolve() for p in args.closed]
    if not args._frozen:
        args.output.mkdir(parents=True,exist_ok=False);snapshot=args.output/'source_snapshot';freeze_sources(snapshot)
        for name in ('extract_difference_carrier_anchors.py','extract_coarse_successor_anchors.py','benchmark_l86_discovery.py'):
            dst=snapshot/'experiments'/name;dst.parent.mkdir(exist_ok=True);shutil.copy2(ROOT/'experiments'/name,dst)
        manifest={str(p.relative_to(snapshot)):file_sha(p) for p in sorted(snapshot.rglob('*'))
                  if p.is_file() and p.name!='source_manifest.json'}
        atomic_write_json(snapshot/'source_manifest.json',manifest)
        command=[sys.executable,str(snapshot/'experiments/extract_difference_carrier_anchors.py'),
                 '--_frozen','--input',str(args.input),'--output',str(args.output)]
        for bank in args.closed:command.extend(['--closed',str(bank)])
        code=subprocess.run(command,cwd=snapshot).returncode
        if code:raise SystemExit(code)
        return
    if ROOT!=(args.output/'source_snapshot'):raise RuntimeError('actual frozen execution required')
    manifest=json.loads((ROOT/'source_manifest.json').read_text())
    if any(file_sha(ROOT/name)!=sha for name,sha in manifest.items()):raise RuntimeError('frozen source changed')
    source_report=json.loads((args.input/'report.json').read_text())
    if not source_report['complete'] or not source_report['all_sampled_supports_decided']:
        raise RuntimeError('terminal, fully decided source trace required')
    trace=args.input/source_report['trace_file'];trace_sha=file_sha(trace)
    closed={};proofs=[]
    for bank in args.closed:
        _,keys=closed_bank(bank);closed.update(keys)
        proofs.append(dict(path=str(bank),certificate_sha256=file_sha(bank/'certificate.json')))
    rows=[];seen=set();observations=[];scanned=carriers=g_only=0;started=monotonic();next_update=started+30
    with gzip.open(trace,'rb') as stream:
        for line in stream:
            scanned+=1
            if monotonic()>=next_update:
                atomic_write_json(args.output/'progress.json',dict(complete=False,lines=scanned,
                    audited_g_carriers=carriers,retained_anchor_points=len(rows),elapsed=monotonic()-started))
                print(json.dumps(dict(lines=scanned,audited_g_carriers=carriers,retained_anchor_points=len(rows))),flush=True)
                next_update=monotonic()+30
            if b'SAT_COMPRESSED' not in line:continue
            source=json.loads(line)
            if source['trial']!=scanned-1:raise RuntimeError('carrier trial provenance mismatch')
            result=source.get('result')
            if result is None:continue
            attempts=result['attempts']
            if 'lifts' not in attempts:raise RuntimeError('grouped trace schema required')
            for lift in attempts['lifts']:
                out=lift['result']
                if out is None or out['status']!='SAT_COMPRESSED':continue
                target=TargetContentProfile(**lift['target']);masks=(source['u'],source['sampling']['mask'])
                gm,wanted,negative=difference_sign_problem(masks,target)
                words=tuple(tuple(w) for w in out['words'])
                if (len(words)!=2 or any(len(w)!=43 or any(type(v) is not int or v not in (-1,0,1) for v in w) for w in words)
                        or tuple(sum((v!=0)<<j for j,v in enumerate(w)) for w in words)!=gm
                        or tuple(w.count(-1) for w in words)!=negative
                        or tuple(x+y for x,y in zip(pacf(words[0]),pacf(words[1])))!=wanted):
                    raise RuntimeError('G carrier failed independent all-shift integer/support/count audit')
                transformed=difference_coarse_target(target.L,target.k,target.eta)
                c_wanted=tuple(v//4 for v in fold_profile(target_profile(transformed['L'],
                    transformed['k'],transformed['eta']),2))
                if c_wanted!=wanted:raise RuntimeError('difference-to-coarse target identity failed')
                carriers+=1
                failed_c=(attempts.get('coarse') or {}).get('status')=='UNSAT_FIXED_SUPPORT'
                g_only+=int(failed_c)
                for side,old_word in enumerate(words):
                    key=coarse_anchor_key(tuple(old_word[2*j%43] for j in range(43)))
                    origin=dict(trial=source['trial'],original_target=lift['target'],side=side,original_c_unsat=failed_c)
                    if key in closed:
                        observations.append(dict(origin,status='ALREADY_CLOSED_ALL_84_TARGETS',proof=closed[key]));continue
                    if key in seen:
                        observations.append(dict(origin,status='DUPLICATE_RADIUS_ZERO_POINT'));continue
                    seen.add(key);pair=words if side==1 else words[::-1]
                    rows.append(dict(target=transformed,result=dict(verified=False,
                        attempts=dict(coarse=dict(status='SAT_COMPRESSED',words=pair))),source=origin))
                    observations.append(dict(origin,status='NEW_G_DERIVED_RADIUS_ZERO_POINT'))
    if scanned!=source_report['tested'] or file_sha(trace)!=trace_sha:
        raise RuntimeError('source trace count or immutable input changed')
    atomic_write_json(args.output/'coarse_witnesses.json',dict(rows=rows,planned_radius=0,
        scope='Audited modulated difference witnesses as coarse pairs. Radius zero only; no full binary discovery claim.'))
    report=dict(complete=True,actual_frozen_execution=True,source_lines=scanned,audited_g_carriers=carriers,
        original_c_unsat_carriers=g_only,new_radius_zero_anchor_points=len(rows),observations=observations,
        input_trace_sha256=trace_sha,input_report_sha256=file_sha(args.input/'report.json'),closed_proofs=proofs,
        source_manifest_sha256=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest(),
        scope='Existing G witnesses converted using an exact norm identity. Full old support failures are not transferred.',
        claim='Eligible new anchor regions only; no full PQCP or discovery-speed claim.',elapsed=monotonic()-started)
    atomic_write_json(args.output/'report.json',report);print(json.dumps({k:v for k,v in report.items() if k!='observations'}),flush=True)


if __name__=='__main__':main()
