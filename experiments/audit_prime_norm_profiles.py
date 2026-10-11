"""Audit p-squared single-norm obstructions on an entirely closed fiber ledger.

Input must have joint-Gram and trace disabled: its masks are ALL periodic
profile preimages. Retain all positive models; independently verify every
prime certificate. This measures profile cuts, not new PQCP discovery speed.
"""
import argparse
from datetime import datetime
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys
from time import perf_counter

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.benchmark_l86_discovery import freeze_sources
from solver.checkpoint import atomic_write_json
from solver.prime_norm_obstruction import prime_norm_obstruction,verify_prime_norm_certificate
from solver.sds_norm_completion import intersection_profile
from solver.two_half_row_completion import two_half_row_fibers


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent',type=Path,default=ROOT/'results/L86_latest_two_halves_radius2_top8_20261008')
    parser.add_argument('--prime-limit',type=int,default=50000)
    parser.add_argument('--output',type=Path,default=ROOT/'results'/('L86_prime_norm_profile_audit_'+datetime.now().strftime('%Y%m%d_%H%M%S')))
    args=parser.parse_args();parent=json.loads((args.parent/'report.json').read_text())
    if not parent['complete'] or parent.get('joint_gram',False) or parent.get('trace_backend',False):
        raise ValueError('closed periodic-norm-only parent required; conditional cross-trace masks are not all periodic preimages')
    inputs=json.loads((args.parent/'input_centers.json').read_text());expected=[]
    for index,item in enumerate(inputs['seeds']):
        expected.extend((index,j['varied'],j['free'],j['flips'],j['fixed_masks'])
                        for j in two_half_row_fibers(*item['words'],radius=parent['radius']) if j['kind']=='ELIGIBLE')
    ledger=args.parent/parent.get('ledger_file','completion_fibers.jsonl')
    open_file=gzip.open if ledger.suffix=='.gz' else open
    profiles={};visits=0;count=0
    with open_file(ledger,'rt') as handle:
        for line in handle:
            row=json.loads(line);key=(row['center_index'],row['varied'],row['free'],row['flips'],tuple(row['fixed_masks']))
            if count>=len(expected) or key!=expected[count] or not row['result']['complete']:
                raise RuntimeError('original fixed-triple ledger omits, duplicates, reorders or leaves a fiber incomplete')
            count+=1
            if not row.get('cached',False):visits+=row['result']['norm_visited']
            for record in row['result']['profile_records']:
                result=record['result']
                if not result['visited']:continue
                wanted=tuple(record['wanted']);masks=tuple(result['masks'])
                if wanted in profiles and profiles[wanted]['masks']!=masks:
                    raise RuntimeError('same periodic profile has different complete literal rotation representatives')
                profiles.setdefault(wanted,dict(masks=masks,family_size=result['family_size'],occurrences=0))['occurrences']+=1
    if count!=len(expected) or count!=parent['eligible_jobs'] or visits!=parent['norm_visited']:
        raise RuntimeError('closed parent coverage/actual norm visits differ from its input ledger')
    args.output.mkdir(parents=True,exist_ok=False);tick=perf_counter();snapshot=args.output/'source_snapshot';freeze_sources(snapshot)
    (snapshot/'experiments').mkdir(exist_ok=True)
    for p in (ROOT/'experiments').glob('*.py'):shutil.copy2(p,snapshot/'experiments'/p.name)
    manifest={str(p.relative_to(snapshot)):hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(snapshot.rglob('*')) if p.is_file() and p.name!='source_manifest.json'}
    atomic_write_json(snapshot/'source_manifest.json',manifest)
    shutil.copy2(args.parent/'report.json',args.output/'parent_report.json');shutil.copy2(args.parent/'input_centers.json',args.output/'input_centers.json')
    excluded=positive=weighted_excluded=weighted_total=0;prime_counts={}
    with (args.output/'profile_certificates.jsonl').open('w') as handle:
        for wanted,info in profiles.items():
            certificate=prime_norm_obstruction(wanted,limit=args.prime_limit)
            if certificate:
                if info['masks'] or not verify_prime_norm_certificate(certificate):raise RuntimeError('prime norm obstruction pruned an actual binary model or lacked an exact proof')
                excluded+=1;weighted_excluded+=info['family_size']*info['occurrences']
                p=certificate['prime'];prime_counts[p]=prime_counts.get(p,0)+1
            if info['masks']:
                positive+=1
                if any(intersection_profile(m,43)!=wanted for m in info['masks']):raise RuntimeError('parent periodic model failed integer audit')
            weighted_total+=info['family_size']*info['occurrences']
            handle.write(json.dumps(dict(wanted=wanted,**info,certificate=certificate))+'\n')
    report=dict(complete=True,parent_fibers=count,parent_coverage_rebuilt=True,unique_psd_admissible_profiles=len(profiles),
                prime_limit=args.prime_limit,excluded_profiles=excluded,positive_model_profiles=positive,all_positive_models_retained=True,
                weighted_norm_visits_excluded=weighted_excluded,weighted_norm_visits=weighted_total,
                profile_cut_fraction=excluded/len(profiles) if profiles else 0,
                weighted_norm_visit_cut_fraction=weighted_excluded/weighted_total if weighted_total else 0,
                prime_certificate_counts=prime_counts,independently_audited_certificates=excluded,elapsed=perf_counter()-tick,
                parent_ledger_sha256=hashlib.sha256(ledger.read_bytes()).hexdigest(),
                source_manifest_sha256=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest(),
                independently_verified_new_classes=0,
                scope='All distinct PSD-admissible norm profiles of the exact declared parent ledger. All original fibers/order rebuilt. '
                      'Only single integer norm p-squared certificates; all binary models retained. No timing comparison.',
                claim='Profile/required-norm cuts are not >=20% new-PQCP discovery-speed evidence.')
    atomic_write_json(args.output/'report.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
