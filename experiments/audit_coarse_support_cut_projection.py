"""Actual compressed-witness G fibers: exact cut audit and paired timing.

Fixed-support timing is explicitly not full-region discovery timing.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
from statistics import median
import sys
from time import perf_counter

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.complete_l86_full_rows import audited_coarse_centers
from experiments.benchmark_l86_discovery import freeze_sources
from solver.coarse_row_completion import coarse_target
from solver.checkpoint import atomic_write_json
from solver.structured_energy import integer_periodic_autocorrelation_profile as pacf
from solver.support_cut_projection import support_cut_obstruction
from solver.ternary_system_native import prepare_native_system,run_native_system
from solver.full_row_completion import reconstruct_crt_binary
from solver.verifier import verify_pqcp


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=ROOT/'results/L86_independent_norm_long_20261007/coarse_witnesses.json')
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--output',type=Path,default=ROOT/'results'/('L86_coarse_cut_support_audit_'+datetime.now().strftime('%Y%m%d_%H%M%S')))
    args=p.parse_args()
    if args.repeats<1:p.error('positive repeat count required')
    inputs=audited_coarse_centers(args.input,0);cases=[];all_bits=(1<<43)-1
    for parent,item in enumerate(inputs['seeds']):
        for side in (0,1):
            ca,cb=(tuple(item['coarse_rows'][j]) for j in (side,1-side))
            for k in range(1,43):
                for eta in (-1,1):
                    spec=coarse_target(ca,k,eta)
                    if spec is None or spec['wanted_c_b']!=tuple(pacf(cb)) or spec['c_b_negative']!=cb.count(-1):continue
                    masks=all_bits^spec['c_mask'],all_bits^sum((v!=0)<<j for j,v in enumerate(cb))
                    for negative in spec['g_negative_pairs']:
                        cases.append(dict(parent=parent,side=side,ca=ca,cb=cb,k=k,eta=eta,masks=masks,
                                          negative=negative,wanted=spec['wanted_g_pair']))
    if not cases:raise ValueError('no exact compressed support fibers')
    args.output.mkdir(parents=True,exist_ok=False);freeze_sources(args.output/'source_snapshot')
    atomic_write_json(args.output/'input_cases.json',cases)
    def run(enabled):
        tick=perf_counter();results=[]
        for case in cases:
            proof=support_cut_obstruction(case['masks'],43,case['wanted'],case['negative']) if enabled else None
            out=(dict(status='UNSAT_FIXED_SUPPORT',words=None,assignments=0,cut_certificate=proof) if proof else
                 run_native_system(prepare_native_system(case['masks'],43,case['wanted'],case['negative']),2**31))
            if out['words']:
                ga,gb=out['words'];words=reconstruct_crt_binary(case['ca'],ga),reconstruct_crt_binary(case['cb'],gb)
                check=verify_pqcp(*words)
                if not check.is_valid or check.nonzero_shifts!=(case['k'],86-case['k']) or check.profile[case['k']]!=4*case['eta']:
                    raise RuntimeError('G support fiber output failed original verifier')
                out['full_words']=words
            results.append(out)
        return dict(wall_seconds=perf_counter()-tick,results=results)
    run(False);run(True);rows=[]
    for repeat in range(args.repeats):
        order=(False,True) if repeat%2==0 else (True,False);variants={}
        for enabled in order:variants['candidate' if enabled else 'rollback']=run(enabled)
        if [r['status'] for r in variants['candidate']['results']]!=[r['status'] for r in variants['rollback']['results']]:
            raise RuntimeError('paired fixed-support decisions differ')
        rows.append(dict(repeat=repeat,variants=variants))
    times={name:median(r['variants'][name]['wall_seconds'] for r in rows) for name in ('rollback','candidate')}
    results=rows[-1]['variants']['candidate']['results']
    report=dict(complete=True,cases=len(cases),rows=rows,median_seconds=times,
        cut_exclusions=sum('cut_certificate' in r for r in results),unknown=sum(r['status']=='UNKNOWN' for r in results),
        verified_full_models=sum(bool(r['words']) for r in results),fixed_fiber_time_reduction=1-times['candidate']/times['rollback'],
        input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
        claim='Restricted literal G-support fibers only. This is not complete C_A-region search or new-class discovery-rate evidence.')
    atomic_write_json(args.output/'report.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='rows'}),flush=True)


if __name__=='__main__':main()
