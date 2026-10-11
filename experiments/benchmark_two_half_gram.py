"""Paired fixed-work mathematical joint-Gram filter replay, with proof audit.

Identical fixed-triple fibers, identical complete full-word outputs. Pure
integer PSD is shared by both variants; only joint Gram is toggled. All
candidate certificates are independently reconstructed after timed work.
Zero/zero discoveries never establish a new-discovery rate improvement.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
from statistics import median
import sys
from time import perf_counter

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.benchmark_l86_discovery import freeze_sources,verified_inventory,discovery_key
from solver.checkpoint import atomic_write_json
from solver.half_row_completion import complete_three_parity_rows
from solver.joint_gram_pruning import verify_joint_gram_certificate
from solver.finite_field_trace_pruning import verify_finite_trace_certificate
from solver.search_runner import append_verified_solution_if_new
from solver.two_half_row_completion import two_half_row_fibers


def signature(out):
    return (out['complete'],out['raw_half_assignments'],tuple(sorted(hit['words'] for hit in out['witnesses'])),
            tuple((r['wanted'],r['targets']) for r in out['profile_records']))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=ROOT/'results/L86_latest_two_halves_radius2_top8_20261008/input_centers.json')
    parser.add_argument('--radius',type=int,default=2);parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--comparison',choices=('gram','finite','combined','batch','affine'),default='gram')
    parser.add_argument('--output',type=Path,default=ROOT/'results'/('L86_two_half_gram_paired_'+datetime.now().strftime('%Y%m%d_%H%M%S')))
    args=parser.parse_args()
    if args.radius not in range(4) or args.repeats<1:parser.error('radius 0..3 and positive repeats required')
    inputs=json.loads(args.input.read_text());jobs=[]
    for item in inputs['seeds']:
        jobs.extend(j for j in two_half_row_fibers(*item['words'],radius=args.radius) if j['kind']=='ELIGIBLE')
    if not jobs:raise ValueError('nonempty audited eligible fixed-triple pool required')
    args.output.mkdir(parents=True,exist_ok=False);snapshot=args.output/'source_snapshot';freeze_sources(snapshot)
    (snapshot/'experiments').mkdir(exist_ok=True)
    for p in (ROOT/'experiments').glob('*.py'):shutil.copy2(p,snapshot/'experiments'/p.name)
    manifest={str(p.relative_to(snapshot)):hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(snapshot.rglob('*')) if p.is_file() and p.name!='source_manifest.json'}
    atomic_write_json(snapshot/'source_manifest.json',manifest);shutil.copy2(args.input,args.output/'input_centers.json')
    raw=(ROOT/'86.txt').read_bytes() if (ROOT/'86.txt').exists() else b''
    (args.output/'initial_inventory.txt').write_bytes(raw);known=verified_inventory(raw)
    print(json.dumps(dict(centers=len(inputs['seeds']),fibers_per_replay=len(jobs),repeats=args.repeats)),flush=True)
    def options(candidate):
        if args.comparison=='affine':return dict(joint_gram=True,finite_trace=True,batched_psd=True,affine_finite=candidate)
        if args.comparison=='batch':return dict(joint_gram=True,finite_trace=True,batched_psd=candidate)
        return dict(joint_gram=candidate and args.comparison in ('gram','combined'),
                    finite_trace=candidate and args.comparison in ('finite','combined'))
    for job in jobs:
        for candidate in (False,True):complete_three_parity_rows(job['halves'],free_component=job['free'],**options(candidate))
    tick=perf_counter();rows=[];times={'rollback':[],'candidate':[]};keys={'rollback':set(),'candidate':set()};audited=0
    for repeat in range(args.repeats):
        order=('rollback','candidate') if repeat%2==0 else ('candidate','rollback');records={};signatures={}
        for variant in order:
            started=perf_counter()
            outputs=[complete_three_parity_rows(j['halves'],free_component=j['free'],**options(variant=='candidate')) for j in jobs]
            elapsed=perf_counter()-started;times[variant].append(elapsed)
            signatures[variant]=tuple(map(signature,outputs));norm_visits=sum(o['norm_visited'] for o in outputs)
            # Sparse batching promises the EXACT same first-frequency PSD
            # certificates, in addition to identical complete full outputs.
            if args.comparison=='batch':
                signatures[variant]+=tuple(tuple((r['wanted'],r['result'].get('psd_certificate')) for r in o['profile_records']) for o in outputs)
            tested=sum(o['joint_gram_targets_tested'] for o in outputs);excluded=sum(o['joint_gram_targets_excluded'] for o in outputs)
            field_tested=sum(o['finite_trace_targets_tested'] for o in outputs)
            field_excluded=sum(o['finite_trace_targets_excluded'] for o in outputs);proof_count=field_proof_count=0
            for out in outputs:
                for r in out['profile_records']:
                    for certificate in r['result'].get('joint_gram_result',{}).get('certificates',()):
                        if not verify_joint_gram_certificate(certificate):raise RuntimeError('native Gram certificate failed independent Python reconstruction')
                        proof_count+=1
                    for certificate in r['result'].get('finite_trace_result',{}).get('certificates',()):
                        if not verify_finite_trace_certificate(certificate):raise RuntimeError('finite-field trace certificate failed independent Python reconstruction')
                        field_proof_count+=1
                for hit in out['witnesses']:
                    key=discovery_key(*hit['words'])
                    if key not in known:
                        keys[variant].add(key);destination=args.output/variant;destination.mkdir(exist_ok=True)
                        append_verified_solution_if_new(86,*hit['words'],destination)
            if variant=='candidate' and proof_count!=excluded:raise RuntimeError('target exclusions do not all have audited certificates')
            if variant=='candidate' and field_proof_count!=field_excluded:raise RuntimeError('finite-field exclusions do not all have audited certificates')
            audited+=proof_count+field_proof_count
            records[variant]=dict(wall_seconds=elapsed,norm_visited=norm_visits,gram_targets_tested=tested,
                                  gram_targets_excluded=excluded,finite_targets_tested=field_tested,finite_targets_excluded=field_excluded,
                                  independently_audited_certificates=proof_count+field_proof_count)
            del outputs
        if signatures['rollback']!=signatures['candidate']:raise RuntimeError('joint Gram filter changed a complete full-word fiber or necessary target set')
        del signatures
        rows.append(dict(repeat=repeat,order=order,variants=records))
        report=dict(complete=len(rows)==args.repeats,comparison=args.comparison,centers=len(inputs['seeds']),radius=args.radius,
                    fibers_per_replay=len(jobs),rows=rows,median_replay_seconds={k:median(v) for k,v in times.items()},
                    local_time_reduction=1-median(times['candidate'])/median(times['rollback']),
                    independently_audited_certificates=audited,all_complete_full_word_sets_equal=True,
                    independently_verified_new_classes={k:len(v) for k,v in keys.items()},elapsed=perf_counter()-tick,
                    source_manifest_sha256=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest(),
                    scope='Warm identical fixed-triple completion replay; native individual PSD shared by both variants, '
                          'named mathematical necessary-condition switch only. All full words and original targets retained. '
                          'Proof audits and serialization outside timed completion. Not complete-pipeline discovery rate.',
                    claim='No >=20% new-class-discovery claim from local timings, pruned norm visits or zero/zero classes.')
        atomic_write_json(args.output/'report.json',report)
        print(json.dumps(dict(repeat=repeat,variants=records,new_classes=report['independently_verified_new_classes'])),flush=True)


if __name__=='__main__':main()
