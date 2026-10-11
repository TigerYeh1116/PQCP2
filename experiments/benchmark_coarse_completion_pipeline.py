"""Paired complete discovery-pipeline replay from audited non-solution inputs.

Each independent process generates the entire declared anchor ball, decides
every strict target, verifies full binary outputs and writes novel classes.
Zero/zero discoveries do not establish a new-discovery speed improvement.
"""
import argparse
import gzip
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
from statistics import median
import subprocess
import sys
from time import perf_counter

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.benchmark_l86_discovery import freeze_sources,verified_inventory
from solver.checkpoint import atomic_write_json
from solver.verifier import verify_pqcp
from experiments.complete_l86_full_rows import audited_coarse_centers,audited_coarse_points
from experiments.prepare_coarse_resume import closed_target_stats


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=ROOT/'results/L86_latest_two_halves_radius2_top8_20261008/input_centers.json')
    p.add_argument('--coarse-input',type=Path)
    p.add_argument('--coarse-anchors',type=Path)
    p.add_argument('--top',type=int,default=0,help='Explicit literal-point prefix; zero selects all')
    isolate=p.add_mutually_exclusive_group()
    isolate.add_argument('--isolate-all-target-psd',action='store_true',help='Same literal points and backend; only candidate shares exact all-target C_A PSD envelope.')
    isolate.add_argument('--isolate-prebatch-dihedral',action='store_true',help='Same literal points and backend; only candidate reuses complete C_B support orbit exclusions before batch work.')
    isolate.add_argument('--candidate-orbit-and-affine-audit',action='store_true',help='Same region/backend; candidate uses prebatch dihedral exclusions and exact affine-space equality audit.')
    isolate.add_argument('--candidate-orbit-and-linear-certificate',action='store_true',help='Same region/backend; candidate uses prebatch dihedral exclusions and an independently checked affine constraint certificate.')
    isolate.add_argument('--candidate-orbit-certificate-and-native-audit',action='store_true',help='Same region/backend; candidate uses prebatch dihedral exclusions, checked affine certificates and full native GF2 norm audits.')
    isolate.add_argument('--isolate-cut-projection',action='store_true',help='Both sides use norm lifts/cache/dihedral; only candidate uses exact graph-cut projection.')
    isolate.add_argument('--isolate-norm-carrier-cache',action='store_true',help='Both sides use norm lifts/UNSAT cache/dihedral; only candidate reuses C_B carriers/deferred support domains.')
    isolate.add_argument('--isolate-unary-batch',action='store_true',help='Both sides use norm lifts/UNSAT cache/dihedral; only candidate completes low-dimensional C_B fibers in independently audited batches.')
    isolate.add_argument('--candidate-unary-batch-and-carriers',action='store_true',help='Compare current mathematical baseline against batch completion plus carrier-domain reuse.')
    p.add_argument('--radius',type=int,default=1);p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--output',type=Path,default=ROOT/'results'/('L86_coarse_pipeline_paired_'+datetime.now().strftime('%Y%m%d_%H%M%S')))
    args=p.parse_args();args.output=args.output.resolve()
    if args.radius not in (0,1,2) or args.repeats<1:p.error('radius 0..2 and positive repeat count required')
    if args.top<0 or (args.coarse_input and args.coarse_anchors):p.error('nonnegative top and one coarse source required')
    inputs=(audited_coarse_points(args.coarse_anchors,args.top) if args.coarse_anchors else
            audited_coarse_centers(args.coarse_input,0) if args.coarse_input else json.loads(args.input.read_text()))
    if (not inputs['seeds'] or (not args.coarse_input and not args.coarse_anchors and any(verify_pqcp(*s['words']).is_valid for s in inputs['seeds']))
            or (args.coarse_input and any(s['source_full_verified'] for s in inputs['seeds']))):
        raise ValueError('never seed discovery performance with an already verified PQCP')
    args.output.mkdir(parents=True,exist_ok=False);source=args.output/'source_snapshot';freeze_sources(source)
    (source/'experiments').mkdir(exist_ok=True)
    for path in (ROOT/'experiments').glob('*.py'):shutil.copy2(path,source/'experiments'/path.name)
    raw=(ROOT/'86.txt').read_bytes() if (ROOT/'86.txt').exists() else b'';known=verified_inventory(raw)
    (args.output/'initial_inventory.txt').write_bytes(raw);(source/'86.txt').write_bytes(raw)
    manifest={str(f.relative_to(source)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(source.rglob('*'))
              if f.is_file() and f.name!='source_manifest.json'}
    atomic_write_json(source/'source_manifest.json',manifest);atomic_write_json(args.output/'input_centers.json',inputs)
    if args.coarse_input:shutil.copy2(args.coarse_input,args.output/'input_coarse_source.json')
    if args.coarse_anchors:
        packet=json.loads(args.coarse_anchors.read_text());packet['points']=packet['points'][:args.top or None]
        atomic_write_json(args.output/'input_coarse_points.json',packet)
    runtime=args.output/'runtime_source';shutil.copytree(source,runtime)
    tick=perf_counter();rows=[];times={'rollback':[],'candidate':[]}
    def run(name,variant,radius):
        folder=args.output/name;folder.mkdir();output=folder/'result';env=dict(os.environ)
        env['PYTHONPYCACHEPREFIX']='/private/tmp/pqcp2_coarse_pipeline_pycache'
        input_option=(('--coarse-anchors',str(args.output/'input_coarse_points.json')) if args.coarse_anchors else
                      ('--coarse-input',str(args.output/'input_coarse_source.json')) if args.coarse_input else ('--input',str(args.output/'input_centers.json')))
        command=[str(ROOT/'.venv/bin/python'),str(runtime/'experiments/complete_l86_coarse_rows.py'),
                 *input_option,'--top','8' if args.coarse_anchors and name.startswith('warm_') else '0','--radius',str(radius),
                 '--support-bounds','--certified-psd','--prime-norm','--stop-on-new','--output',str(output)]
        command+=(['--two-adic','--dihedral-support','--norm-unsat-cache','4096'] if variant=='candidate' or args.isolate_cut_projection or args.isolate_norm_carrier_cache or args.isolate_unary_batch or args.candidate_unary_batch_and_carriers else
                  ['--no-two-adic','--no-dihedral-support','--norm-unsat-cache','0'])
        command+=['--cut-projection' if args.isolate_cut_projection and variant=='candidate' else '--no-cut-projection']
        command+=['--norm-carrier-cache','4096' if (args.isolate_norm_carrier_cache or args.candidate_unary_batch_and_carriers) and variant=='candidate' else '0']
        command+=['--unary-batch-free','12' if (args.isolate_unary_batch or args.candidate_unary_batch_and_carriers) and variant=='candidate' else '-1']
        if args.isolate_all_target_psd or args.isolate_prebatch_dihedral or args.candidate_orbit_and_affine_audit or args.candidate_orbit_and_linear_certificate or args.candidate_orbit_certificate_and_native_audit:
            command+=['--two-adic','--dihedral-support','--norm-unsat-cache','4096',
                      '--norm-carrier-cache','4096','--unary-batch-free','12']
            command+=['--all-target-psd' if args.isolate_all_target_psd and variant=='candidate' else '--no-all-target-psd',
                      '--prebatch-dihedral' if (args.isolate_prebatch_dihedral or args.candidate_orbit_and_affine_audit or args.candidate_orbit_and_linear_certificate or args.candidate_orbit_certificate_and_native_audit) and variant=='candidate' else '--no-prebatch-dihedral',
                      '--unary-affine-audit' if args.candidate_orbit_and_affine_audit and variant=='candidate' else '--no-unary-affine-audit',
                      '--unary-linear-certificate' if (args.candidate_orbit_and_linear_certificate or args.candidate_orbit_certificate_and_native_audit) and variant=='candidate' else '--no-unary-linear-certificate',
                      '--native-support-audit' if args.candidate_orbit_certificate_and_native_audit and variant=='candidate' else '--no-native-support-audit']
        atomic_write_json(folder/'command.json',command);start=perf_counter()
        with (folder/'console.log').open('w') as log:
            child=subprocess.Popen(command,cwd=str(runtime),env=env,stdout=log,stderr=subprocess.STDOUT)
            atomic_write_json(folder/'process.json',dict(pid=child.pid,command=command))
            code=child.wait()
        elapsed=perf_counter()-start
        if code:raise RuntimeError(f'{name} exited with status {code}; inspect {folder}/console.log')
        report=json.loads((output/'report.json').read_text());solution=output/'86.txt'
        keys=verified_inventory(solution.read_bytes() if solution.exists() else b'')-known
        if len(keys)!=report['independently_verified_new_classes']:raise RuntimeError('full novel inventory disagrees with report')
        if report['unknown_target_fibers']:raise RuntimeError('paired replay has unresolved target fibers')
        if not report['complete'] and not (report['stopped_on_new'] and keys):raise RuntimeError('run is neither a full exhaustion nor a verified new discovery')
        if report['complete']:
            with gzip.open(output/'completed_anchors.jsonl.gz','rt') as stream:anchors=[json.loads(line) for line in stream]
            if len(anchors)!=report['distinct_anchors'] or any(
                    r['anchor']!=j or r['target_fibers_decided']!=84 or r['statuses']!={'UNSAT_FIXED_SIGNED_C_A':84}
                    for j,r in enumerate(anchors)):
                raise RuntimeError('completed anchor evidence contains partial/SAT/UNKNOWN branches')
            with gzip.open(output/'full_row_target_fibers.jsonl.gz','rt') as stream:
                audited=closed_target_stats(anchors,(json.loads(line) for line in stream))
            if any(audited[k]!=report[k] for k in audited):raise RuntimeError('all-target ledger differs from whole-pipeline report')
        signature=hashlib.sha256()
        with gzip.open(output/'full_row_target_fibers.jsonl.gz','rt') as stream:
            for line in stream:
                row=json.loads(line)
                signature.update(json.dumps([row[k] for k in ('anchor','k','eta','status','complete','words')],separators=(',',':')).encode())
                signature.update(b'\n')
        return dict(wall_seconds=elapsed,new_classes=len(keys),report=report,target_decisions_sha256=signature.hexdigest())
    # Warm compilation/OS caches on non-solution parent anchors. The fresh
    # process still performs its own math setup; all of that time is counted.
    for variant in times:run('warm_'+variant,variant,0)
    for repeat in range(args.repeats):
        order=('rollback','candidate') if repeat%2==0 else ('candidate','rollback');variants={}
        for variant in order:
            result=run('repeat_'+str(repeat)+'_'+variant,variant,args.radius)
            variants[variant]=result;times[variant].append(result['wall_seconds'])
        a,b=(variants[v]['report'] for v in ('rollback','candidate'))
        if (args.isolate_all_target_psd or args.isolate_prebatch_dihedral or args.candidate_orbit_and_affine_audit or args.candidate_orbit_and_linear_certificate or args.candidate_orbit_certificate_and_native_audit) and variants['rollback']['target_decisions_sha256']!=variants['candidate']['target_decisions_sha256']:
            raise RuntimeError('same literal region has different complete target decisions')
        if not variants['rollback']['new_classes'] and not variants['candidate']['new_classes']:
            if (a['distinct_anchors'],a['target_fibers_decided'],a['all_planned_anchors_exhausted'])!=(
                    b['distinct_anchors'],b['target_fibers_decided'],b['all_planned_anchors_exhausted']):
                raise RuntimeError('variants did not decide the same entire declared region')
        rows.append(dict(repeat=repeat,order=order,variants=variants))
        med={v:median(t) for v,t in times.items()};all_hits=all(r['variants'][v]['new_classes']>0 for r in rows for v in times)
        out=dict(complete=len(rows)==args.repeats,radius=args.radius,rows=rows,median_pipeline_seconds=med,
                 direct_coarse_input=bool(args.coarse_input),isolate_cut_projection=args.isolate_cut_projection,
                 literal_coarse_points_input=bool(args.coarse_anchors),selected_top=args.top,
                 isolate_all_target_psd=args.isolate_all_target_psd,isolate_prebatch_dihedral=args.isolate_prebatch_dihedral,
                 candidate_orbit_and_affine_audit=args.candidate_orbit_and_affine_audit,
                 candidate_orbit_and_linear_certificate=args.candidate_orbit_and_linear_certificate,
                 candidate_orbit_certificate_and_native_audit=args.candidate_orbit_certificate_and_native_audit,
                 isolate_norm_carrier_cache=args.isolate_norm_carrier_cache,
                 isolate_unary_batch=args.isolate_unary_batch,candidate_unary_batch_and_carriers=args.candidate_unary_batch_and_carriers,
                 fixed_region_time_reduction=1-med['candidate']/med['rollback'],
                 verified_first_new_time_comparison=all_hits,
                 new_discovery_time_reduction=(1-med['candidate']/med['rollback']) if all_hits else None,
                 source_manifest_sha256=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest(),elapsed=perf_counter()-tick,
                 scope='Independent full processes from frozen non-solution source input. Anchor generation, all target/norm/sign math, '
                       'inline exclusion proof checks, full original verifier, deduplication, proof ledgers, setup and new solution saving are timed. '
                       'Alternating paired order; source snapshots/runtime separated. No observation timeout used as termination.',
                 claim='No >=20% new-discovery claim from complete-region timing when actual full novel classes are zero.')
        atomic_write_json(args.output/'report.json',out)
        print(json.dumps(dict(repeat=repeat,times={v:variants[v]['wall_seconds'] for v in times},
                             new_classes={v:variants[v]['new_classes'] for v in times})),flush=True)


if __name__=='__main__':main()
