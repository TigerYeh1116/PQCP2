"""Frozen full C/G pipeline comparison with identical literal random samples.

No timeout and no discovery-speed claim when both sides find zero new classes.
Each result is retained, including UNKNOWN; compare all exact outputs afterwards.
"""
import argparse
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.benchmark_l86_discovery import freeze_sources,verified_inventory
from experiments.search_norm_supports import semantic_trial
from solver.checkpoint import atomic_write_json
from solver.verifier import verify_pqcp
from solver.odd_lift_native import difference_sign_problem
from solver.structured_energy import fold_profile,target_profile,integer_periodic_autocorrelation_profile as pacf
from solver.target_profiles import TargetContentProfile
from solver.ternary_support_search import support_mask


def audit_sign_witnesses(row):
    if row['result'] is None:return
    attempts=row['result']['attempts']
    if 'independent_targets' in attempts:
        for item in attempts['independent_targets']:
            audit_sign_witnesses(dict(row,target=item['target'],result=item['result']))
        return
    if 'lifts' in attempts:
        audit_sign_witnesses(dict(row,result=dict(attempts=dict(coarse=attempts['coarse']))))
        for item in attempts['lifts']:
            if item['result'] is not None:
                audit_sign_witnesses(dict(row,target=item['target'],result=dict(attempts=dict(difference=item['result']))))
        return
    if not any(isinstance(a,dict) and a.get('words') is not None for a in attempts.values()):return
    target=TargetContentProfile(**row['target']);masks=(row['u'],row['sampling']['mask'])
    other,wanted_g,negative_g=difference_sign_problem(masks,target)
    wanted_c=tuple(v//4 for v in fold_profile(target_profile(86,target.k,target.eta),2))
    for name,supports,wanted,negative in (('coarse',masks,wanted_c,tuple(c[0] for c in row['contents'])),
                                        ('difference',other,wanted_g,negative_g)):
        words=(attempts.get(name) or {}).get('words')
        if words is None:continue
        words=tuple(tuple(w) for w in words)
        if (tuple(support_mask(w) for w in words)!=supports or tuple(w.count(-1) for w in words)!=negative
                or tuple(x+y for x,y in zip(pacf(words[0]),pacf(words[1])))!=wanted):
            raise RuntimeError('intermediate signed witness failed independent literal norm/count audit')


def audit_pair(left,right,comparison='phase-count'):
    digest=hashlib.sha256();count=0
    with gzip.open(left/'literal_trials.jsonl.gz','rt') as a,gzip.open(right/'literal_trials.jsonl.gz','rt') as b:
        for x,y in itertools.zip_longest(a,b):
            if x is None or y is None:raise RuntimeError('unequal trace lengths')
            rows=[json.loads(line) for line in (x,y)]
            values=[semantic_trial(row) for row in rows]
            if comparison in ('phase-count','hybrid-residual'):
                if values[0]!=values[1]:raise RuntimeError('literal input/output mismatch at trial '+str(count))
            else:
                keys=('trial','seed','target','contents','u','sampling')+(('target_group',) if comparison in ('projection-groups','count-affine','bit-tail','lazy-c-gate','count-tail-depth','weight-shared-polynomials') else ())
                inputs=[{k:v[k] for k in keys} for v in values]
                if inputs[0]!=inputs[1] or rows[0]['status']!=rows[1]['status']:
                    raise RuntimeError('literal inputs or complete decision mismatch at trial '+str(count))
                attempts=[r['result']['attempts'] if r['result'] is not None else {} for r in rows]
                if comparison=='even-sign-orbit' and {k:v['status'] for k,v in attempts[0].items()}!={k:v['status'] for k,v in attempts[1].items()}:
                    raise RuntimeError('independent sign-problem decisions disagree')
            if rows[0]['trial']!=count:raise RuntimeError('nonconsecutive literal trials')
            for row in rows:
                audit_sign_witnesses(row)
                result=row['result']
                if result is not None and result['verified']:
                    if not verify_pqcp(*result['words']).is_valid:raise RuntimeError('full witness failed independent verifier')
            digest.update((json.dumps(values[0],sort_keys=True,separators=(',',':'))+'\n').encode());count+=1
    reports=[json.loads((directory/'report.json').read_text()) for directory in (left,right)]
    for j,report in enumerate(reports):
        if not report['complete'] or report['tested']!=count or (comparison in ('phase-count','hybrid-residual') and report['semantic_trials_sha256']!=digest.hexdigest()):
            raise RuntimeError('trace/report coverage or digest mismatch')
    return dict(trials=count,rollback_semantic_trials_sha256=digest.hexdigest(),
                exact_literal_inputs_and_outputs_match=comparison in ('phase-count','hybrid-residual'),
                exact_literal_inputs_and_complete_decisions_match=True,intermediate_sign_witnesses_independently_audited=True,
                unknown_count=reports[0]['statuses'].get('UNKNOWN',0))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--trials',type=int,default=16384)
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--seeds',type=int,nargs='+',default=[123,456,789])
    p.add_argument('--max-assignments',type=int,default=2**23)
    p.add_argument('--comparison',choices=('phase-count','even-sign-orbit','projection-groups','hybrid-residual','count-affine','bit-tail','lazy-c-gate','count-tail-depth','weight-shared-polynomials'),default='phase-count')
    p.add_argument('--legacy-c-kernel',type=Path,help='Archived pre-pruning C source for a genuine original-kernel rollback.')
    p.add_argument('--legacy-group-solver',type=Path,help='Archived pre-lazy C/G group solver for an isolated Python rollback.')
    p.add_argument('--legacy-system-wrapper',type=Path,help='Archived native dispatch policy for a matched kernel-and-policy rollback.')
    args=p.parse_args()
    if args.trials<1 or not 1<=args.workers<=16 or len(set(args.seeds))!=len(args.seeds):p.error('valid trials/workers/distinct seeds required')
    if args.comparison in ('hybrid-residual','count-affine','bit-tail','count-tail-depth','weight-shared-polynomials') and args.legacy_c_kernel is None:p.error('archived pre-change C kernel required')
    if args.comparison=='weight-shared-polynomials' and args.legacy_system_wrapper is None:p.error('archived dispatch policy required')
    if args.legacy_system_wrapper is not None and args.comparison!='weight-shared-polynomials':p.error('dispatch rollback is specific to weight-shared-polynomials')
    if args.comparison=='lazy-c-gate' and (args.legacy_group_solver is None or args.legacy_c_kernel is not None):
        p.error('lazy-c-gate requires its archived group solver without a separate C rollback')
    if args.legacy_group_solver is not None and args.comparison!='lazy-c-gate':p.error('group rollback is specific to lazy-c-gate')
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    source=out/'source';freeze_sources(source);(source/'experiments').mkdir()
    for name in ('search_norm_supports.py','benchmark_l86_discovery.py'):
        shutil.copy2(ROOT/'experiments'/name,source/'experiments'/name)
    inventory=(ROOT/'86.txt').read_bytes() if (ROOT/'86.txt').exists() else b''
    verified_inventory(inventory);(out/'initial_inventory.txt').write_bytes(inventory)
    if inventory:(source/'86.txt').write_bytes(inventory)
    manifest={str(f.relative_to(source)):hashlib.sha256(f.read_bytes()).hexdigest()
              for f in sorted(source.rglob('*')) if f.is_file() and f.name!='source_manifest.json'}
    atomic_write_json(source/'source_manifest.json',manifest)
    runtime_sources={'candidate':source,'rollback':source}
    if args.legacy_group_solver is not None:
        legacy=args.legacy_group_solver.read_text()
        if 'lazy_c_gate' in legacy:p.error('rollback group source already contains the lazy gate')
        rollback=out/'rollback_source';shutil.copytree(source,rollback)
        (rollback/'solver/projection_group_completion.py').write_text(legacy)
        rollback_manifest={str(f.relative_to(rollback)):hashlib.sha256(f.read_bytes()).hexdigest()
            for f in sorted(rollback.rglob('*')) if f.is_file() and f.name!='source_manifest.json'}
        atomic_write_json(rollback/'source_manifest.json',rollback_manifest)
        shutil.copy2(args.legacy_group_solver,out/'original_rollback_group_solver.py')
        runtime_sources['rollback']=rollback
    if args.legacy_c_kernel is not None:
        if args.comparison not in ('even-sign-orbit','hybrid-residual','count-affine','bit-tail','count-tail-depth','weight-shared-polynomials'):p.error('legacy C rollback is specific to kernel comparisons')
        legacy=args.legacy_c_kernel.read_text()
        if args.comparison=='even-sign-orbit' and 'pqcp_quadratic_orbit_hybrid_signs' in legacy:p.error('rollback source must precede the orbit implementation')
        if args.comparison=='hybrid-residual' and 'PQCP_HYBRID_AUDIT_RESIDUAL' in legacy:p.error('rollback source already contains the residual implementation')
        rollback=out/'rollback_source';shutil.copytree(source,rollback)
        shim='' if 'pqcp_quadratic_orbit_hybrid_signs' in legacy else '\nint pqcp_quadratic_orbit_hybrid_signs(const uint64_t *s,const uint64_t *b,const uint64_t *t,int q,int d,const int *w,const int *n,int tail,uint64_t budget,uint64_t *out,uint64_t *stats){memset(stats,0,9*sizeof(uint64_t));return pqcp_quadratic_hybrid_signs(s,b,t,q,d,w,n,tail,budget,out,stats);}\n'
        (rollback/'csrc/ternary_quadratic_signs.c').write_text(legacy+shim)
        if args.legacy_system_wrapper is not None:
            shutil.copy2(args.legacy_system_wrapper,rollback/'solver/ternary_system_native.py')
            shutil.copy2(args.legacy_system_wrapper,out/'original_rollback_system_wrapper.py')
        rollback_manifest={str(f.relative_to(rollback)):hashlib.sha256(f.read_bytes()).hexdigest()
            for f in sorted(rollback.rglob('*')) if f.is_file() and f.name!='source_manifest.json'}
        atomic_write_json(rollback/'source_manifest.json',rollback_manifest)
        shutil.copy2(args.legacy_c_kernel,out/'original_rollback_kernel.c')
        runtime_sources['rollback']=rollback
    rows=[];pairs=[]
    for index,(seed,trials) in enumerate([(-1,min(512,args.trials))]+[(s,args.trials) for s in args.seeds]):
        label='warm' if index==0 else 'seed'+str(seed)
        directories={}
        for variant in (('rollback','candidate') if index%2==0 else ('candidate','rollback')):
            destination=out/(label+'_'+variant);directories[variant]=destination
            runtime=runtime_sources[variant]
            command=[sys.executable,str(runtime/'experiments/search_norm_supports.py'),'--output',str(destination),
                     '--trials',str(trials),'--workers',str(args.workers),'--seed',str(seed),
                     '--max-assignments',str(args.max_assignments),'--gzip-trace',
                     '--phase-count-pruning' if variant=='candidate' or args.comparison!='phase-count' else '--no-phase-count-pruning',
                     '--even-sign-orbit-pruning' if variant=='candidate' and args.comparison=='even-sign-orbit' else '--no-even-sign-orbit-pruning']
            if args.comparison=='projection-groups':
                command.extend(['--projection-groups','--shared-c-completion' if variant=='candidate' else '--no-shared-c-completion'])
            if args.comparison in ('hybrid-residual','count-affine'):command.extend(['--projection-groups','--shared-c-completion'])
            if args.comparison=='bit-tail':command.extend(['--projection-groups','--shared-c-completion'])
            if args.comparison=='lazy-c-gate':command.extend(['--projection-groups','--shared-c-completion'])
            if args.comparison=='count-tail-depth':command.extend(['--projection-groups','--shared-c-completion'])
            if args.comparison=='weight-shared-polynomials':command.extend(['--projection-groups','--shared-c-completion'])
            print(json.dumps(dict(event='start',label=label,variant=variant,trials=trials)),flush=True)
            with (out/(label+'_'+variant+'.console.log')).open('w') as log:
                code=subprocess.run(command,cwd=runtime,stdout=log,stderr=subprocess.STDOUT).returncode
            if code:raise RuntimeError('child failed; retained console: '+str(destination))
            row=dict(label=label,variant=variant,report=json.loads((destination/'report.json').read_text()))
            rows.append(row);print(json.dumps(dict(event='finish',label=label,variant=variant,
                elapsed=row['report']['elapsed'],statuses=row['report']['statuses'])),flush=True)
        pair=dict(label=label,audit=audit_pair(directories['rollback'],directories['candidate'],args.comparison))
        pairs.append(pair)
        report=dict(rows=rows,pairs=pairs,complete=index==len(args.seeds),comparison=args.comparison,
                    rollback_kernel='archived pre-lazy C/G Python group solver, same native C kernels' if args.legacy_group_solver else ({'hybrid-residual':'archived pre-residual C loop',
                                      'count-affine':'archived pre-affine-count C loop with residual optimization',
                                      'bit-tail':'archived pre-CPU-bit-tail C loop with affine-count and residual optimization',
                                      'count-tail-depth':'archived affine-count C kernel before q>=10 bit-tail dispatch with retained width; same lazy C gate',
                                      'weight-shared-polynomials':'archived native kernel AND dispatch policy before large-pattern polynomial sharing; same lazy C gate',
                                      'even-sign-orbit':'archived original C loop plus unused compatibility export'}[args.comparison]
                                     if args.legacy_c_kernel else 'same source, feature flag rollback'),
                    source_manifest_sha256=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest(),
                    claim='Identical literal-sample throughput only. Zero/zero is not measured new-class discovery speed.')
        if index:
            times={v:[r['report']['elapsed'] for r in rows if r['variant']==v and r['label']!='warm'] for v in ('rollback','candidate')}
            medians={v:statistics.median(t) for v,t in times.items()}
            report.update(median_seconds=medians,fixed_sample_elapsed_reduction=1-medians['candidate']/medians['rollback'])
        atomic_write_json(out/'report.json',report)


if __name__=='__main__':main()
