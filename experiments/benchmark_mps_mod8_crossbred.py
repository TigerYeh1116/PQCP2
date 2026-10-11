"""Frozen-source complete candidate and timing audit of Crossbred GPU pruning."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
from random import Random
import shutil
import statistics
import subprocess
import sys
from time import perf_counter

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.benchmark_l86_discovery import freeze_sources
from solver.checkpoint import atomic_write_json
from solver.compressed_repair import _validated_reference
from solver.mod8_crossbred_native import lookup_tail_consequences,specialize_lookup
from solver.mod8_model_reference import all_mod8_indices
from solver.mps_mod8_prefix import quadratic_coefficients,enumerate_mod8_candidates,complete_mod8_models
from solver.odd_lift_native import difference_sign_problem
from solver.target_profiles import TargetContentProfile
from solver.ternary_system_native import prepare_native_system,run_native_system


def literal_frontiers(count):
    path=ROOT/'results/L86_projection_groups_twenty_million_20261008/literal_trials.jsonl.gz'
    results=[];digest=hashlib.sha256();scanned=0
    with gzip.open(path,'rb') as stream:
        for line in stream:
            scanned+=1;digest.update(line);row=json.loads(line)
            if row['sampling']['mask'] is None:continue
            targets=tuple(TargetContentProfile(**t) for t in row['target_group'])
            references=tuple(tuple([-1]*n+[0]*z+[1]*p) for n,z,p in row['contents'])
            a,b,d,wanted=_validated_reference(*references,targets[0]);masks=row['u'],row['sampling']['mask']
            c=prepare_native_system(masks,d,wanted,(a.count(-1),b.count(-1)))
            g_data=[difference_sign_problem(masks,t) for t in targets]
            gs=[prepare_native_system(gm,d,gw,gn) for gm,gw,gn in g_data]
            c_first=c['consistent'] and c['estimated_assignments']<=sum(g['estimated_assignments'] for g in gs)
            plan=c if c_first else min(gs,key=lambda p:p['estimated_assignments'])
            if not plan['consistent'] or plan['selected']>=0 or not 16<=plan['meta'][2]<=23:continue
            results.append(dict(masks=plan['masks'],wanted=tuple(plan['wanted']),negative=tuple(plan['negative']),
                                q=int(plan['meta'][2]),kind='C_FRONTIER' if c_first else 'G_FRONTIER',trial=row['trial']))
            if len(results)==count:break
    if len(results)!=count:raise RuntimeError('insufficient complete raw production frontiers')
    return results,dict(literal_prefix_lines=scanned,literal_prefix_sha256=digest.hexdigest(),source=str(path))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--_frozen',action='store_true',help=argparse.SUPPRESS);args=p.parse_args();args.output=args.output.resolve()
    if not args._frozen:
        args.output.mkdir(parents=True,exist_ok=False)
        fixtures=json.loads((ROOT/'results/L86_mps_mod8_prefix_controls_audit_20261008/report.json').read_text())['actual_cases']
        frontiers,provenance=literal_frontiers(64)
        atomic_write_json(args.output/'input_cases.json',dict(fixtures=fixtures,frontiers=frontiers,provenance=provenance))
        snapshot=args.output/'source_snapshot';freeze_sources(snapshot)
        for name in ('benchmark_mps_mod8_crossbred.py','benchmark_l86_discovery.py'):
            target=snapshot/'experiments'/name;target.parent.mkdir(exist_ok=True);shutil.copy2(ROOT/'experiments'/name,target)
        manifest={str(f.relative_to(snapshot)):hashlib.sha256(f.read_bytes()).hexdigest()
                  for f in sorted(snapshot.rglob('*')) if f.is_file() and f.name!='source_manifest.json'}
        atomic_write_json(snapshot/'source_manifest.json',manifest)
        code=subprocess.run([sys.executable,str(snapshot/'experiments/benchmark_mps_mod8_crossbred.py'),
                             '--_frozen','--output',str(args.output)],cwd=snapshot).returncode
        if code:raise SystemExit(code)
        return
    if ROOT.resolve()!=(args.output/'source_snapshot').resolve():raise RuntimeError('actual frozen execution required')
    manifest=json.loads((ROOT/'source_manifest.json').read_text())
    if any(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest for name,digest in manifest.items()):
        raise RuntimeError('frozen source manifest mismatch')
    inputs=json.loads((args.output/'input_cases.json').read_text());cases=inputs['fixtures']+inputs['frontiers']
    plans=[prepare_native_system(tuple(r['masks']),43,tuple(r['wanted']),tuple(r['negative']),conditioning=False) for r in cases]
    references=[all_mod8_indices(quadratic_coefficients(plan),capacity=4096) for plan in plans]
    if any(not r['complete'] for r in references):raise RuntimeError('reference capacity insufficient; no exclusions justified')
    configs=[dict(name='old_gpu_t6_serial',crossbred=False,tail=6,preparation_workers=1),
             dict(name='crossbred_t6_serial',crossbred=True,tail=6,prefix_multipliers=8,preparation_workers=1),
             dict(name='crossbred_t8_parallel',crossbred=True,tail=8,prefix_multipliers=13,preparation_workers=8),
             dict(name='bit_sliced_t6_serial',bit_sliced=True,tail=6,preparation_workers=1),
             dict(name='bit_sliced_t7_serial',bit_sliced=True,tail=7,preparation_workers=1)]
    expected=[run_native_system(plan,2**23) for plan in plans]
    for config in configs:
        kwargs={k:v for k,v in config.items() if k!='name'}
        models=enumerate_mod8_candidates(plans,capacity=4096,**kwargs)
        for plan,out,reference,decision in zip(plans,models,references,expected):
            if not out['complete'] or out['indices']!=reference['indices']:
                raise RuntimeError('entire GPU modular candidate set differs from independent complete reference')
            if complete_mod8_models(plan,out)['status']!=decision['status']:
                raise RuntimeError('exact integer sign decision differs')
        print(json.dumps(dict(stage='whole_zero_sets_pass',config=config['name'],systems=len(plans))),flush=True)
    # Independently replay selected proof rows and table values on actual
    # q=21 problems. Diagnostic overhead is kept out of the timed path.
    rng=Random(862026);proofs=0;lookups=0
    for plan in plans:
        coef=quadratic_coefficients(plan);tail=min(8,len(coef[1]))
        lookup=lookup_tail_consequences(coef,tail,prefix_multipliers=13,audit_certificates=True)
        proofs+=lookup['selected_count']
        for bits in rng.sample(range(1<<lookup['prefix']),min(32,1<<lookup['prefix'])):
            constant,columns=specialize_lookup(lookup,bits)
            for suffix in (0,(1<<tail)-1):
                actual=constant
                for j,column in enumerate(columns):
                    if suffix>>j&1:actual^=column
                index=bits|(suffix<<lookup['prefix']);direct=0
                for equation,terms in enumerate(lookup['terms']):
                    direct|=(sum(index&term==term for term in terms)%2)<<equation
                if actual!=direct:raise RuntimeError('lookup table failed independent symbolic evaluation')
                lookups+=1
    positive=next(i for i,r in enumerate(references) if r['candidate_count']>1)
    zero=prepare_native_system((0,0),3,(0,0,0),(0,0),conditioning=False)
    control=enumerate_mod8_candidates([zero,plans[positive]],crossbred=True,tail=8,prefix_multipliers=13,capacity=1)
    if (control[0]['indices']!=(0,) or control[1]['complete'] or control[1]['indices'] is not None
            or complete_mod8_models(plans[positive],control[1])['status']!='UNKNOWN'):
        raise RuntimeError('mixed dimension or overflow UNKNOWN control failed')
    bit_control=enumerate_mod8_candidates([zero,plans[positive]],bit_sliced=True,tail=6,capacity=1)
    if (bit_control[0]['indices']!=(0,) or bit_control[1]['complete'] or bit_control[1]['indices'] is not None
            or bit_control[1]['candidate_count']!=references[positive]['candidate_count']
            or complete_mod8_models(plans[positive],bit_control[1])['status']!='UNKNOWN'):
        raise RuntimeError('bit-sliced mixed dimension or overflow UNKNOWN control failed')
    timings={c['name']:[] for c in configs};timings['native_cpu']=[]
    for repeat in range(6):
        order=list(range(len(configs)+1));order=order[repeat%len(order):]+order[:repeat%len(order)]
        for i in order:
            tick=perf_counter()
            if i==len(configs):
                out=[run_native_system(plan,2**23) for plan in plans];name='native_cpu'
            else:
                config=configs[i];kwargs={k:v for k,v in config.items() if k!='name'}
                models=enumerate_mod8_candidates(plans,capacity=4096,**kwargs)
                out=[complete_mod8_models(plan,model) for plan,model in zip(plans,models)];name=config['name']
            seconds=perf_counter()-tick
            if [o['status'] for o in out]!=[o['status'] for o in expected]:raise RuntimeError('timed whole sign decisions changed')
            timings[name].append(seconds)
    medians={k:statistics.median(v) for k,v in timings.items()}
    result=dict(complete=True,actual_frozen_execution=True,source_manifest_sha256=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest(),
        fixture_domains=32,fresh_literal_frontier_domains=64,whole_mod8_zero_sets_audited=len(plans),
        all_original_mod8_zero_counts=[r['candidate_count'] for r in references],
        independently_replayed_selected_proofs=proofs,independent_lookup_evaluations=lookups,
        overflow_unknown_and_mixed_dimensions_pass=True,bit_sliced_overflow_unknown_and_mixed_dimensions_pass=True,
        timings=timings,median_seconds=medians,
        source_input_sha256=hashlib.sha256((args.output/'input_cases.json').read_bytes()).hexdigest(),
        claim='Complete sign-subproblem timings only; no full novel L86 discovery-speed claim.')
    atomic_write_json(args.output/'report.json',result);print(json.dumps(dict(complete=True,median_seconds=medians)),flush=True)


if __name__=='__main__':main()
