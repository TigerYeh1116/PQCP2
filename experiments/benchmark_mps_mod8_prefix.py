"""Hardware audit of ALL GPU modular candidates; production remains off."""
import argparse
from dataclasses import asdict
from itertools import product
import hashlib
import json
from pathlib import Path
from random import Random
import sys
from time import perf_counter

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from solver.checkpoint import atomic_write_json
from solver.mod8_model_reference import all_mod8_indices
from solver.mps_mod8_prefix import quadratic_coefficients,enumerate_mod8_candidates,complete_mod8_models
from solver.odd_lift_native import difference_sign_problem
from solver.structured_energy import integer_periodic_autocorrelation_profile as pacf
from solver.target_profiles import TargetContentProfile
from solver.ternary_support_search import support_mask
from solver.ternary_system_native import prepare_native_system,run_native_system


def direct_indices(plan):
    q=int(plan['meta'][2]);d=plan['d'];results=[]
    for index in range(1<<q):
        signs=[int(v) for v in plan['base']]
        for j in range(q):
            if index>>j&1:
                for s in range(2):signs[s]^=int(plan['toggles'][2*j+s])
        words=tuple(tuple((-1 if signs[s]>>i&1 else 1) if plan['masks'][s]>>i&1 else 0 for i in range(d)) for s in range(2))
        actual=tuple(x+y for x,y in zip(pacf(words[0]),pacf(words[1])))
        if all((x-y)%8==0 for x,y in zip(actual,plan['wanted'])) and all((w.count(-1)-n)%4==0 for w,n in zip(words,plan['negative'])):
            results.append(index)
    return tuple(results)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);rng=Random(11486);small=[]
    for d in (3,5,7,9,11):
        for case in range(16):
            occupied=set(rng.sample(range(2*d),d));masks=tuple(sum(1<<i for i in range(d) if s*d+i in occupied) for s in range(2))
            words=tuple(tuple(rng.choice((-1,1)) if m>>i&1 else 0 for i in range(d)) for m in masks)
            wanted=tuple(x+y for x,y in zip(pacf(words[0]),pacf(words[1])))
            if case%2:
                limit=sum(pacf(tuple(int(m>>i&1) for i in range(d)))[1] for m in masks)
                alternatives=[v for v in (wanted[1]-4,wanted[1]+4) if abs(v)<=limit]
                if alternatives:wanted=(wanted[0],alternatives[0])+wanted[2:-1]+(alternatives[0],)
            plan=prepare_native_system(masks,d,wanted,tuple(w.count(-1) for w in words),conditioning=False)
            if plan['consistent']:small.append(plan)
    for tail in (1,3,6,8):
        gpu=enumerate_mod8_candidates(small,tail=tail,capacity=2048)
        for plan,out in zip(small,gpu):
            expected=direct_indices(plan);reference=all_mod8_indices(quadratic_coefficients(plan),capacity=2048)
            if not out['complete'] or out['indices']!=expected or reference['indices']!=expected:
                raise RuntimeError('small full modular-set mismatch')
        print(json.dumps(dict(stage='all_small_integer_sets_pass',tail=tail,cases=len(small))),flush=True)
    bank=json.loads((ROOT/'results/L86_independent_norm_phase_ten_million_20261008/coarse_witnesses.json').read_text())['rows'][:16]
    actual=[]
    for row in bank:
        target=TargetContentProfile(**row['target']);c=tuple(tuple(w) for w in row['result']['attempts']['coarse']['words']);masks=tuple(support_mask(w) for w in c)
        wanted=tuple(x+y for x,y in zip(pacf(c[0]),pacf(c[1])))
        mg,wg,ng=difference_sign_problem(masks,target)
        for supports,profile,negative in ((masks,wanted,tuple(w.count(-1) for w in c)),(mg,wg,ng)):
            plan=prepare_native_system(supports,43,profile,negative,conditioning=False)
            if plan['consistent']:actual.append(plan)
    gpu=enumerate_mod8_candidates(actual,capacity=4096);rows=[]
    for plan,out in zip(actual,gpu):
        reference=all_mod8_indices(quadratic_coefficients(plan),capacity=4096)
        if out!=dict(complete=reference['complete'],status='MOD8_COMPLETE',candidate_count=reference['candidate_count'],indices=reference['indices']):
            raise RuntimeError('L43 entire GPU/CPU modular candidate-set mismatch')
        completed=complete_mod8_models(plan,out);old=run_native_system(plan,2**23)
        if completed['status']!=old['status']:raise RuntimeError('L43 exact norm decision mismatch')
        rows.append(dict(masks=plan['masks'],wanted=tuple(plan['wanted']),negative=tuple(plan['negative']),
                         q=int(plan['meta'][2]),candidate_count=out['candidate_count'],indices=out['indices'],decision=completed['status']))
    overflow_index=next(i for i,out in enumerate(gpu) if out['candidate_count']>1)
    zero=prepare_native_system((0,0),3,(0,0,0),(0,0),conditioning=False)
    controls=enumerate_mod8_candidates([zero,actual[overflow_index]],capacity=1)
    if (not controls[0]['complete'] or controls[0]['indices']!=(0,)
            or controls[1]['complete'] or controls[1]['status']!='DEFERRED_OVERFLOW'
            or controls[1]['indices'] is not None
            or controls[1]['candidate_count']!=gpu[overflow_index]['candidate_count']
            or complete_mod8_models(actual[overflow_index],controls[1])['status']!='UNKNOWN'):
        raise RuntimeError('mixed-dimension or overflow fallback control failed')
    timings=[]
    for repeat in range(3):
        batch=actual*2
        tick=perf_counter();old=[run_native_system(plan,2**23) for plan in batch];cpu=perf_counter()-tick
        tick=perf_counter();models=enumerate_mod8_candidates(batch,capacity=4096);new=[complete_mod8_models(plan,out) for plan,out in zip(batch,models)];gpu_seconds=perf_counter()-tick
        if [x['status'] for x in old]!=[x['status'] for x in new]:raise RuntimeError('timed batch decisions differ')
        timings.append(dict(cpu_seconds=cpu,gpu_pipeline_seconds=gpu_seconds,systems=len(batch)))
    sources=['csrc/mod8_prefix.metal','solver/mps_mod8_prefix.py','csrc/mod8_model_reference.c','solver/mod8_model_reference.py','experiments/benchmark_mps_mod8_prefix.py']
    report=dict(complete=True,small_cases=len(small),small_tails=[1,3,6,8],actual_cases=rows,timings=timings,
                mixed_dimension_q0_and_q21_control=True,overflow_unknown_control=True,
                source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources},
                claim='Entire modular candidate sets and exact sign decisions only; no new-class discovery-speed claim.')
    atomic_write_json(args.output/'report.json',report);print(json.dumps(dict(complete=True,timings=timings)),flush=True)


if __name__=='__main__':main()
