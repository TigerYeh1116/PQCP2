"""IDE-runnable exact two-half repair: arbitrary flips plus a fully free row.

All 12 ordered roles. Default radius two, eight audited latest-log centers.
Count filtering is a mathematical obstruction; center selection is bounded
and heuristic. A complete finite negative result is not global nonexistence.
"""
import argparse
from collections import OrderedDict
from datetime import datetime
import hashlib
import gzip
import json
from pathlib import Path
import shutil
import sys
from time import perf_counter

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.benchmark_l86_discovery import discovery_key,freeze_sources,verified_inventory
from experiments.repair_unrestricted_flips import audited_log_centers
from solver.checkpoint import atomic_write_json
from solver.correlation import normalize_binary_sequence
from solver.half_row_completion import complete_three_parity_rows
from solver.search_runner import append_verified_solution_if_new
from solver.solution_equivalence import pqcp_equivalence_key
from solver.two_half_row_completion import two_half_row_fibers
from solver.verifier import verify_pqcp

DEFAULT_RADIUS=2
DEFAULT_TOP=8  # 0 selects all input centers; selection is not a mathematical exclusion.
DEFAULT_JOINT_GRAM=True  # Complete-fiber paired replay and integer certificate audits passed.
DEFAULT_FINITE_TRACE=True  # Complete-fiber output equality and all finite-field proof audits passed.
DEFAULT_BATCHED_PSD=True  # Same individual certificates; complete paired replay improved local time.


def packed_profile(words):
    """Literal integer PACF through cyclic XOR/popcount, no FFT."""
    length=len(words[0]);all_bits=(1<<length)-1
    masks=tuple(sum(v<<i for i,v in enumerate(row)) for row in words);profile=[]
    for u in range(length):
        value=2*length
        for mask in masks:
            rotated=((mask<<u)|(mask>>(length-u)))&all_bits if u else mask
            value-=2*bin(mask^rotated).count('1')
        profile.append(value)
    return tuple(profile)


def select_json_centers(inputs,top):
    ordered=[];skipped=0
    for index,item in enumerate(inputs['seeds']):
        words=tuple(tuple(normalize_binary_sequence(row)) for row in item['words'])
        if tuple(map(len,words))!=(86,86):raise ValueError('equal L86 binary centers required')
        profile=packed_profile(words);nonzero=[u for u in range(1,86) if profile[u]]
        if len(nonzero)==2 and all(abs(profile[u])==4 for u in nonzero):skipped+=1;continue
        if any(v%4 for v in profile):raise RuntimeError('L86 binary pair profile lacks integral correlation units')
        k=max(range(1,43),key=lambda u:abs(profile[u]));eta=1 if profile[k]>=0 else -1
        energy=sum((profile[u]//4)**2 for u in range(1,44))-2*eta*(profile[k]//4)+1
        ordered.append(dict(words=words,k=k,eta=eta,independent_full_energy=energy,source_index=index,profile=profile))
    ordered.sort(key=lambda row:(row['independent_full_energy'],row['source_index']))
    selected=[];keys=set()
    for row in ordered:
        key=pqcp_equivalence_key(*row['words'])
        if key in keys:continue
        check=verify_pqcp(*row['words'])
        if check.is_valid or tuple(check.profile)!=row['profile']:raise RuntimeError('selected center failed independent full PACF audit')
        keys.add(key);selected.append(row)
        if top and len(selected)==top:break
    return dict(seeds=selected,literal_input_solutions_skipped=skipped,
                selection='Lowest exact best-target energy; bounded centers, no mathematical exclusion of other centers.')


def main(*,default_varied_halves=1,default_radius=DEFAULT_RADIUS):
    parser=argparse.ArgumentParser(description=__doc__);group=parser.add_mutually_exclusive_group()
    group.add_argument('--input',type=Path);group.add_argument('--log',type=Path)
    parser.add_argument('--top',type=int,default=DEFAULT_TOP)
    parser.add_argument('--radius',type=int,default=default_radius)
    parser.add_argument('--varied-halves',type=int,choices=(1,2,3),default=default_varied_halves,
                        help='one bounded half, or two nontrivially changed known halves sharing the total radius')
    parser.add_argument('--batched-psd',action=argparse.BooleanOptionalAction,default=DEFAULT_BATCHED_PSD)
    parser.add_argument('--affine-finite',action=argparse.BooleanOptionalAction,default=False)
    parser.add_argument('--trace-backend',action=argparse.BooleanOptionalAction,default=False)
    parser.add_argument('--certified-psd',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--joint-gram',action=argparse.BooleanOptionalAction,default=DEFAULT_JOINT_GRAM)
    parser.add_argument('--finite-trace',action=argparse.BooleanOptionalAction,default=DEFAULT_FINITE_TRACE)
    parser.add_argument('--fiber-cache',type=int,default=256)
    parser.add_argument('--gzip-ledger',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--output',type=Path,default=ROOT/'results'/('L86_two_half_completion_'+datetime.now().strftime('%Y%m%d_%H%M%S')))
    args=parser.parse_args()
    if args.top<0 or args.radius not in range(5) or not 0<=args.fiber_cache<=4096:
        parser.error('nonnegative top, radius 0..4, and fiber cache 0..4096 required')
    if args.varied_halves==2 and args.radius<2:parser.error('two varied halves require total radius 2..4')
    if args.varied_halves==3 and args.radius<3:parser.error('three varied halves require total radius 3..4')
    if args.input is None and args.log is None:
        logs=tuple((ROOT/'logs').glob('torch_L86_seed*.jsonl'))
        if not logs:raise ValueError('no root L86 log; input JSON required')
        args.log=max(logs,key=lambda p:p.stat().st_mtime)
    source=args.log if args.log is not None else args.input
    inputs=audited_log_centers(args.log,args.top if args.top else sys.maxsize) if args.log is not None else select_json_centers(json.loads(args.input.read_text()),args.top)
    if not inputs['seeds']:raise ValueError('no audited non-solution centers')
    args.output.mkdir(parents=True,exist_ok=False);tick=perf_counter();snapshot=args.output/'source_snapshot';freeze_sources(snapshot)
    (snapshot/'experiments').mkdir(exist_ok=True)
    for path in (ROOT/'experiments').glob('*.py'):shutil.copy2(path,snapshot/'experiments'/path.name)
    manifest={str(p.relative_to(snapshot)):hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(snapshot.rglob('*')) if p.is_file() and p.name!='source_manifest.json'}
    atomic_write_json(snapshot/'source_manifest.json',manifest)
    shutil.copy2(source,args.output/('input_log.jsonl' if args.log is not None else 'input_source.json'))
    atomic_write_json(args.output/'input_centers.json',inputs)
    raw=(ROOT/'86.txt').read_bytes() if (ROOT/'86.txt').exists() else b''
    (args.output/'initial_inventory.txt').write_bytes(raw);known=verified_inventory(raw);keys=set();first_hit=None
    totals=dict(generated_triples=0,content_excluded=0,identical_triples=0,eligible_jobs=0,cache_hits=0,
                actual_fibers_solved=0,norm_visited=0,profiles_decided=0,psd_certificates=0,
                support_orbits=0,norm_trace_candidates=0,full_models=0,joint_gram_targets_tested=0,joint_gram_targets_excluded=0,
                finite_trace_targets_tested=0,finite_trace_targets_excluded=0)
    rows=[];cache=OrderedDict()
    ordered_roles=4 if args.varied_halves==3 else 12
    print(json.dumps(dict(centers=len(inputs['seeds']),radius=args.radius,ordered_roles=ordered_roles,varied_halves=args.varied_halves)),flush=True)
    ledger_path=args.output/('completion_fibers.jsonl.gz' if args.gzip_ledger else 'completion_fibers.jsonl')
    ledger=gzip.open(ledger_path,'at',compresslevel=1) if args.gzip_ledger else ledger_path.open('a')
    try:
        for center_index,item in enumerate(inputs['seeds']):
            center_count=dict(generated_triples=0,content_excluded=0,identical_triples=0,eligible_jobs=0,witnesses=0)
            from solver.three_half_row_completion import (three_half_row_fibers,three_half_triple_count,
                                                         four_half_row_fibers,four_half_triple_count)
            fiber_generator={1:two_half_row_fibers,2:three_half_row_fibers,3:four_half_row_fibers}[args.varied_halves]
            for job in fiber_generator(*item['words'],radius=args.radius):
                totals['generated_triples']+=1;center_count['generated_triples']+=1
                if job['kind']!='ELIGIBLE':
                    name='content_excluded' if job['kind']=='CONTENT_EXCLUDED' else 'identical_triples'
                    totals[name]+=1;center_count[name]+=1;continue
                totals['eligible_jobs']+=1;center_count['eligible_jobs']+=1
                key=(job['free'],job['fixed_masks']);cached=key in cache
                if cached:
                    out=cache.pop(key);cache[key]=out;totals['cache_hits']+=1
                else:
                    out=complete_three_parity_rows(job['halves'],free_component=job['free'],
                        trace_backend=args.trace_backend,certified_psd=args.certified_psd,joint_gram=args.joint_gram,
                        finite_trace=args.finite_trace,batched_psd=args.batched_psd,affine_finite=args.affine_finite)
                    if not out['complete']:raise RuntimeError('eligible fixed-triple completion unfinished')
                    totals['actual_fibers_solved']+=1;totals['norm_visited']+=out['norm_visited']
                    if args.fiber_cache:
                        cache[key]=out
                        if len(cache)>args.fiber_cache:cache.popitem(last=False)
                totals['profiles_decided']+=out['profiles_decided'];totals['support_orbits']+=out['completed_support_orbits']
                totals['norm_trace_candidates']+=out['norm_trace_candidates']
                totals['joint_gram_targets_tested']+=out['joint_gram_targets_tested']
                totals['joint_gram_targets_excluded']+=out['joint_gram_targets_excluded']
                totals['finite_trace_targets_tested']+=out['finite_trace_targets_tested']
                totals['finite_trace_targets_excluded']+=out['finite_trace_targets_excluded']
                totals['psd_certificates']+=sum('psd_certificate' in r['result'] for r in out['profile_records'])
                for hit in out['witnesses']:
                    totals['full_models']+=1;center_count['witnesses']+=1;new_key=discovery_key(*hit['words'])
                    if new_key not in known:
                        if new_key not in keys and first_hit is None:first_hit=perf_counter()-tick
                        keys.add(new_key);append_verified_solution_if_new(86,*hit['words'],args.output)
                ledger.write(json.dumps(dict(center_index=center_index,varied=job['varied'],free=job['free'],
                    flips=job['flips'],fixed_masks=job['fixed_masks'],cached=cached,result=out))+'\n')
                if totals['eligible_jobs']%2000==0:
                    ledger.flush();print(json.dumps(dict(eligible_jobs=totals['eligible_jobs'],full_new=len(keys),elapsed=perf_counter()-tick)),flush=True)
            from math import comb
            expected=(12*sum(comb(43,r) for r in range(args.radius+1)) if args.varied_halves==1 else
                      three_half_triple_count(43,args.radius) if args.varied_halves==2 else four_half_triple_count(43,args.radius))
            if center_count['generated_triples']!=expected:raise RuntimeError('all ordered-role flip assignments not covered')
            rows.append(dict(center_index=center_index,**center_count));ledger.flush()
            report=dict(complete=len(rows)==len(inputs['seeds']),centers=len(inputs['seeds']),completed_centers=len(rows),
                radius=args.radius,ordered_roles=ordered_roles,certified_psd=args.certified_psd,trace_backend=args.trace_backend,joint_gram=args.joint_gram,
                finite_trace=args.finite_trace,
                varied_halves=args.varied_halves,batched_psd=args.batched_psd,
                affine_finite=args.affine_finite,
                fiber_cache_capacity=args.fiber_cache,rows=rows,**totals,raw_assignments_with_overlaps=totals['generated_triples']*(1<<43),
                ledger_file=ledger_path.name,
                independently_verified_new_classes=len(keys),first_verified_elapsed=first_hit,elapsed=perf_counter()-tick,
                source_manifest_sha256=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest(),
                scope=('All 12 varied/free roles at selected audited centers. Arbitrary-position flips through radius in varied half; '
                       'all 43 unknown bits in free half. Other two literal halves fixed. ' if args.varied_halves==1 else
                       'All 12 roles: two nontrivially changed known halves share the total arbitrary-position flip radius; '
                       'all 43 unknown bits in the free half; only the fourth literal half fixed. ' if args.varied_halves==2 else
                       'All four free roles: three known halves all change nontrivially within the total arbitrary-position flip radius; '
                       'all 43 unknown bits in the free half; no literal row held fixed. ')+
                      'Counts/targets unrestricted. '
                      'Identical literal triples only are cached; no homometric or witness cap. Finite region, no global exclusion.',
                claim='No >=20% new-class-discovery claim from local timings or zero discoveries.')
            atomic_write_json(args.output/'report.json',report)
            print(json.dumps(dict(completed_centers=len(rows),complete=report['complete'],new_classes=len(keys),elapsed=report['elapsed'])),flush=True)
    finally:ledger.close()


if __name__=='__main__':main()
