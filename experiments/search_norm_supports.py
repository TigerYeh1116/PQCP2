"""Independent norm-compatible support sampling and exact C+G completion.

Do not seed from saved PQCPs or nearest-distance candidates. Explicitly cycle
all proven content choices; choose a fixed-weight U and a literal compatible
V, then solve both sign problems. No sample or coarse SAT is a full discovery.
"""
import argparse
from collections import Counter,deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import gzip
import json
import multiprocessing
from pathlib import Path
import random
import shutil
import sys
import subprocess
import math
from time import clock_gettime,CLOCK_MONOTONIC

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.benchmark_l86_discovery import discovery_key,freeze_sources,verified_inventory
from solver.alternation_orbits import reduce_alternation_profile_orbits
from solver.checkpoint import atomic_write_json
from solver.cyclic_support import parity_support_family
from solver.cyclic_support_sampling import sample_weighted_support
from solver import cyclic_support_sampling as support_sampling
from solver.pqcp_support_solver import complete_pqcp_supports
from solver.search_runner import append_verified_solution_if_new
from solver.target_profiles import canonical_target_content_profiles
from solver.torch_compressed_search import search_paired_contents

CHOICES=None
PHASE_COUNT_PRUNING=False
EVEN_SIGN_ORBIT=False
TARGET_GROUPS=None
SHARED_C_COMPLETION=True
DIFFERENCE_FIRST=False
WEIGHT_PREFIX=False


def search_clock():
    # POSIX clock shared by parent and frozen child, including setup/imports.
    return clock_gettime(CLOCK_MONOTONIC)


def semantic_trial(row):
    """Literal inputs, exact outputs and work counts, excluding timing/phase scheduling."""
    def clean(value):
        if isinstance(value,dict):
            return {k:clean(v) for k,v in value.items()
                    if k not in ('elapsed','construction_seconds','solve_seconds')
                    and not k.startswith('phase_') and k!='counted_family_size'}
        if isinstance(value,(tuple,list)):return [clean(v) for v in value]
        return value
    return clean(row)


def initialize(alternation_pruning=False,phase_count_pruning=False,even_sign_orbit=False,difference_first=False,weight_prefix=False):
    global CHOICES,PHASE_COUNT_PRUNING,EVEN_SIGN_ORBIT,TARGET_GROUPS,SHARED_C_COMPLETION,DIFFERENCE_FIRST
    PHASE_COUNT_PRUNING=phase_count_pruning
    EVEN_SIGN_ORBIT=even_sign_orbit
    DIFFERENCE_FIRST=difference_first
    global WEIGHT_PREFIX
    WEIGHT_PREFIX=weight_prefix
    TARGET_GROUPS=None;SHARED_C_COMPLETION=True
    targets=canonical_target_content_profiles(86)
    if alternation_pruning:
        targets,_=reduce_alternation_profile_orbits(targets)
    CHOICES=tuple((target,contents) for target in targets
                  for contents in search_paired_contents(target))


def _initialize_worker_choices(choices,phase_count_pruning,even_sign_orbit,target_groups=None,shared_c_completion=True,difference_first=False,weight_prefix=False):
    """Trusted parent IPC only: reuse its already proven content domain.

    No user checkpoint/JSON supplies this list. The frozen parent computes
    search_paired_contents once; every worker retains its exact tuple/order.
    """
    global CHOICES,PHASE_COUNT_PRUNING,EVEN_SIGN_ORBIT,TARGET_GROUPS,SHARED_C_COMPLETION,DIFFERENCE_FIRST
    if not isinstance(choices,tuple) or not choices:raise ValueError('nonempty parent content domain required')
    from solver.compressed_repair import _validated_reference
    for target,contents in choices:
        references=tuple(tuple([-1]*n+[0]*z+[1]*p) for n,z,p in contents)
        _validated_reference(*references,target)
    CHOICES=choices;PHASE_COUNT_PRUNING=phase_count_pruning;EVEN_SIGN_ORBIT=even_sign_orbit
    if target_groups is not None and len(target_groups)!=len(choices):raise ValueError('one target group per content choice required')
    TARGET_GROUPS=target_groups;SHARED_C_COMPLETION=shared_c_completion
    DIFFERENCE_FIRST=difference_first
    global WEIGHT_PREFIX
    WEIGHT_PREFIX=weight_prefix


def trial(job):
    number,seed,budget=job
    index=number%len(CHOICES);target,contents=CHOICES[index]
    rng=random.Random(seed+1000003*number)
    weights=tuple(n+p for n,z,p in contents)
    u=sum(1<<i for i in rng.sample(range(43),weights[0]))
    k=min(target.k,43-target.k)
    family=parity_support_family(u,43,k)
    sampled=sample_weighted_support(family,weights[1],rng,phase_count_pruning=PHASE_COUNT_PRUNING)
    row=dict(trial=number,seed=seed,target=asdict(target),contents=contents,u=u,sampling=sampled)
    if TARGET_GROUPS is not None:row['target_group']=[asdict(t) for t in TARGET_GROUPS[index]]
    if sampled['mask'] is None:
        row.update(status=sampled['status'],result=None);return row
    references=tuple(tuple([-1]*n+[0]*z+[1]*p) for n,z,p in contents)
    # The same content-only predicate was already applied in the parent to
    # every scheduled choice. It cannot exclude any of these trial supports.
    if TARGET_GROUPS is None:
        out=complete_pqcp_supports(*references,target,(u,sampled['mask']),max_assignments=budget,
                                 even_sign_orbit=EVEN_SIGN_ORBIT,binary_content_pruning=False)
    elif SHARED_C_COMPLETION:
        from solver.projection_group_completion import complete_projection_group
        out=complete_projection_group(*references,TARGET_GROUPS[index],(u,sampled['mask']),
                                     max_assignments=budget,even_sign_orbit=EVEN_SIGN_ORBIT,difference_first=DIFFERENCE_FIRST,weight_prefix=WEIGHT_PREFIX)
    else:
        tick=search_clock();attempts=[];witness=None;actual=None;coarse={}
        for t in TARGET_GROUPS[index]:
            result=complete_pqcp_supports(*references,t,(u,sampled['mask']),max_assignments=budget,
                                         even_sign_orbit=EVEN_SIGN_ORBIT,binary_content_pruning=False)
            attempts.append(dict(target=asdict(t),result=result))
            c=result['attempts'].get('coarse')
            if c is not None and (not coarse or c['status']=='SAT_COMPRESSED'):coarse=c
            if result['verified']:witness=result['words'];actual=asdict(t);break
        status='SAT_PQCP' if witness is not None else 'UNKNOWN' if any(a['result']['status']=='UNKNOWN' for a in attempts) else 'UNSAT_BINARY_SUPPORT_GROUP'
        out=dict(status=status,words=witness,verified=witness is not None,actual_target=actual,
                 attempts=dict(coarse=coarse,independent_targets=attempts),elapsed=search_clock()-tick)
    row.update(status=out['status'],result=out)
    return row


def parse_c_weights(text):
    """'19-24' or '19,20,21' -> frozenset of C_A weights; None keeps every proven content."""
    if text is None:return None
    wanted=set()
    for part in text.split(','):
        lo,_,hi=part.strip().partition('-')
        wanted.update(range(int(lo),int(hi or lo)+1))
    if not wanted or not all(0<w<43 for w in wanted):raise ValueError('C_A weights must lie in 1..42')
    return frozenset(wanted)


def block_trial(jobs):
    return [trial(job) for job in jobs]


def bounded_results(pool,jobs,workers,block_size=32):
    """Keep at most 2*workers blocks in flight, preserving trial order."""
    iterator=iter(jobs)
    pending=deque()
    def submit():
        block=[]
        for _ in range(block_size):
            try:block.append(next(iterator))
            except StopIteration:break
        if block:pending.append(pool.submit(block_trial,tuple(block)))
        return bool(block)
    for _ in range(2*workers):
        if not submit():break
    while pending:
        block=pending.popleft().result()
        submit()
        yield from block


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--trials',type=int,default=10000)
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--seed',type=int,default=123)
    parser.add_argument('--max-assignments',type=int,default=2**23)
    parser.add_argument('--alternation-orbit-pruning',action=argparse.BooleanOptionalAction,default=False)
    parser.add_argument('--phase-count-pruning',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--gzip-trace',action=argparse.BooleanOptionalAction,default=False)
    parser.add_argument('--even-sign-orbit-pruning',action=argparse.BooleanOptionalAction,default=False)
    parser.add_argument('--projection-groups',action=argparse.BooleanOptionalAction,default=False)
    parser.add_argument('--shared-c-completion',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--difference-first',action=argparse.BooleanOptionalAction,default=False,
                        help='Experimental exact G-first ordering to retain difference carriers; default keeps the estimated-cost ordering.')
    parser.add_argument('--stop-on-new',action=argparse.BooleanOptionalAction,default=False,
                        help='Stop after the first independently verified new class; unvisited trials remain unclaimed.')
    parser.add_argument('--weight-prefix',action=argparse.BooleanOptionalAction,default=False)
    parser.add_argument('--c-weights',type=str,default=None,
                        help='Sample only content choices whose C_A weight lies here, e.g. 19-24. '
                             'Focuses the proposal distribution; every sampled support is still decided exactly.')
    parser.add_argument('--_frozen-runner',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--_started-clock',type=float,help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.trials<1 or not 1<=args.workers<=16 or not 1<=args.max_assignments<=2**31:
        parser.error('positive bounded trials, 1..16 workers and valid sign budget required')
    if args.difference_first and (not args.projection_groups or not args.shared_c_completion):
        parser.error('difference-first requires projection groups with shared C completion')
    if args.weight_prefix and (not args.projection_groups or not args.shared_c_completion):
        parser.error('weight-prefix requires projection groups with shared C completion')
    try:c_weights=parse_c_weights(args.c_weights)
    except ValueError as error:parser.error(str(error))
    args.output=args.output.resolve()
    tick=args._started_clock if args._frozen_runner else search_clock()
    if tick is None or not math.isfinite(tick) or not 0<tick<=search_clock():parser.error('valid shared start clock required')
    statuses,keys,first_hit=Counter(),set(),None
    snapshot=args.output/'source_snapshot'
    if not args._frozen_runner:
        args.output.mkdir(parents=True,exist_ok=False);freeze_sources(snapshot)
        for name in ('search_norm_supports.py','benchmark_l86_discovery.py'):
            destination=snapshot/'experiments'/name
            destination.parent.mkdir(exist_ok=True);shutil.copy2(ROOT/'experiments'/name,destination)
        manifest={str(p.relative_to(snapshot)):hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in sorted(snapshot.rglob('*')) if p.is_file() and p.name!='source_manifest.json'}
        atomic_write_json(snapshot/'source_manifest.json',manifest)
        inventory=(ROOT/'86.txt').read_bytes() if (ROOT/'86.txt').exists() else b''
        verified_inventory(inventory);(args.output/'initial_inventory.txt').write_bytes(inventory)
        command=[sys.executable,str(snapshot/'experiments/search_norm_supports.py'),'--_frozen-runner',
            '--_started-clock',str(tick),'--output',str(args.output),'--trials',str(args.trials),
            '--workers',str(args.workers),'--seed',str(args.seed),'--max-assignments',str(args.max_assignments)]
        for option,enabled in (('phase-count-pruning',args.phase_count_pruning),('alternation-orbit-pruning',args.alternation_orbit_pruning),
                               ('gzip-trace',args.gzip_trace),('even-sign-orbit-pruning',args.even_sign_orbit_pruning),
                               ('projection-groups',args.projection_groups),('shared-c-completion',args.shared_c_completion),
                               ('difference-first',args.difference_first),('stop-on-new',args.stop_on_new),('weight-prefix',args.weight_prefix)):
            command.append('--'+('' if enabled else 'no-')+option)
        if args.c_weights is not None:command.extend(('--c-weights',args.c_weights))
        code=subprocess.run(command,cwd=snapshot).returncode
        if code:raise SystemExit(code)
        return
    if ROOT.resolve()!=snapshot.resolve():parser.error('internal frozen mode must execute the archived source')
    manifest=json.loads((snapshot/'source_manifest.json').read_text())
    if any(hashlib.sha256((snapshot/name).read_bytes()).hexdigest()!=digest for name,digest in manifest.items()):
        raise RuntimeError('frozen source changed before worker startup')
    inventory=(args.output/'initial_inventory.txt').read_bytes()
    script_digest=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    known=verified_inventory(inventory)
    initialize(args.alternation_orbit_pruning,args.phase_count_pruning,args.even_sign_orbit_pruning,args.difference_first,args.weight_prefix)
    group_targets=None
    if args.projection_groups:
        from solver.projection_group_targets import group_projection_targets
        groups=group_projection_targets(CHOICES)
        choices=tuple((g['targets'][0],g['contents']) for g in groups)
        group_targets=tuple(g['targets'] for g in groups)
        atomic_write_json(args.output/'projection_groups.json',[dict(contents=g['contents'],
            targets=[asdict(t) for t in g['targets']],original_targets=[asdict(t) for t in g['original_targets']],
            target_units=g['target_units']) for g in groups])
    else:choices=CHOICES
    if c_weights is not None:
        keep=[i for i,(_,contents) in enumerate(choices) if contents[0][0]+contents[0][2] in c_weights]
        if not keep:parser.error('no proven content choice has a C_A weight in --c-weights')
        choices=tuple(choices[i] for i in keep)
        if group_targets is not None:group_targets=tuple(group_targets[i] for i in keep)
    from solver.l86_content_certificate import l86_content_certificate
    certificate=l86_content_certificate()
    atomic_write_json(args.output/'content_generation_certificate.json',dict(certificate=asdict(certificate),
        choices=[dict(target=asdict(t),contents=c) for t,c in choices],
        scope='Parent runtime proof and exact ordered content domain shared via trusted process initialization, not imported as user proof.'))
    context=multiprocessing.get_context('spawn')
    tested=0;scheduled_targets=0;semantic_digest=hashlib.sha256();coarse_witnesses=0;coarse_rows=[];stopped_on_new=False
    witnesses_written=0;witnesses_clock=search_clock()
    trace=args.output/('literal_trials.jsonl.gz' if args.gzip_trace else 'literal_trials.jsonl')
    opener=(lambda:gzip.open(trace,'wt',encoding='utf-8',compresslevel=1)) if args.gzip_trace else (lambda:trace.open('w'))
    with opener() as log:
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=context,initializer=_initialize_worker_choices,
                                 initargs=(choices,args.phase_count_pruning,args.even_sign_orbit_pruning,group_targets,args.shared_c_completion,args.difference_first,args.weight_prefix)) as pool:
            jobs=((i,args.seed,args.max_assignments) for i in range(args.trials))
            for row in bounded_results(pool,jobs,args.workers):
                tested+=1;statuses[row['status']]+=1
                scheduled_targets+=len(row.get('target_group',[row['target']]))
                semantic_digest.update((json.dumps(semantic_trial(row),sort_keys=True,separators=(',',':'))+'\n').encode())
                if row['result'] is not None and (row['result'].get('attempts',{}).get('coarse') or {}).get('status')=='SAT_COMPRESSED':
                    coarse_witnesses+=1
                    coarse_rows.append(row)
                if row['result'] is not None and row['result']['verified']:
                    words=row['result']['words'];key=discovery_key(*words)
                    if key not in known:
                        if key not in keys and first_hit is None:first_hit=search_clock()-tick
                        keys.add(key);append_verified_solution_if_new(86,*words,args.output)
                        if args.stop_on_new:stopped_on_new=True
                log.write(json.dumps(row,separators=(',',':'))+'\n')
                if tested%4096==0 or tested==args.trials or row['status']=='SAT_PQCP':
                    log.flush()
                    report=dict(seed=args.seed,trials=args.trials,tested=tested,workers=args.workers,
                        content_choices=len(choices),statuses=dict(statuses),complete=tested==args.trials,
                        projection_groups=args.projection_groups,shared_c_completion=args.shared_c_completion,
                        difference_first=args.difference_first,
                        weight_prefix=args.weight_prefix,
                        c_weights=sorted(c_weights) if c_weights is not None else None,
                        stopped_on_new=stopped_on_new,stop_on_new=args.stop_on_new,
                        scheduled_full_target_cases=scheduled_targets,
                        alternation_orbit_pruning=args.alternation_orbit_pruning,
                        phase_count_pruning=args.phase_count_pruning,
                        phase_rank_pruning=args.phase_count_pruning and support_sampling.DEFAULT_PHASE_RANK_PRUNING,
                        even_sign_orbit_pruning=args.even_sign_orbit_pruning,
                        trace_file=trace.name,semantic_trials_sha256=semantic_digest.hexdigest(),
                        coarse_witnesses=coarse_witnesses,
                        all_sampled_supports_decided=not statuses['UNKNOWN'],
                        source_manifest_sha256=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest(),
                        first_verified_elapsed=first_hit,independently_verified_new_classes=len(keys),
                        elapsed=search_clock()-tick,script_sha256=script_digest,
                        executed_frozen_source=True,clock='POSIX CLOCK_MONOTONIC, parent start includes archive/setup/child imports',
                        parent_content_domain_shared=True,
                        scope=('Independent sampled supports with one shared exact C problem and its requested full G target alternatives; first verified model, UNKNOWN retained.'
                               if args.projection_groups else 'Independent sampled parity-compatible supports with bounded exact two-sign-problem completion.')+
                              ' No global exclusion from misses/UNKNOWN.')
                    atomic_write_json(args.output/'report.json',report)
                    # Rewriting every witness each 4096 trials grows quadratically; refresh at most each minute.
                    if coarse_rows and (len(coarse_rows)!=witnesses_written and search_clock()-witnesses_clock>=60
                                        or tested==args.trials or stopped_on_new or row['status']=='SAT_PQCP'):
                        witnesses_written=len(coarse_rows);witnesses_clock=search_clock()
                        atomic_write_json(args.output/'coarse_witnesses.json',dict(rows=coarse_rows,complete=tested==args.trials,
                            coarse_only=all(not r['result']['verified'] for r in coarse_rows),full_new_classes=len(keys),source=trace.name,
                            semantic_trials_sha256=semantic_digest.hexdigest()))
                    print(json.dumps(report),flush=True)
                if stopped_on_new:break


if __name__=='__main__':main()
