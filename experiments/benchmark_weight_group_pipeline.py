"""IDE Run: frozen, matched row-normalized support-domain benchmark.

Both variants sample the SAME canonical small-row supports, with the SAME
seeds and all requested target/content edges. Rollback regenerates the
support for each C group; candidate generates once and memoizes exact sign
nodes. This does not compare the old large-row proposal distribution or
establish the waiting time for a new L86 equivalence class.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, is_dataclass
from datetime import datetime
import gzip
import hashlib
import json
import multiprocessing
from pathlib import Path
import random
import shutil
import statistics
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.benchmark_l86_discovery import discovery_key, freeze_sources, verified_inventory
from solver.checkpoint import atomic_write_json
from solver.cyclic_support import parity_support_family
from solver.cyclic_support_sampling import sample_weighted_support
from solver.projection_group_completion import complete_projection_group
from solver.projection_group_targets import group_projection_targets
from solver.weight_group_completion import complete_weight_group
from solver.weight_group_targets import group_weight_targets
from solver.verifier import verify_pqcp

GROUPS = None
PAIRED_C = False
WEIGHT_PREFIX = False
ORIGINAL_NORM = False
COUNT_ELIMINATION = False
WIDE_PREFIX = False
AFFINE_PRUNING = False
FIELD_REBASING = False
MULTI_PIVOT = False
PHASE_GRID = False


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      default=lambda obj: asdict(obj) if is_dataclass(obj) else obj)


def initialize_worker(groups, paired_c=False, weight_prefix=False,original_norm=False,count_elimination=False,wide_prefix=False,affine_pruning=False,field_rebasing=False,multi_pivot=False,phase_grid=False):
    global GROUPS, PAIRED_C, WEIGHT_PREFIX,ORIGINAL_NORM,COUNT_ELIMINATION,WIDE_PREFIX,AFFINE_PRUNING,FIELD_REBASING,MULTI_PIVOT,PHASE_GRID
    if type(paired_c) is not bool or type(weight_prefix) is not bool:
        raise ValueError('boolean paired-C and weight-prefix flags required')
    GROUPS = groups
    PAIRED_C = paired_c
    WEIGHT_PREFIX = weight_prefix
    ORIGINAL_NORM=original_norm
    COUNT_ELIMINATION=count_elimination
    WIDE_PREFIX=wide_prefix
    AFFINE_PRUNING=affine_pruning
    FIELD_REBASING=field_rebasing
    MULTI_PIVOT=multi_pivot
    PHASE_GRID=phase_grid
    if original_norm:
        import experiments.search_norm_supports as search
        search._initialize_worker_choices(tuple((g['targets'][0],g['contents']) for g in groups),
            True,False,tuple(g['targets'] for g in groups),True,False,False)


def sample(group, number, seed):
    rng = random.Random(seed + 1000003 * number)
    d = group['c_groups'][0]['targets'][0].L // 2
    target = group['c_groups'][0]['targets'][0]
    u = sum(1 << i for i in rng.sample(range(d), group['weights'][0]))
    family = parity_support_family(u, d, min(target.k, d - target.k))
    return u, sample_weighted_support(family, group['weights'][1], rng,
                                     phase_count_pruning=True)


def trial(job):
    number, seed, budget, variant = job
    index = number % len(GROUPS)
    group = GROUPS[index]
    from solver import weight_prefix_native as prefix
    from solver import cyclic_support_sampling as sampling
    sampling.DEFAULT_PHASE_RANK_PRUNING=PHASE_GRID and variant=='candidate'
    prefix.DEFAULT_COUNT_ELIMINATION=COUNT_ELIMINATION and (variant=='candidate' or WIDE_PREFIX or AFFINE_PRUNING or FIELD_REBASING or MULTI_PIVOT or PHASE_GRID)
    prefix.DEFAULT_AFFINE_PRUNING=AFFINE_PRUNING and variant=='candidate'
    prefix.DEFAULT_FIELD_REBASING=FIELD_REBASING and variant=='candidate'
    prefix.DEFAULT_MULTI_PIVOT=MULTI_PIVOT and variant=='candidate'
    prefix.MAX_FREE_BITS=14 if not WIDE_PREFIX or variant=='candidate' else 12
    if ORIGINAL_NORM:
        import experiments.search_norm_supports as search
        search.WEIGHT_PREFIX=WEIGHT_PREFIX and (variant=='candidate' or COUNT_ELIMINATION or WIDE_PREFIX or AFFINE_PRUNING)
        row=search.trial((number,seed,budget))
        result=row['result'];words=None if result is None else result['words']
        row.update(group=index,verified=words is not None,words=words,
            actual_target=None if result is None else result['actual_target'],
            discovery_key=discovery_key(*words) if words is not None else None)
        return row
    witness = actual = None
    unknown = False
    if variant in ('candidate', 'rollback-shared', 'rollback-paired','rollback-prefix'):
        u, sampled = sample(group, number, seed)
        result = None if sampled['mask'] is None else complete_weight_group(
            group, (u, sampled['mask']), max_assignments=budget,
            paired_c=PAIRED_C and variant in ('candidate','rollback-paired','rollback-prefix'),
            weight_prefix=WEIGHT_PREFIX and variant in ('candidate','rollback-prefix'))
        if result is not None:
            witness, actual = result['words'], result['actual_target']
            unknown = result['status'] == 'UNKNOWN'
    else:
        results = []
        literal = None
        for entry in group['c_groups']:
            u, sampled = sample(group, number, seed)
            if literal is None:
                literal = (u, sampled)
            elif literal != (u, sampled):
                raise RuntimeError('canonical rollback supports differ')
            if sampled['mask'] is None:
                continue
            references = tuple(tuple([-1]*n + [0]*z + [1]*p)
                               for n, z, p in entry['contents'])
            out = complete_projection_group(*references, entry['targets'],
                (u, sampled['mask']), max_assignments=budget)
            results.append(out)
            unknown |= out['status'] == 'UNKNOWN'
            if out['verified'] and witness is None:
                witness, actual = out['words'], out['actual_target']
        result = results
    status = ('SAT_PQCP' if witness is not None else 'UNKNOWN' if unknown else
              'UNSAT_WEIGHT_FOR_FIXED_SIDE' if sampled['mask'] is None else
              'UNSAT_BINARY_WEIGHT_GROUP')
    # A full witness is independently checked and classified here, in both variants.
    if witness is not None and not verify_pqcp(*witness).is_valid:
        raise RuntimeError('whole pipeline witness failed independent verifier')
    # Small positive controls exercise this pipeline without entering L86 accounting.
    key = discovery_key(*witness) if witness is not None and len(witness[0]) == 86 else None
    return dict(trial=number, seed=seed, group=index, u=u, sampling=sampled,
                status=status, verified=witness is not None, words=witness,
                actual_target=actual, discovery_key=key, result=result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'results'/(
        'L86_weight_group_pipeline_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f')))
    parser.add_argument('--trials', type=int, default=32768)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--seeds', type=int, nargs='+', default=(123, 456, 789))
    parser.add_argument('--max-assignments', type=int, default=2**23)
    parser.add_argument('--paired-c', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--weight-prefix', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--count-elimination',action=argparse.BooleanOptionalAction,default=False)
    parser.add_argument('--wide-prefix',action=argparse.BooleanOptionalAction,default=False)
    parser.add_argument('--affine-pruning',action=argparse.BooleanOptionalAction,default=False)
    parser.add_argument('--field-rebasing',action=argparse.BooleanOptionalAction,default=False)
    parser.add_argument('--multi-pivot',action=argparse.BooleanOptionalAction,default=False)
    parser.add_argument('--phase-grid',action=argparse.BooleanOptionalAction,default=False)
    parser.add_argument('--original-norm',action='store_true',
                        help='Replay the established 27 ordered groups and original support proposal direction')
    parser.add_argument('--comparison', choices=('generation', 'paired-c','weight-prefix'), default='generation',
                        help='paired-c rollback already shares support generation; isolate the paired native kernel')
    parser.add_argument('--_frozen-runner', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 1 <= args.trials <= 262144 or not 1 <= args.workers <= 16 or not 1 <= args.max_assignments <= 2**31:
        parser.error('bounded trials, workers and native assignment budget required')
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        parser.error('distinct seeds required')
    if args.original_norm and args.paired_c:parser.error('original norm groups use one C node, not paired-C grouping')
    if args.affine_pruning and (not args.weight_prefix or not args.count_elimination or args.wide_prefix):
        parser.error('affine comparison requires weight-prefix and count-elimination, with one common width cap')
    if args.field_rebasing and (not args.weight_prefix or not args.count_elimination or args.wide_prefix or args.affine_pruning):
        parser.error('field rebasing requires weight-prefix and count-elimination, as the only comparison')
    if args.multi_pivot and (not args.weight_prefix or not args.count_elimination or args.wide_prefix or args.affine_pruning or args.field_rebasing):
        parser.error('multi-pivot requires weight-prefix and count-elimination, as the only comparison')
    if args.phase_grid and (not args.original_norm or not args.weight_prefix or not args.count_elimination or
                           args.wide_prefix or args.affine_pruning or args.field_rebasing or args.multi_pivot):
        parser.error('phase-grid comparison requires the common current original-norm backend')
    output = args.output.resolve()
    snapshot = output/'source_snapshot'
    if not args._frozen_runner:
        output.mkdir(parents=True, exist_ok=False)
        freeze_sources(snapshot)
        for name in ('benchmark_l86_discovery.py', 'search_norm_supports.py', Path(__file__).name):
            dest = snapshot/'experiments'/name
            dest.parent.mkdir(exist_ok=True)
            shutil.copy2(ROOT/'experiments'/name, dest)
        manifest = {str(p.relative_to(snapshot)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted(snapshot.rglob('*')) if p.is_file() and p.name != 'source_manifest.json'}
        atomic_write_json(snapshot/'source_manifest.json', manifest)
        inventory = (ROOT/'86.txt').read_bytes() if (ROOT/'86.txt').exists() else b''
        verified_inventory(inventory)
        (output/'initial_inventory.txt').write_bytes(inventory)
        command = [sys.executable, str(snapshot/'experiments'/Path(__file__).name),
            '--_frozen-runner', '--output', str(output), '--trials', str(args.trials),
            '--workers', str(args.workers), '--max-assignments', str(args.max_assignments),
            '--seeds', *map(str, args.seeds), '--comparison', args.comparison,
            '--paired-c' if args.paired_c else '--no-paired-c',
            '--weight-prefix' if args.weight_prefix else '--no-weight-prefix']
        command.append('--count-elimination' if args.count_elimination else '--no-count-elimination')
        command.append('--wide-prefix' if args.wide_prefix else '--no-wide-prefix')
        command.append('--affine-pruning' if args.affine_pruning else '--no-affine-pruning')
        command.append('--field-rebasing' if args.field_rebasing else '--no-field-rebasing')
        command.append('--multi-pivot' if args.multi_pivot else '--no-multi-pivot')
        command.append('--phase-grid' if args.phase_grid else '--no-phase-grid')
        if args.original_norm:command.append('--original-norm')
        atomic_write_json(output/'execution.json', dict(command=command, cwd=str(snapshot),
            manifest_sha256=hashlib.sha256(encode(manifest).encode()).hexdigest()))
        raise SystemExit(subprocess.run(command, cwd=snapshot).returncode)
    if ROOT != snapshot:
        raise RuntimeError('benchmark must execute the archived sources')
    manifest = json.loads((snapshot/'source_manifest.json').read_text())
    if any(hashlib.sha256((snapshot/name).read_bytes()).hexdigest() != digest
           for name, digest in manifest.items()):
        raise RuntimeError('source manifest mismatch')
    import experiments.search_norm_supports as search
    started = perf_counter()
    search.initialize()
    projected=group_projection_targets(search.CHOICES)
    groups = projected if args.original_norm else group_weight_targets(projected)
    packet = encode(groups)
    (output/'target_domain.json').write_text(packet+'\n')
    known = verified_inventory((output/'initial_inventory.txt').read_bytes())
    rows = []
    context = multiprocessing.get_context('spawn')
    for seed_index, seed in enumerate(args.seeds):
        order = ('rollback', 'candidate') if seed_index % 2 == 0 else ('candidate', 'rollback')
        for variant in order:
            tick = perf_counter()
            digest = hashlib.sha256()
            statuses = Counter()
            keys = set()
            trace = output/f'{variant}_seed{seed}.jsonl.gz'
            with gzip.open(trace, 'wt', compresslevel=1) as log:
                with ProcessPoolExecutor(args.workers, mp_context=context,
                    initializer=initialize_worker, initargs=(groups, args.paired_c,args.weight_prefix,args.original_norm,args.count_elimination,args.wide_prefix,args.affine_pruning,args.field_rebasing,args.multi_pivot,args.phase_grid)) as pool:
                    job_variant = ('rollback-prefix' if variant=='rollback' and args.count_elimination else
                                   'rollback-paired' if variant=='rollback' and args.comparison=='weight-prefix' else
                                   'rollback-shared' if variant=='rollback' and args.comparison=='paired-c' else variant)
                    jobs = ((i, seed, args.max_assignments, job_variant) for i in range(args.trials))
                    for row in pool.map(trial, jobs, chunksize=32):
                        statuses[row['status']] += 1
                        # Only literal inputs and composite decisions must match:
                        # two correct sign solvers may return different full witnesses.
                        comparable = {k: row[k] for k in ('trial', 'seed', 'group', 'u', 'sampling', 'status', 'verified')}
                        digest.update((encode(comparable)+'\n').encode())
                        if row['discovery_key'] is not None:
                            keys.add(row['discovery_key'])
                        log.write(encode(row)+'\n')
            rows.append(dict(seed=seed, variant=variant, pipeline_seconds=perf_counter()-tick,
                tested=sum(statuses.values()), statuses=dict(statuses), literal_decisions_sha256=digest.hexdigest(),
                independently_verified_new_classes=len(keys-known), trace=trace.name))
            report = dict(complete=False, executed_frozen_source=True, trials=args.trials,
                workers=args.workers, support_domains=len(groups), c_groups=len(groups) if args.original_norm else sum(len(g['c_groups']) for g in groups),
                paired_c=args.paired_c, weight_prefix=args.weight_prefix,comparison=args.comparison,
                full_target_cases=sum(len(g['targets']) for g in groups) if args.original_norm else sum(len(e['targets']) for g in groups for e in g['c_groups']),
                original_norm=args.original_norm,
                count_elimination=args.count_elimination,
                wide_prefix=args.wide_prefix,
                affine_pruning=args.affine_pruning,
                field_rebasing=args.field_rebasing,
                multi_pivot=args.multi_pivot,
                phase_grid=args.phase_grid,
                target_domain_sha256=hashlib.sha256(packet.encode()).hexdigest(), rows=rows,
                total_wall_seconds=perf_counter()-started,
                scope=('Identical established 27 ordered support proposals and all 53 requested target/content cases. No proposal-direction or group scheduling changes. '
                       if args.original_norm else 'Matched canonical small-row proposals, full requested target edges and literal decisions. Different from the old large-row proposal distribution. ')+
                      'Concurrent research jobs may affect timings. Zero discoveries do not measure discovery waiting time.')
            atomic_write_json(output/'report.json', report)
            print(encode(rows[-1]), flush=True)
    for seed in args.seeds:
        pair = [r for r in rows if r['seed'] == seed]
        if pair[0]['literal_decisions_sha256'] != pair[1]['literal_decisions_sha256']:
            raise RuntimeError('paired literal inputs/full composite decisions differ')
    medians = {v: statistics.median(r['pipeline_seconds'] for r in rows if r['variant'] == v)
               for v in ('rollback', 'candidate')}
    report.update(complete=True, all_literal_inputs_and_composite_decisions_match=True,
        median_pipeline_seconds=medians,
        throughput_improvement_percent=100*(medians['rollback']/medians['candidate']-1),
        time_reduction_percent=100*(1-medians['candidate']/medians['rollback']))
    atomic_write_json(output/'report.json', report)
    print(encode({k: report[k] for k in ('complete', 'median_pipeline_seconds', 'throughput_improvement_percent')}), flush=True)


if __name__ == '__main__':
    main()
