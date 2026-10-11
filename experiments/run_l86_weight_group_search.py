"""IDE Run: experimental canonical weight-domain search, all 53 cases.

Fresh parent content proof, frozen sources, bounded exact C/G decisions,
literal replay traces and independently verified new-class accounting.
The established norm-search entry remains unchanged. This experiment
changes the proposal distribution, not the mathematical solution domain.
"""
import argparse
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import gzip
import hashlib
import json
import multiprocessing
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
from time import clock_gettime, CLOCK_MONOTONIC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import benchmark_weight_group_pipeline as pipeline
from experiments.benchmark_l86_discovery import freeze_sources, verified_inventory
from solver.checkpoint import atomic_write_json
from solver.projection_group_targets import group_projection_targets
from solver.search_runner import append_verified_solution_if_new
from solver.weight_group_targets import group_weight_targets

RUN_TRIALS = 1000000
RUN_SEED = None  # None records a fresh seed on every IDE Run.
RUN_PAIRED_C = True  # Exact common-basis raw pruning; false restores separate C solves.
RUN_WEIGHT_PREFIX = True  # Incremental exact-weight prefix; false restores the prior backend.
RUN_COUNT_ELIMINATION = True
RUN_FIELD_REBASING = False  # Research control; the large matched run showed no gain.
RUN_PHASE_RANK_PRUNING = True  # Exact literal-rank transfer; keeps RNG and masks.


def clock():
    return clock_gettime(CLOCK_MONOTONIC)


def block(jobs):
    return [pipeline.trial(job) for job in jobs]


def bounded_trials(pool, jobs, workers):
    pending = deque()
    iterator = iter(jobs)

    def submit():
        batch = []
        for _ in range(32):
            try:
                batch.append(next(iterator))
            except StopIteration:
                break
        if batch:
            pending.append(pool.submit(block, tuple(batch)))
        return bool(batch)

    for _ in range(2*workers):
        if not submit():
            break
    while pending:
        rows = pending.popleft().result()
        submit()
        yield from rows


def coarse_carriers(row, group, sharing):
    """Export every visited positive C node in the established carrier format."""
    if row['result'] is None:
        return
    records = row['result']['c_groups'] if sharing else row['result']
    for index, record in enumerate(records):
        coarse = record['coarse'] if sharing else record['attempts']['coarse']
        if coarse is None or coarse['status'] != 'SAT_COMPRESSED':
            continue
        entry = group['c_groups'][index]
        yield dict(trial=row['trial'], seed=row['seed'], weight_group=row['group'],
            c_group=index, target=pipeline.asdict(entry['targets'][0]), contents=entry['contents'],
            target_group=[pipeline.asdict(t) for t in entry['targets']],
            u=row['u'], sampling=row['sampling'],
            result=dict(verified=False, attempts=dict(coarse=coarse)),
            scope='Compressed carrier only; full discoveries are counted separately.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'results'/(
        'L86_weight_group_search_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f')))
    parser.add_argument('--trials', type=int, default=RUN_TRIALS)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=secrets.randbelow(2**31) if RUN_SEED is None else RUN_SEED)
    parser.add_argument('--max-assignments', type=int, default=2**23)
    parser.add_argument('--sharing', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--paired-c', action=argparse.BooleanOptionalAction, default=RUN_PAIRED_C)
    parser.add_argument('--weight-prefix',action=argparse.BooleanOptionalAction,default=RUN_WEIGHT_PREFIX)
    parser.add_argument('--count-elimination',action=argparse.BooleanOptionalAction,default=RUN_COUNT_ELIMINATION)
    parser.add_argument('--field-rebasing',action=argparse.BooleanOptionalAction,default=RUN_FIELD_REBASING)
    parser.add_argument('--phase-rank-pruning',action=argparse.BooleanOptionalAction,default=RUN_PHASE_RANK_PRUNING)
    parser.add_argument('--_frozen-runner', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--_started-clock', type=float, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.trials < 1 or not 1 <= args.workers <= 16 or not 1 <= args.max_assignments <= 2**31:
        parser.error('positive trials, 1..16 workers and bounded native budget required')
    if args.field_rebasing and (not args.weight_prefix or not args.count_elimination):
        parser.error('field rebasing requires the count-eliminated weight-prefix backend')
    output = args.output.resolve()
    snapshot = output/'source_snapshot'
    started = args._started_clock if args._frozen_runner else clock()
    if started is None or not 0 < started <= clock():
        parser.error('valid parent monotonic start clock required')
    if not args._frozen_runner:
        output.mkdir(parents=True, exist_ok=False)
        freeze_sources(snapshot)
        for name in ('benchmark_l86_discovery.py', 'search_norm_supports.py',
                     'benchmark_weight_group_pipeline.py', Path(__file__).name):
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
            '--_frozen-runner', '--_started-clock', str(started), '--output', str(output),
            '--trials', str(args.trials), '--workers', str(args.workers), '--seed', str(args.seed),
            '--max-assignments', str(args.max_assignments), '--sharing' if args.sharing else '--no-sharing',
            '--paired-c' if args.paired_c else '--no-paired-c',
            '--weight-prefix' if args.weight_prefix else '--no-weight-prefix']
        command.extend(('--count-elimination' if args.count_elimination else '--no-count-elimination',
                        '--field-rebasing' if args.field_rebasing else '--no-field-rebasing'))
        command.append('--phase-rank-pruning' if args.phase_rank_pruning else '--no-phase-rank-pruning')
        atomic_write_json(output/'execution.json', dict(command=command, cwd=str(snapshot),
            actual_frozen_execution=True, start_clock=started,
            source_manifest_sha256=hashlib.sha256(pipeline.encode(manifest).encode()).hexdigest()))
        raise SystemExit(subprocess.run(command, cwd=snapshot).returncode)
    if ROOT != snapshot:
        raise RuntimeError('search must execute archived sources')
    manifest = json.loads((snapshot/'source_manifest.json').read_text())
    if any(hashlib.sha256((snapshot/name).read_bytes()).hexdigest() != digest
           for name, digest in manifest.items()):
        raise RuntimeError('frozen source changed')
    import experiments.search_norm_supports as search
    search.initialize()
    groups = group_weight_targets(group_projection_targets(search.CHOICES))
    atomic_write_json(output/'target_domain.json', json.loads(pipeline.encode(groups)))
    known = verified_inventory((output/'initial_inventory.txt').read_bytes())
    statuses = Counter()
    carriers = []
    keys = set()
    first = None
    tested = scheduled = paired_trials = prefix_trials = 0
    context = multiprocessing.get_context('spawn')
    with gzip.open(output/'literal_trials.jsonl.gz', 'wt', compresslevel=1) as log:
        with ProcessPoolExecutor(args.workers, mp_context=context,
            initializer=pipeline.initialize_worker,
            initargs=(groups, args.paired_c,args.weight_prefix,False,args.count_elimination,False,False,args.field_rebasing,False,args.phase_rank_pruning)) as pool:
            jobs = ((i, args.seed, args.max_assignments, 'candidate' if args.sharing else 'rollback')
                    for i in range(args.trials))
            for row in bounded_trials(pool, jobs, args.workers):
                tested += 1
                statuses[row['status']] += 1
                group = groups[row['group']]
                scheduled += sum(len(e['targets']) for e in group['c_groups'])
                if args.sharing and row['result'] is not None:
                    paired_trials += any(c['coarse'].get('hybrid_backend') == 'exact_paired_bit_tail'
                                         for c in row['result']['c_groups'])
                    prefix_trials += any(c['coarse'].get('hybrid_backend') == 'exact_weight_prefix'
                                         for c in row['result']['c_groups'])
                carriers.extend(coarse_carriers(row, group, args.sharing))
                key = row['discovery_key']
                is_new = key is not None and key not in known and key not in keys
                if is_new:
                    first = clock()-started if first is None else first
                    keys.add(key)
                    append_verified_solution_if_new(86, *row['words'], output)
                log.write(pipeline.encode(row)+'\n')
                if tested % 4096 == 0 or tested == args.trials or is_new:
                    log.flush()
                    report = dict(complete=tested == args.trials, seed=args.seed, tested=tested,
                        planned_trials=args.trials, workers=args.workers, sharing=args.sharing,
                        paired_c=args.paired_c, paired_native_trials=paired_trials,
                        weight_prefix=args.weight_prefix,weight_prefix_native_trials=prefix_trials,
                        count_elimination=args.count_elimination,field_rebasing=args.field_rebasing,
                        phase_rank_pruning=args.phase_rank_pruning,
                        support_domains=len(groups), c_groups=sum(len(g['c_groups']) for g in groups),
                        full_target_cases=sum(len(e['targets']) for g in groups for e in g['c_groups']),
                        scheduled_full_target_cases=scheduled, statuses=dict(statuses),
                        coarse_witnesses=len(carriers), independently_verified_new_classes=len(keys),
                        stopped_on_new=is_new, first_verified_elapsed=first, elapsed=clock()-started,
                        actual_frozen_execution=True,
                        scope='Canonical small-row proposals with all 53 target/content cases. Composite exact decisions retain UNKNOWN. Stop at first verified new class. No global exclusions or discovery-speed claim from misses.')
                    atomic_write_json(output/'report.json', report)
                    atomic_write_json(output/'coarse_witnesses.json', dict(rows=carriers,
                        complete=tested == args.trials, coarse_only=True, full_new_classes=len(keys),
                        source='literal_trials.jsonl.gz'))
                    print(pipeline.encode(report), flush=True)
                if is_new:
                    break


if __name__ == '__main__':
    main()
