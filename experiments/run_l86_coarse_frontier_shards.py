"""IDE Run: finish the radius-two coarse frontier in parallel chunks.

Continues results/L86_coarse_neighbour_radius2_frontier_20261011 from the
first literal point that was not fully closed (zero-based 7977; that run
stopped after 18 of its 84 targets). The remaining literal points are dealt
round-robin into chunk banks; each chunk is completed by the unchanged
coarse-row completer (complete_l86_full_rows.py) in its own process with
the same switches as the original run, except that it does not stop on a
new class. All 84 strict targets per fixed C_A; G_A and B stay free.

Per-support ledgers are off by default; per-target and per-anchor ledgers
are always written. A finished campaign only closes this finite frontier,
never all L86 PQCPs. SAT keeps the first verified model per target.

Interrupted? Set RUN_RESUME to the printed output directory and run again:
chunks whose own report shows every anchor exhausted with zero UNKNOWN are
kept, unfinished chunks are moved aside and rerun from their start.
"""
import argparse
from datetime import datetime
import gzip
import json
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
from time import monotonic, sleep

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.benchmark_l86_discovery import discovery_key, freeze_sources, verified_inventory
from experiments.extract_coarse_successor_anchors import file_sha
from solver.checkpoint import atomic_write_json
from solver.search_runner import append_verified_solution_if_new

RUN_PARENT = ROOT/'results/L86_coarse_neighbour_radius2_frontier_20261011'
RUN_START = 7977      # First literal point not fully closed by RUN_PARENT.
RUN_WORKERS = 10      # Concurrent completer processes (12-core machine).
RUN_CHUNKS = 48       # Resume granularity; about 5000 points per chunk.
RUN_SUPPORT_LEDGER = False
RUN_RESUME = None     # e.g. ROOT/'results/L86_coarse_frontier_shards_20261021_120000'
RUN_LIMIT = 0         # 0 = every remaining point; positive = only that many (smoke test).

COMPLETER_SWITCHES = ('--mode', 'coarse-row', '--radius', '0', '--anchor-side', '0', '--top', '0',
    '--two-adic', '--dihedral-support', '--norm-unsat-cache', '4096', '--norm-carrier-cache', '4096',
    '--unary-batch-free', '12', '--prebatch-dihedral', '--unary-linear-certificate',
    '--native-support-audit', '--no-stop-on-new')


def audit_closed_prefix(parent, points, start):
    """Credit only parent anchors 0..start-1 whose 84 targets were all UNSAT."""
    report = json.loads((parent/'coverage/report.json').read_text())
    if (not report['literal_coarse_points_input'] or report['mode'] != 'coarse-row' or report['radius'] != 0
            or report['unknown_target_fibers'] != 0 or report['support_bounds'] is not True):
        raise ValueError('parent must be a literal coarse-row run with zero UNKNOWN targets')
    if file_sha(parent/'coarse_points.json') != file_sha(parent/'coverage/input_coarse_points.json'):
        raise ValueError('parent completer input differs from the frontier point bank')
    credited = 0
    with gzip.open(parent/'coverage/completed_anchors.jsonl.gz', 'rt') as stream:
        for row in stream:
            row = json.loads(row)
            anchor = row['anchor']
            if anchor >= start:
                continue
            if (anchor != credited or row['target_fibers_decided'] != 84
                    or sum(row['statuses'].values()) != 84
                    or any(not name.startswith('UNSAT') for name in row['statuses'])
                    or list(row['anchor_key']) != list(points[anchor]['row'])):
                raise ValueError(f'parent anchor {anchor} is not a fully closed literal match')
            credited += 1
    if credited != start:
        raise ValueError(f'parent closed only {credited} contiguous anchors, fewer than start {start}')
    return dict(parent=str(parent), credited_points=start, parent_report_sha256=file_sha(parent/'coverage/report.json'),
                parent_anchor_ledger_sha256=file_sha(parent/'coverage/completed_anchors.jsonl.gz'),
                scope='Each credited point: parent ledger records all 84 strict targets UNSAT, literal row identical.')


def prepare(args, output):
    output.mkdir(parents=True, exist_ok=False)
    bank = json.loads((args.parent/'coarse_points.json').read_text())
    if bank.get('format') != 'literal_coarse_points_v1' or bank.get('planned_radius') != 0:
        raise ValueError('radius-zero literal coarse point bank required')
    points = bank['points']
    prefix = audit_closed_prefix(args.parent, points, args.start)
    stop = len(points) if not args.limit else min(len(points), args.start+args.limit)
    remaining = list(range(args.start, stop))
    chunks = min(args.chunks, len(remaining))
    (output/'chunks').mkdir()
    for c in range(chunks):
        indices = remaining[c::chunks]
        chunk = [dict(points[i], provenance=dict(points[i].get('provenance') or {}, frontier_index=i)) for i in indices]
        atomic_write_json(output/'chunks'/f'chunk_{c:02d}.json', dict(format='literal_coarse_points_v1', planned_radius=0,
            points=chunk, frontier_indices=indices, source_shell_radius=bank.get('source_shell_radius'),
            scope='Round-robin slice of the parent literal frontier; each row is a fixed point.'))
    snapshot = output/'source_snapshot'
    freeze_sources(snapshot)
    (snapshot/'experiments').mkdir(exist_ok=True)
    for path in (ROOT/'experiments').glob('*.py'):
        shutil.copy2(path, snapshot/'experiments'/path.name)
    raw = (ROOT/'86.txt').read_bytes() if (ROOT/'86.txt').exists() else b''
    verified_inventory(raw)
    (snapshot/'86.txt').write_bytes(raw)
    manifest = {str(p.relative_to(snapshot)): file_sha(p) for p in sorted(snapshot.rglob('*'))
                if p.is_file() and p.name != 'source_manifest.json'}
    atomic_write_json(snapshot/'source_manifest.json', manifest)
    plan = dict(parent=str(args.parent), parent_points_sha256=file_sha(args.parent/'coarse_points.json'),
        frontier_points=len(points), start=args.start, stop=stop, remaining_points=len(remaining), chunks=chunks,
        support_ledger=args.support_ledger, closed_prefix=prefix, completer_switches=list(COMPLETER_SWITCHES),
        initial_inventory_sha256=file_sha(snapshot/'86.txt'), created=datetime.now().isoformat(timespec='seconds'))
    atomic_write_json(output/'plan.json', plan)
    return plan


def chunk_report(output, c):
    path = output/'chunks'/f'chunk_{c:02d}'/'report.json'
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def chunk_done(report):
    return bool(report and report['all_planned_anchors_exhausted'] and report['unknown_target_fibers'] == 0)


def solution_pairs(path):
    if not path.exists():
        return []
    pairs = re.findall(r'^a=([01]+)\r?\nb=([01]+)\r?$', path.read_text(), re.MULTILINE)
    return [(tuple(map(int, a)), tuple(map(int, b))) for a, b in pairs]


def summarize(output, plan, elapsed):
    reports = [chunk_report(output, c) for c in range(plan['chunks'])]
    initial = verified_inventory((output/'source_snapshot/86.txt').read_bytes())
    new = {}
    for c in range(plan['chunks']):
        for a, b in solution_pairs(output/'chunks'/f'chunk_{c:02d}'/'86.txt'):
            key = discovery_key(a, b)  # Raises unless the full binary pair verifies.
            if key not in initial:
                new.setdefault(key, (a, b))
    done = [r for r in reports if chunk_done(r)]
    summary = dict(chunks=plan['chunks'], finished_chunks=len(done),
        remaining_points=plan['remaining_points'],
        finished_points=sum(r['literal_anchors'] for r in done),
        target_fibers_decided=sum(r['target_fibers_decided'] for r in reports if r),
        unknown_target_fibers=sum(r['unknown_target_fibers'] for r in reports if r),
        sat_target_fibers=sum(r['sat_target_fibers'] for r in reports if r),
        independently_verified_new_classes=len(new), campaign_elapsed=elapsed,
        closed_prefix=plan['closed_prefix'],
        complete=len(done) == plan['chunks'] and all(r['unknown_target_fibers'] == 0 for r in done),
        scope=('Radius-two literal frontier of the parent, points '+str(plan['start'])+'..'+str(plan['stop']-1)
               +' searched here, earlier points credited from the parent ledger. Every target of a finished point '
               'is UNSAT or SAT (first model kept). Finite region only; no global L86 nonexistence.'))
    return summary, new


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, default=RUN_PARENT)
    parser.add_argument('--start', type=int, default=RUN_START)
    parser.add_argument('--workers', type=int, default=RUN_WORKERS)
    parser.add_argument('--chunks', type=int, default=RUN_CHUNKS)
    parser.add_argument('--support-ledger', action=argparse.BooleanOptionalAction, default=RUN_SUPPORT_LEDGER)
    parser.add_argument('--resume', type=Path, default=RUN_RESUME)
    parser.add_argument('--limit', type=int, default=RUN_LIMIT)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--append-root', action=argparse.BooleanOptionalAction, default=True,
                        help='Append verified new classes to the project 86.txt at the end')
    args = parser.parse_args()
    if args.workers < 1 or args.chunks < 1 or args.start < 0 or args.limit < 0:
        parser.error('positive workers/chunks and nonnegative start/limit required')
    if args.resume:
        output = args.resume.resolve()
        plan = json.loads((output/'plan.json').read_text())
        manifest = json.loads((output/'source_snapshot/source_manifest.json').read_text())
        if any(file_sha(output/'source_snapshot'/name) != digest for name, digest in manifest.items()):
            raise RuntimeError('frozen source snapshot changed; start a new campaign instead')
    else:
        args.parent = args.parent.resolve()
        output = (args.output or ROOT/'results'/('L86_coarse_frontier_shards_'+datetime.now().strftime('%Y%m%d_%H%M%S'))).resolve()
        plan = prepare(args, output)
    print('output:', output, flush=True)
    print(f"points {plan['start']}..{plan['stop']-1} ({plan['remaining_points']}) in {plan['chunks']} chunks, "
          f"{args.workers} workers, support ledger {'on' if plan['support_ledger'] else 'off'}", flush=True)
    snapshot = output/'source_snapshot'
    (output/'logs').mkdir(exist_ok=True)
    pending = []
    for c in range(plan['chunks']):
        if chunk_done(chunk_report(output, c)):
            continue
        partial = output/'chunks'/f'chunk_{c:02d}'
        if partial.exists():
            partial.rename(partial.with_name(partial.name+'.partial_'+datetime.now().strftime('%Y%m%d_%H%M%S')))
        pending.append(c)
    running, failed, tick, last_print = {}, [], monotonic(), 0.0
    try:
        while pending or running:
            while pending and len(running) < args.workers:
                c = pending.pop(0)
                command = [sys.executable, str(snapshot/'experiments/complete_l86_full_rows.py'), *COMPLETER_SWITCHES,
                           '--coarse-anchors', str(output/'chunks'/f'chunk_{c:02d}.json'),
                           '--output', str(output/'chunks'/f'chunk_{c:02d}'),
                           '--support-ledger' if plan['support_ledger'] else '--no-support-ledger']
                log = (output/'logs'/f'chunk_{c:02d}.log').open('a')
                running[c] = (subprocess.Popen(command, cwd=snapshot, stdout=log, stderr=subprocess.STDOUT,
                                               start_new_session=True), log)
            for c, (process, log) in list(running.items()):
                if process.poll() is not None:
                    log.close()
                    del running[c]
                    if process.returncode != 0 or not chunk_done(chunk_report(output, c)):
                        failed.append(c)
                        print(f'chunk {c:02d} FAILED (exit {process.returncode}); see logs/chunk_{c:02d}.log', flush=True)
            if monotonic()-last_print >= 60 or not (pending or running):
                last_print = monotonic()
                summary, new = summarize(output, plan, monotonic()-tick)
                done_targets = summary['target_fibers_decided']
                fraction = done_targets/(84*plan['remaining_points'])
                eta = (monotonic()-tick)*(1-fraction)/fraction if fraction > 0 else float('inf')
                print(f"[{datetime.now():%H:%M:%S}] {fraction:6.2%} targets, chunks {summary['finished_chunks']}/{plan['chunks']}, "
                      f"UNKNOWN {summary['unknown_target_fibers']}, new classes {len(new)}, ETA {eta/60:.0f} min", flush=True)
            sleep(2)
    except KeyboardInterrupt:
        for process, log in running.values():
            process.send_signal(signal.SIGTERM)
        for process, log in running.values():
            process.wait()
            log.close()
        where = f'ROOT/{output.relative_to(ROOT).as_posix()!r}' if output.is_relative_to(ROOT) else repr(str(output))
        print(f'\nInterrupted. Set RUN_RESUME = {where} to continue.', flush=True)
        raise SystemExit(130)
    summary, new = summarize(output, plan, monotonic()-tick)
    summary['failed_chunks'] = failed
    appended = []
    if args.append_root:
        for key, (a, b) in sorted(new.items()):
            if append_verified_solution_if_new(86, a, b, ROOT):
                appended.append(dict(a=''.join(map(str, a)), b=''.join(map(str, b))))
    summary['appended_to_project_86_txt'] = appended
    atomic_write_json(output/'campaign_report.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in ('closed_prefix', 'scope')}, indent=1), flush=True)
    raise SystemExit(1 if failed else 0)


if __name__ == '__main__':
    main()
