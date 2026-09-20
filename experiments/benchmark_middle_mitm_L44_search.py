"""Cold-start L44 search comparison, with frozen full-equivalence novelty.

Production baseline is main.py's MPS + C + Z3 path, with unchanged defaults.
MITM is the experimental fresh coarse FKM -> middle MITM -> binary Z3 path.
Each subprocess has its own copy of the SAME initial 44.txt, no seed bank or
completion checkpoint. Existing solutions are read ONLY by the result writer.
No results are added to the real 44.txt until all measurements have finished.

SA restarts and coarse jobs are different work units. Never compare their
counts as if they were the same. Discovery times include imports, FKM setup,
verification and persistence. Deadline misses are right-censored, not successes.
Shutdown work is reported separately; discoveries after the deadline do not
count towards throughput. Parallel scheduling makes wall-clock runs stochastic
even with a deterministic seed set.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import signal
import statistics
import subprocess
import sys
import threading
from time import CLOCK_MONOTONIC, clock_gettime

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from solver.checkpoint import atomic_write_json


def perf_counter():
    """Shared OS clock: Python 3.9/macOS perf_counter has process-local origins.

    The orchestrator passes timestamps to fresh Python processes. Explicit
    clock_gettime(CLOCK_MONOTONIC), unlike the old Python wrapper, preserves
    the same epoch across them, without relying on adjustable calendar time.
    """
    return clock_gettime(CLOCK_MONOTONIC)


def summarize(rows):
    """Report censored first hits and exposure/discovery rates separately."""
    for row in rows:
        first = row['first_new_seconds']
        if first is not None and not 0 <= first <= row['budget_seconds']:
            raise ValueError('invalid first-hit timestamp')
        if row.get('elapsed', 0) < 0:
            raise ValueError('negative duration')
    result = {}
    for method in sorted({r['method'] for r in rows}):
        group = [r for r in rows if r['method'] == method]
        hits = [r['first_new_seconds'] for r in group if r['first_new_seconds'] is not None]
        count = sum(r['new_count'] for r in group)
        exposure = sum(r['budget_seconds'] for r in group)
        work = sum(r['work_count_at_shutdown'] for r in group)
        result[method] = dict(
            runs=len(group), successful_runs=len(hits), censored_runs=len(group)-len(hits),
            new_count=count, exposure_seconds=exposure,
            seconds_per_new=exposure/count if count else None,
            mean_first_new_reached_only=statistics.mean(hits) if hits else None,
            median_first_new_reached_only=statistics.median(hits) if hits else None,
            work_unit=group[0]['work_unit'], work_count_at_shutdown=work,
            work_per_new_including_shutdown=work/count if count else None,
            shutdown_seconds=sum(r['shutdown_seconds'] for r in group),
        )
    return result


class Observation:
    """Observe existing atomic writes without changing search decisions."""

    def __init__(self, directory, started, seconds):
        self.directory, self.started, self.seconds = directory, started, seconds
        self.lock = threading.RLock()
        self.events = []
        self.work_count = 0
        self.work_at_deadline_lower_bound = 0
        self.gpu_cycles = 0
        self.work_unit = 'SA restarts'
        self.stream = (directory/'discoveries.jsonl').open('x', encoding='utf-8')

    def install(self):
        import solver.search_runner as runner
        original = runner.append_verified_solution_if_new

        def observed(length, a, b, project_root):
            # Preserve the production writer and serialize only its observation.
            with self.lock:
                new = original(length, a, b, project_root)
                if new:
                    from solver.verifier import verify_pqcp
                    check = verify_pqcp(a, b)
                    if not check.is_valid:
                        raise RuntimeError('writer accepted an invalid candidate')
                    elapsed = perf_counter()-self.started
                    if elapsed < 0:
                        raise RuntimeError('clock epochs do not match')
                    event = dict(elapsed=elapsed, within_budget=elapsed <= self.seconds,
                                 A=list(a), B=list(b), profile=list(check.profile),
                                 work_last_observed=self.work_count, work_unit=self.work_unit,
                                 verified=True)
                    self.events.append(event)
                    self.stream.write(json.dumps(event)+'\n')
                    self.stream.flush()
                return new

        # Already-imported aliases and future imports both see the same wrapper.
        for name, module in list(sys.modules.items()):
            if name == 'main' or name.startswith('solver.'):
                if getattr(module, 'append_verified_solution_if_new', None) is original:
                    setattr(module, 'append_verified_solution_if_new', observed)


class ObservedOutput:
    """Keep unmodified output and the most recently published cumulative counter."""

    def __init__(self, stream, observation):
        self.stream, self.observation = stream, observation
        self.pending = ''

    def write(self, value):
        with self.observation.lock:
            self.stream.write(value)
            self.pending += value
            while '\n' in self.pending:
                line, self.pending = self.pending.split('\n', 1)
                match = re.search(r'搜尋統計：restart=(\d+),', line)
                if match:
                    self.observation.work_count = max(self.observation.work_count, int(match[1]))
                    if perf_counter()-self.observation.started <= self.observation.seconds:
                        self.observation.work_at_deadline_lower_bound = self.observation.work_count
                cycles = re.search(r'\[MPS\] cycle=(\d+)', line)
                if cycles:
                    self.observation.gpu_cycles = max(self.observation.gpu_cycles, int(cycles[1]))
        return len(value)

    def flush(self):
        self.stream.flush()


def run_mitm(seed, observation):
    from experiments.search_middle_mitm_L44 import coarse_jobs
    from experiments.benchmark_middle_mitm import bounded_run
    from solver.compressed_equivalence import compressed_job_equivalence_key
    from solver.frequency_constraints import quarter_frequency_feasible
    from solver.search_runner import append_verified_solution_if_new
    from solver.target_profiles import canonical_target_content_profiles
    observation.work_unit = 'coarse lift jobs (NOT SA restarts)'
    rng, seen = random.Random(seed), set()
    profiles = [p for p in canonical_target_content_profiles(44) if quarter_frequency_feasible(p)]
    deadline = observation.started + observation.seconds
    with (observation.directory/'attempts.jsonl').open('x') as stream:
        for profile in profiles:
            pairs = coarse_jobs(profile)
            print('Fresh coarse matches:', len(pairs), flush=True)
            rng.shuffle(pairs)
            for a, b in pairs:
                remaining = deadline-perf_counter()
                if remaining <= 0:
                    return
                key = compressed_job_equivalence_key(a, b, profile, 4)
                if key in seen:
                    continue
                seen.add(key)
                job = dict(compressed_a=a, compressed_b=b, target=asdict(profile))
                job['id'] = hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest()[:20]
                observation.work_count += 1
                # Exactly the prior successful experiment's settings, not tuned.
                result = bounded_run(job, 'mitm', min(12.0, remaining), 250000, 1000)
                result.update(job_spec=job, total_elapsed=perf_counter()-observation.started,
                              coarse_attempt=observation.work_count)
                stream.write(json.dumps(result)+'\n')
                stream.flush()
                print('job', observation.work_count, result['status'], flush=True)
                if result['status'] == 'ERROR':
                    raise RuntimeError(result['reason'])
                if result['status'] == 'SAT':
                    append_verified_solution_if_new(44, tuple(result['A']), tuple(result['B']), Path('.'))
        raise RuntimeError('candidate stream exhausted before benchmark deadline')


def worker(args):
    directory = args.worker.resolve()
    os.chdir(directory)
    observation = Observation(directory, args.started, args.seconds)
    original_stdout = sys.stdout
    sys.stdout = ObservedOutput(original_stdout, observation)
    error = None
    try:
        if args.method == 'production_mps':
            import main as production
            import torch
            if not torch.backends.mps.is_available():
                raise RuntimeError('MPS unavailable; refusing a CPU substitute baseline')
            cli = production.parse_args(['--L', '44', '--seed', str(args.seed),
                                         '--torch-device', 'mps', '--z3'])
            atomic_write_json(directory/'parameters.json', vars(cli))
            observation.install()
            production._run_torch_backend(cli, max(.001, args.seconds-(perf_counter()-args.started)))
        else:
            observation.install()
            run_mitm(args.seed, observation)
    except KeyboardInterrupt:
        pass
    except BaseException as exc:
        import traceback
        error = traceback.format_exc()
        print(error, flush=True)
    finally:
        elapsed = perf_counter()-args.started
        eligible = [e for e in observation.events if e['within_budget']]
        atomic_write_json(directory/'result.json', dict(
            method=args.method, seed=args.seed, budget_seconds=args.seconds,
            elapsed=elapsed, shutdown_seconds=max(0, elapsed-args.seconds),
            new_count=len(eligible), total_saved=len(observation.events),
            first_new_seconds=eligible[0]['elapsed'] if eligible else None,
            work_count_at_shutdown=observation.work_count, work_unit=observation.work_unit,
            work_at_deadline_lower_bound=observation.work_at_deadline_lower_bound,
            gpu_cycles=observation.gpu_cycles,
            error=error))
        observation.stream.close()
        sys.stdout = original_stdout
    if error:
        raise RuntimeError('benchmark worker failed; see worker.log')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--seconds', type=float, default=60)
    parser.add_argument('--seeds', type=int, nargs='+', default=[123,456,789,1024,2026])
    parser.add_argument('--methods', nargs='+', choices=['production_mps','mitm'],
                        default=['production_mps','mitm'])
    parser.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--method', help=argparse.SUPPRESS)
    parser.add_argument('--seed', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--started', type=float, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error('seconds must be positive')
    if args.worker:
        worker(args)
        return
    if args.output_dir is None:
        parser.error('--output-dir is required')
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    snapshot = out/'44.before.txt'
    shutil.copy2(ROOT/'44.txt', snapshot)
    atomic_write_json(out/'manifest.json', dict(
        L=44, seeds=args.seeds, seconds=args.seconds, methods=args.methods,
        snapshot_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest(),
        timing='cold process start through verified atomic write',
        novelty='same frozen initial full equivalence classes; unique within each run',
        order='alternating method order by seed; sequential runs'))
    rows = []
    for index, seed in enumerate(args.seeds):
        for method in (args.methods if index % 2 == 0 else list(reversed(args.methods))):
            directory = out/'{}_seed{}'.format(method,seed)
            directory.mkdir()
            shutil.copy2(snapshot, directory/'44.txt')
            started = perf_counter()
            command = [sys.executable, str(Path(__file__).resolve()), '--worker', str(directory),
                       '--method', method, '--seed', str(seed), '--seconds', str(args.seconds),
                       '--started', str(started)]
            print('START', method, seed, flush=True)
            with (directory/'worker.log').open('x') as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=True)
                try:
                    process.wait(timeout=args.seconds+2)
                except subprocess.TimeoutExpired:
                    # Signal only the orchestration process first for safe cleanup.
                    process.send_signal(signal.SIGINT)
                    try:
                        process.wait(timeout=90)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGTERM)
                        process.wait(timeout=10)
                        raise RuntimeError('worker did not stop; preserved all artifacts')
            result_path = directory/'result.json'
            if process.returncode != 0 or not result_path.exists():
                raise RuntimeError('worker failed: {}'.format(directory))
            row = json.loads(result_path.read_text())
            if row['error']:
                raise RuntimeError(row['error'])
            rows.append(row)
            atomic_write_json(out/'runs.json', rows)
            atomic_write_json(out/'summary.json', summarize(rows))
            print('END', method, seed, 'new=', row['new_count'],
                  'first=', row['first_new_seconds'], 'work=', row['work_count_at_shutdown'], flush=True)
    print(json.dumps(summarize(rows), indent=2), flush=True)


if __name__ == '__main__':
    main()
