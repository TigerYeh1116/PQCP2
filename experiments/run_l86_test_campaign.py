"""IDE Run: the 2026-10-21 L86 test campaign, every step in one press.

Step 1  finish the radius-two coarse frontier (run_l86_coarse_frontier_shards.py)
Step 2  weight-focused norm sampling, C_A weight 19..24 (search_norm_supports.py)
Step 3  test the pre-registered weight law on step 2 and summarize everything

Law under test, derived from the uniform ten-million run of 2026-10-09:
after the GF(2) support filter, a support pair whose C_A has weight w
admits C signs with probability c*M(w)^(-21/2), M(w)=w(w-1)+(43-w)(42-w).
The complementary G pair has the same M, so a full class needs both and
the per-pair full probability is about (c*M(w)^(-21/2))^2. All
predictions are written to plan.json before any search starts.

Every searched support and fixed point is still decided exactly; weight
focusing only changes which supports are proposed. A miss in this finite
campaign never excludes any L86 PQCP.

Interrupted? Set RUN_RESUME to the printed campaign directory and run again.
"""
import argparse
from datetime import datetime
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import shutil
import signal
import subprocess
import sys
from time import monotonic, sleep

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.benchmark_l86_discovery import discovery_key, verified_inventory
from solver.checkpoint import atomic_write_json
from solver.search_runner import append_verified_solution_if_new

RUN_FRONTIER = False            # Step 1 finished in results/L86_test_campaign_20261021_123359; rerunning repeats it.
RUN_SAMPLER = True              # Step 2.
RUN_SAMPLER_TRIALS = 80_000_000  # About 8400 trials/s with 10 workers: roughly 2.6 hours, 10 GB trace.
RUN_SAMPLER_WEIGHTS = '19-24'
RUN_WORKERS = 10
RUN_RESUME = None               # e.g. ROOT/'results/L86_test_campaign_20261021_130000'

# Calibration: C-SAT supports / sampled supports per C_A weight in
# results/L86_phase_grid_default_original_ten_million_20261009 (uniform w=9..35).
CALIBRATION_SOURCE = 'results/L86_phase_grid_default_original_ten_million_20261009/literal_trials.jsonl.gz'
CALIBRATION_POSITIVE = {9: 0, 10: 2, 11: 1, 12: 6, 13: 7, 14: 9, 15: 13, 16: 13, 17: 26, 18: 23, 19: 30, 20: 32,
    21: 38, 22: 36, 23: 36, 24: 23, 25: 34, 26: 27, 27: 19, 28: 17, 29: 9, 30: 9, 31: 4, 32: 7, 33: 1, 34: 3, 35: 2}
CALIBRATION_TRIALS = {w: 370371 if w in (9, 11, 13, 15, 17, 19, 21, 23, 25, 27) else 370370 for w in range(9, 36)}
UNIFORM_BASELINE = dict(trials=10_000_000, workers=8, elapsed=1438.9047640000063, coarse_witnesses=427)


def M(w):
    return w*(w-1)+(43-w)*(42-w)


def law_constant():
    """Poisson maximum likelihood scale with the shape M(w)^(-21/2) held fixed."""
    return sum(CALIBRATION_POSITIVE.values())/sum(n*M(w)**-10.5 for w, n in CALIBRATION_TRIALS.items())


def predicted_c_rate(w):
    return law_constant()*M(w)**-10.5


def weights_of(text):
    wanted = set()
    for part in text.split(','):
        lo, _, hi = part.strip().partition('-')
        wanted.update(range(int(lo), int(hi or lo)+1))
    return sorted(wanted)


def make_plan(args, campaign):
    weights = weights_of(args.sampler_weights)
    per_weight = args.sampler_trials/len(weights)
    rates = {w: predicted_c_rate(w) for w in weights}
    uniform_rate = sum(predicted_c_rate(w)*n for w, n in CALIBRATION_TRIALS.items())/sum(CALIBRATION_TRIALS.values())
    return dict(created=datetime.now().isoformat(timespec='seconds'), workers=args.workers, frontier_limit=args.frontier_limit,
        frontier=args.frontier, sampler=args.sampler, sampler_trials=args.sampler_trials, sampler_weights=weights,
        initial_inventory_sha256=hashlib.sha256((campaign/'initial_inventory.txt').read_bytes()).hexdigest(),
        law=dict(statement='P(C signs | GF(2)-compatible support pair, C_A weight w) = c*M(w)^(-21/2), M(w)=w(w-1)+(43-w)(42-w)',
                 calibration_source=CALIBRATION_SOURCE, c=law_constant()),
        predictions=dict(
            c_rate_per_weight={str(w): rates[w] for w in weights},
            expected_c_witnesses=sum(per_weight*rates[w] for w in weights),
            focused_over_uniform_c_rate=(sum(rates.values())/len(weights))/uniform_rate,
            expected_full_new_classes_from_sampler=sum(per_weight*rates[w]**2 for w in weights),
            note='Full-class expectation assumes C and G feasibility independent given w; calibrated on about one historical hit.'))


def stream_sampler_counts(directory):
    """Per-weight sampled supports and C-SAT supports from one literal trace."""
    total, positive = {}, {}
    pattern = re.compile(r'"contents":\[\[(\d+),\d+,(\d+)\]')
    try:
        with gzip.open(directory/'literal_trials.jsonl.gz', 'rt') as stream:
            for line in stream:
                match = pattern.search(line)
                if not match or not line.endswith('\n'):
                    continue  # An interrupted run may end in a partial line; it is not credited.
                w = int(match.group(1))+int(match.group(2))
                total[w] = total.get(w, 0)+1
                if 'SAT_COMPRESSED' in line:
                    coarse = ((json.loads(line).get('result') or {}).get('attempts') or {}).get('coarse') or {}
                    if coarse.get('status') == 'SAT_COMPRESSED':
                        positive[w] = positive.get(w, 0)+1
    except EOFError:
        pass  # Truncated gzip after an interrupt: every complete line before it was counted.
    return total, positive


def poisson_two_sided(observed, expected):
    """Exact two-sided Poisson p-value (doubling the smaller tail), log-space terms."""
    if expected <= 0:
        return 1.0 if observed == 0 else 0.0
    def pmf(k):
        return math.exp(k*math.log(expected)-expected-math.lgamma(k+1))
    lower = sum(pmf(k) for k in range(observed+1))
    upper = 1-sum(pmf(k) for k in range(observed)) if observed > 0 else 1.0
    return min(1.0, 2*min(lower, upper))


def solution_pairs(path):
    if not path.exists():
        return []
    pairs = re.findall(r'^a=([01]+)\r?\nb=([01]+)\r?$', path.read_text(), re.MULTILINE)
    return [(tuple(map(int, a)), tuple(map(int, b))) for a, b in pairs]


def stop_group(process):
    """Forward an interrupt to the child's whole process group, then make sure it ends."""
    for signal_number, grace in ((signal.SIGINT, 60), (signal.SIGTERM, 30), (signal.SIGKILL, None)):
        try:
            os.killpg(process.pid, signal_number)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=grace)
            return
        except subprocess.TimeoutExpired:
            continue


def run_group(command, log, poll=None):
    """Run a step in its own process group so an interrupt always reaches every worker."""
    process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT if log else None,
                               start_new_session=True)
    try:
        while process.poll() is None:
            sleep(2)
            if poll:
                poll()
    except KeyboardInterrupt:
        stop_group(process)
        raise
    return process.returncode


def run_frontier(args, campaign):
    target = campaign/'frontier'
    report = target/'campaign_report.json'
    if report.exists() and json.loads(report.read_text())['complete']:
        print('step 1 already complete', flush=True)
        return
    command = [sys.executable, str(ROOT/'experiments/run_l86_coarse_frontier_shards.py'), '--workers', str(args.workers),
               '--limit', str(args.frontier_limit)]
    command += ['--resume', str(target)] if (target/'plan.json').exists() else ['--output', str(target)]
    if target.exists() and not (target/'plan.json').exists():
        shutil.rmtree(target)  # Interrupted before its plan existed: nothing was searched there.
    print('\n=== step 1: radius-two coarse frontier ===', flush=True)
    code = run_group(command, None)
    if code:
        raise SystemExit(f'step 1 stopped with exit code {code}; rerun with RUN_RESUME to continue')


def sampler_runs(campaign):
    return sorted(p for p in campaign.glob('sampler_*') if (p/'report.json').exists())


def run_sampler(args, campaign):
    done = sum(json.loads((p/'report.json').read_text())['tested'] for p in sampler_runs(campaign))
    remaining = args.sampler_trials-done
    if remaining <= 0:
        print('step 2 already complete', flush=True)
        return
    index = len([p for p in campaign.glob('sampler_*') if p.is_dir()])+1
    output = campaign/f'sampler_{index:02d}'
    seed = secrets.randbelow(2**31)
    command = [sys.executable, str(ROOT/'experiments/search_norm_supports.py'), '--output', str(output),
               '--trials', str(remaining), '--workers', str(min(args.workers, 16)), '--seed', str(seed),
               '--gzip-trace', '--projection-groups', '--weight-prefix', '--c-weights', args.sampler_weights]
    print(f'\n=== step 2: weight-focused sampling, w={args.sampler_weights}, {remaining} trials, seed {seed} ===', flush=True)
    tick, last = monotonic(), [0.0]

    def progress():
        if monotonic()-last[0] < 60:
            return
        last[0] = monotonic()
        try:
            report = json.loads((output/'report.json').read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return
        speed = report['tested']/max(monotonic()-tick, 1e-9)
        eta = (remaining-report['tested'])/speed if speed else float('inf')
        print(f"[{datetime.now():%H:%M:%S}] {report['tested']/remaining:6.2%} trials, "
              f"C witnesses {report['coarse_witnesses']}, new classes {report['independently_verified_new_classes']}, "
              f"{speed:.0f} trials/s, ETA {eta/60:.0f} min", flush=True)

    with (campaign/f'sampler_{index:02d}.log').open('w') as log:
        code = run_group(command, log, progress)
    if code:
        raise SystemExit(f'step 2 stopped with exit code {code}; see {log.name}')


def analyse(campaign, plan):
    print('\n=== step 3: law test and summary ===', flush=True)
    summary = dict(plan=plan)
    frontier = campaign/'frontier'/'campaign_report.json'
    if frontier.exists():
        summary['frontier'] = {k: v for k, v in json.loads(frontier.read_text()).items() if k != 'closed_prefix'}
    runs = sampler_runs(campaign)
    total, positive, reports = {}, {}, []
    for run in runs:
        reports.append(json.loads((run/'report.json').read_text()))
        t, p = stream_sampler_counts(run)
        for w in t:
            total[w] = total.get(w, 0)+t[w]
            positive[w] = positive.get(w, 0)+p.get(w, 0)
    if total:
        rows = []
        for w in sorted(total):
            expected = total[w]*predicted_c_rate(w)
            rows.append(dict(w=w, trials=total[w], c_sat=positive.get(w, 0), predicted=expected,
                             p_value=poisson_two_sided(positive.get(w, 0), expected)))
        observed, expected = sum(r['c_sat'] for r in rows), sum(r['predicted'] for r in rows)
        trials = sum(total.values())
        cpu = sum(r['elapsed']*r['workers'] for r in reports)
        baseline_rate = UNIFORM_BASELINE['coarse_witnesses']/UNIFORM_BASELINE['trials']
        baseline_cpu = UNIFORM_BASELINE['coarse_witnesses']/(UNIFORM_BASELINE['elapsed']*UNIFORM_BASELINE['workers'])
        summary['law_test'] = dict(per_weight=rows, observed_c_sat=observed, predicted_c_sat=expected,
            total_p_value=poisson_two_sided(observed, expected), trials=trials,
            c_sat_rate=observed/trials, uniform_baseline_c_sat_rate=baseline_rate,
            focused_over_uniform_per_trial=(observed/trials)/baseline_rate,
            c_sat_per_cpu_second=observed/cpu if cpu else None, uniform_baseline_per_cpu_second=baseline_cpu,
            focused_over_uniform_per_cpu=(observed/cpu)/baseline_cpu if cpu else None,
            scope='Wall-clock workers*elapsed as CPU proxy; both runs on this machine but different dates and worker counts.')
        print(f'{"w":>3} {"trials":>10} {"C-SAT":>6} {"predicted":>9} {"p":>6}')
        for r in rows:
            print(f"{r['w']:>3} {r['trials']:>10} {r['c_sat']:>6} {r['predicted']:>9.1f} {r['p_value']:>6.3f}")
        law = summary['law_test']
        print(f"total C-SAT {observed} vs predicted {expected:.1f} (p={law['total_p_value']:.3f}); "
              f"focused/uniform C-SAT per trial {law['focused_over_uniform_per_trial']:.2f}x, "
              f"per CPU {law['focused_over_uniform_per_cpu']:.2f}x", flush=True)
    initial = verified_inventory((campaign/'initial_inventory.txt').read_bytes())
    new, appended = {}, []
    for run in runs:
        for a, b in solution_pairs(run/'86.txt'):
            key = discovery_key(a, b)  # Raises unless the full binary pair verifies.
            if key not in initial:
                new.setdefault(key, (a, b))
    for key, (a, b) in sorted(new.items()):
        if append_verified_solution_if_new(86, a, b, ROOT):
            appended.append(dict(a=''.join(map(str, a)), b=''.join(map(str, b))))
    # Counts come from the literal traces: an interrupted run's trace may hold complete trials
    # written after its last periodic report.
    summary['sampler'] = dict(runs=[str(r) for r in runs], trials=sum(total.values()),
        c_witnesses=sum(positive.values()), unknown_in_reports=sum(r['statuses'].get('UNKNOWN', 0) for r in reports),
        independently_verified_new_classes=len(new), appended_to_project_86_txt=appended)
    atomic_write_json(campaign/'campaign_report.json', summary)
    frontier_new = summary.get('frontier', {}).get('independently_verified_new_classes', 0)
    print(f"frontier: {summary.get('frontier', {}).get('finished_points', 0)} points finished, "
          f"UNKNOWN {summary.get('frontier', {}).get('unknown_target_fibers', 0)}, new classes {frontier_new}")
    print(f"sampler: {summary['sampler']['trials']} trials, {summary['sampler']['c_witnesses']} C witnesses, "
          f"new classes {len(new)} (predicted {plan['predictions']['expected_full_new_classes_from_sampler']:.2f})")
    print('report:', campaign/'campaign_report.json', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frontier', action=argparse.BooleanOptionalAction, default=RUN_FRONTIER)
    parser.add_argument('--sampler', action=argparse.BooleanOptionalAction, default=RUN_SAMPLER)
    parser.add_argument('--sampler-trials', type=int, default=RUN_SAMPLER_TRIALS)
    parser.add_argument('--sampler-weights', default=RUN_SAMPLER_WEIGHTS)
    parser.add_argument('--workers', type=int, default=RUN_WORKERS)
    parser.add_argument('--resume', type=Path, default=RUN_RESUME)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--frontier-limit', type=int, default=0, help='Smoke tests only: first N remaining frontier points')
    args = parser.parse_args()
    if args.sampler_trials < 1 or args.workers < 1:
        parser.error('positive trials and workers required')
    if args.resume:
        campaign = args.resume.resolve()
        plan = json.loads((campaign/'plan.json').read_text())
        args.frontier, args.sampler = plan['frontier'], plan['sampler']
        args.sampler_trials, args.workers = plan['sampler_trials'], plan['workers']
        args.frontier_limit = plan.get('frontier_limit', 0)
        args.sampler_weights = ','.join(map(str, plan['sampler_weights']))
    else:
        campaign = (args.output or ROOT/'results'/('L86_test_campaign_'+datetime.now().strftime('%Y%m%d_%H%M%S'))).resolve()
        campaign.mkdir(parents=True, exist_ok=False)
        raw = (ROOT/'86.txt').read_bytes()
        verified_inventory(raw)
        (campaign/'initial_inventory.txt').write_bytes(raw)
        plan = make_plan(args, campaign)
        atomic_write_json(campaign/'plan.json', plan)
    print('campaign:', campaign, flush=True)
    p = plan['predictions']
    print(f"pre-registered: {p['expected_c_witnesses']:.0f} C witnesses expected from the sampler "
          f"({p['focused_over_uniform_c_rate']:.2f}x the uniform per-trial rate), "
          f"{p['expected_full_new_classes_from_sampler']:.2f} full new classes expected", flush=True)
    try:
        if args.frontier:
            run_frontier(args, campaign)
        if args.sampler:
            run_sampler(args, campaign)
    except KeyboardInterrupt:
        where = f'ROOT/{campaign.relative_to(ROOT).as_posix()!r}' if campaign.is_relative_to(ROOT) else repr(str(campaign))
        print(f'\nInterrupted. Set RUN_RESUME = {where} to continue.', flush=True)
        raise SystemExit(130)
    analyse(campaign, plan)


if __name__ == '__main__':
    main()
