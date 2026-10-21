"""IDE Run: measure the C-sign feasibility law at L=94 (d=47) against an L=86 control.

Both lengths use identical code: a uniform C_A support of weight w, a uniform
GF(2)-compatible C_B support (solver/gf2_support_family.py) and the generic
exact C sign solver. A C-SAT support pair is completed exactly (G signs too);
a full L94 pair would be verified and appended to 94.txt.

Output: per-weight C-SAT rates, the fitted law c*M(w)^(-n/2) with
M(w)=w(w-1)+(d-w)(d-1-w), n=(d-1)/2, and the implied CPU cost per expected
new class. Sampling only; a miss excludes nothing.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import json
import multiprocessing
from pathlib import Path
import random
import secrets
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from solver.checkpoint import atomic_write_json

RUN_PLAN = {86: (15, 18, 21, 22, 25, 28), 94: (17, 20, 23, 24, 27, 30)}  # C_A weights per length.
RUN_TRIALS = {86: 200_000, 94: 500_000}                                   # Per weight.
RUN_WORKERS = 10
RUN_MAX_ASSIGNMENTS = 2**25  # The d=47 sign space is 2^23; leave room so no budget UNKNOWN.

_CHOICES = {}


def _choices(L, w):
    key = (L, w)
    if key not in _CHOICES:
        from solver.target_profiles import canonical_target_content_profiles
        from solver.torch_compressed_search import search_paired_contents
        _CHOICES[key] = [(t, c) for t in canonical_target_content_profiles(L)
                         for c in search_paired_contents(t) if c[0][0]+c[0][2] == w]
    return _CHOICES[key]


def run_block(job):
    """Trials [start, stop) of one (L, w); returns counts and any witnesses."""
    L, w, seed, start, stop, budget = job
    from solver.compressed_repair import _validated_reference
    from solver.gf2_support_family import gf2_support_ring, sample_support
    from solver.pqcp_support_solver import complete_pqcp_supports
    from solver.ternary_signs_native import prepare_sign_enumeration, solve_sign_system
    from solver.ternary_support_search import support_sign_system
    d = L//2
    ring = gf2_support_ring(d)
    choices = _choices(L, w)
    counts, witnesses, tick = Counter(), [], perf_counter()
    for n in range(start, stop):
        rng = random.Random(seed*1_000_003+n)
        target, contents = choices[n % len(choices)]
        a = tuple([-1]*contents[0][0]+[0]*contents[0][1]+[1]*contents[0][2])
        b = tuple([-1]*contents[1][0]+[0]*contents[1][1]+[1]*contents[1][2])
        a, b, d, wanted = _validated_reference(a, b, target)
        u = sum(1 << j for j in rng.sample(range(d), w))
        v = sample_support(ring, u, min(target.k, d-target.k), d-w, rng)
        counts['trials'] += 1
        if v is None:
            counts['no_weight_match'] += 1
            continue
        masks, negative = (u, v), (a.count(-1), b.count(-1))
        system = support_sign_system(masks, d, wanted, negative)
        if not system.consistent:
            counts['c_unsat'] += 1
            continue
        out = solve_sign_system(system, masks, d, wanted, negative, budget,
                                prepared=prepare_sign_enumeration(system, masks, negative))
        status = out['status']
        counts[{'SAT_COMPRESSED': 'c_sat', 'UNKNOWN': 'unknown'}.get(status, 'c_unsat')] += 1
        if status != 'SAT_COMPRESSED':
            continue
        full = complete_pqcp_supports(a, b, target, masks, max_assignments=budget,
                                      native_preprocessing=False, binary_content_pruning=False)
        counts['full_sat' if full['verified'] else 'g_unsat' if full['status'] != 'UNKNOWN' else 'g_unknown'] += 1
        witnesses.append(dict(L=L, w=w, trial=n, target=vars(target), contents=contents, masks=list(masks),
                              full_status=full['status'], words=full['words']))
    counts['seconds'] = perf_counter()-tick
    return L, w, dict(counts), witnesses


def M(d, w):
    return w*(w-1)+(d-w)*(d-1-w)


def analyse(rows):
    """Fit the law per length with its shape fixed; derive cost per expected new class."""
    out = {}
    for L in sorted({r['L'] for r in rows}):
        d, n = L//2, (L//2-1)//2
        mine = [r for r in rows if r['L'] == L]
        eligible = sum(r['trials']-r.get('no_weight_match', 0) for r in mine)
        hits = sum(r.get('c_sat', 0) for r in mine)
        scale = hits/sum((r['trials']-r.get('no_weight_match', 0))*M(d, r['w'])**(-n/2) for r in mine)
        cost = sum(r['seconds'] for r in mine)/sum(r['trials'] for r in mine)
        best = min(range(1, d), key=lambda w: M(d, w))
        p_best = scale*M(d, best)**(-n/2)
        out[str(L)] = dict(d=d, constraints=n, eligible_trials=eligible, c_sat=hits, law_scale=scale,
            best_weight=best, predicted_c_rate_at_best=p_best,
            cpu_seconds_per_trial=cost, cpu_hours_per_expected_class=cost/p_best**2/3600 if p_best else None,
            per_weight=[dict(w=r['w'], trials=r['trials'], no_weight_match=r.get('no_weight_match', 0),
                             c_sat=r.get('c_sat', 0), unknown=r.get('unknown', 0), full_sat=r.get('full_sat', 0),
                             rate=r.get('c_sat', 0)/max(1, r['trials']-r.get('no_weight_match', 0)),
                             law=scale*M(d, r['w'])**(-n/2)) for r in sorted(mine, key=lambda r: r['w'])])
    if '86' in out and '94' in out and out['86']['c_sat'] and out['94']['c_sat']:
        out['ratio_94_over_86'] = dict(
            c_rate_at_best=out['94']['predicted_c_rate_at_best']/out['86']['predicted_c_rate_at_best'],
            cpu_per_trial=out['94']['cpu_seconds_per_trial']/out['86']['cpu_seconds_per_trial'],
            cpu_per_expected_class=out['94']['cpu_hours_per_expected_class']/out['86']['cpu_hours_per_expected_class'])
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=RUN_WORKERS)
    parser.add_argument('--scale', type=float, default=1.0, help='Multiply every per-weight trial count (smoke tests)')
    parser.add_argument('--seed', type=int, default=secrets.randbelow(2**31))
    parser.add_argument('--output', type=Path, default=ROOT/'results'/('L94_pilot_c_rates_'+datetime.now().strftime('%Y%m%d_%H%M%S')))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    block = 2000
    jobs = []
    for L, weights in RUN_PLAN.items():
        total = max(1, int(RUN_TRIALS[L]*args.scale))
        for w in weights:
            jobs += [(L, w, args.seed, s, min(s+block, total), RUN_MAX_ASSIGNMENTS) for s in range(0, total, block)]
    random.Random(args.seed).shuffle(jobs)  # Interleave lengths/weights so partial runs stay balanced.
    print('output:', args.output, ' seed', args.seed, ' blocks', len(jobs), flush=True)
    totals, witnesses, tick, last = {}, [], perf_counter(), 0.0
    found = []
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        for done, (L, w, counts, rows) in enumerate(pool.map(run_block, jobs, chunksize=1), 1):
            row = totals.setdefault((L, w), Counter(L=0, w=0))
            row.update(counts)
            witnesses += rows
            for r in rows:
                if r['words'] is not None and L == 94:
                    from solver.search_runner import append_verified_solution_if_new
                    if append_verified_solution_if_new(94, *r['words'], ROOT):
                        found.append(r)
                        print('*** NEW VERIFIED L94 PQCP appended to 94.txt ***', flush=True)
            if perf_counter()-last >= 60 or done == len(jobs):
                last = perf_counter()
                rows_now = [dict(v, L=k[0], w=k[1]) for k, v in totals.items()]
                report = dict(seed=args.seed, blocks_done=done, blocks=len(jobs), elapsed=perf_counter()-tick,
                              complete=done == len(jobs), analysis=analyse(rows_now), new_l94=found,
                              scope='Sampled supports only; exact per-support decisions; no exclusion claimed.')
                atomic_write_json(args.output/'report.json', report)
                atomic_write_json(args.output/'witnesses.json', witnesses)
                a = report['analysis']
                line = ' '.join('L{}: {} C-SAT'.format(L, a[L]['c_sat']) for L in ('86', '94') if L in a)
                print('[{:%H:%M:%S}] {}/{} blocks, {}'.format(datetime.now(), done, len(jobs), line), flush=True)
    a = report['analysis']
    for L in ('86', '94'):
        if L not in a:
            continue
        x = a[L]
        print('\nL={} (d={}, {} constraints): C-SAT {} / {} trials; {:.2f} ms CPU per trial'.format(
            L, x['d'], x['constraints'], x['c_sat'], x['eligible_trials'], 1e3*x['cpu_seconds_per_trial']))
        for r in x['per_weight']:
            print('  w={:>2}  C-SAT {:>4}  rate {:.2e}  law {:.2e}  unknown {}'.format(
                r['w'], r['c_sat'], r['rate'], r['law'], r['unknown']))
        if x['cpu_hours_per_expected_class'] is not None:
            print('  best w={}: C rate {:.2e}; about {:.0f} CPU-hours per expected new class'.format(
                x['best_weight'], x['predicted_c_rate_at_best'], x['cpu_hours_per_expected_class']))
    if 'ratio_94_over_86' in a:
        r = a['ratio_94_over_86']
        print('\nL94 / L86: C rate x{:.3f}, CPU per trial x{:.1f}, CPU per expected class x{:.0f}'.format(
            r['c_rate_at_best'], r['cpu_per_trial'], r['cpu_per_expected_class']))


if __name__ == '__main__':
    main()
