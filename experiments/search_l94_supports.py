"""IDE Run: weight-focused support sampling for L=94 (d=47); runs until its trial budget or Ctrl+C.

Per trial: a uniform C_A support of weight w, a uniform GF(2)-compatible C_B
support (solver/gf2_support_family.py), one exact C sign problem with mod-8
filtering enabled for d=47 (2.5x, checked against the default path), and,
for a C witness, the exact G problems of EVERY full target sharing that C
problem (k odd and k even; group_projection_targets). Verified new pairs are
appended to 94.txt. Pilot L94_pilot_c_rates_20261021_213709 measured the C
rate; expect thousands of CPU-hours per class. A miss excludes nothing.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime
import multiprocessing
from pathlib import Path
import random
import secrets
import signal
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from solver.checkpoint import atomic_write_json

RUN_WEIGHTS = (22, 24)           # Even weights = eta -1 (45 = 3^2+6^2): C and best G rate ~2x the eta +1 types
                                 # (7414 C witnesses, 71 direct G hits); about 1.27x yield per CPU over 22..25.
RUN_TRIALS = 0                   # 0 = run until Ctrl+C.
RUN_WORKERS = 12                 # M6: 2 super + 4 performance + 6 efficiency cores; 12 measured 9% faster than 10.
BLOCK = 500
FAST = dict(quadratic=True, conditioned_hybrid=True, hybrid_tail=6)

_GROUPS = None


def _groups(weights, length=94):
    global _GROUPS
    if _GROUPS is None:
        from solver.projection_group_targets import group_projection_targets
        from solver.target_profiles import canonical_target_content_profiles
        from solver.torch_compressed_search import search_paired_contents
        choices = tuple((t, c) for t in canonical_target_content_profiles(length) for c in search_paired_contents(t))
        _GROUPS = [g for g in group_projection_targets(choices)
                   if g['contents'][0][0]+g['contents'][0][2] in weights]
    return _GROUPS


def _ignore_interrupt():
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # The parent owns Ctrl+C and shuts the pool down.


def run_block(job):
    seed, start, stop, weights, length = job
    from solver.compressed_repair import _validated_reference
    from solver.gf2_support_family import gf2_support_ring, sample_support
    from solver.pqcp_support_solver import complete_pqcp_supports
    from solver.ternary_signs_native import prepare_sign_enumeration, solve_sign_system
    from solver.ternary_support_search import support_sign_system
    ring, groups = gf2_support_ring(length//2), _groups(weights, length)
    counts, witnesses, tick = Counter(), [], perf_counter()
    for n in range(start, stop):
        rng = random.Random(seed*1_000_003+n)
        group = groups[n % len(groups)]
        (an, az, ap), (bn, bz, bp) = group['contents']
        a, b, d, wanted = _validated_reference((-1,)*an+(0,)*az+(1,)*ap, (-1,)*bn+(0,)*bz+(1,)*bp, group['targets'][0])
        w = an+ap
        counts['trials'] += 1
        u = sum(1 << j for j in rng.sample(range(d), w))
        v = sample_support(ring, u, min(group['targets'][0].k, d-group['targets'][0].k), d-w, rng)
        if v is None:
            counts['no_weight_match'] += 1
            continue
        masks, negative = (u, v), (an, bn)
        system = support_sign_system(masks, d, wanted, negative)
        if not system.consistent:
            continue
        out = solve_sign_system(system, masks, d, wanted, negative, 2**25,
                                prepared=prepare_sign_enumeration(system, masks, negative), **FAST)
        if out['status'] == 'UNKNOWN':
            counts['unknown'] += 1
        if out['status'] != 'SAT_COMPRESSED':
            continue
        counts['c_witnesses'] += 1
        for target in group['targets']:
            full = complete_pqcp_supports(a, b, target, masks, max_assignments=2**25,
                                          native_preprocessing=False, binary_content_pruning=False)
            counts['g_checks'] += 1
            counts['g_unknown'] += full['status'] == 'UNKNOWN'
            witnesses.append(dict(trial=n, w=w, target=vars(target), contents=group['contents'],
                                  masks=list(masks), full_status=full['status'], words=full['words']))
            if full['verified']:
                break
    counts['seconds'] = perf_counter()-tick
    return dict(counts), witnesses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trials', type=int, default=RUN_TRIALS)
    parser.add_argument('--workers', type=int, default=RUN_WORKERS)
    parser.add_argument('--weights', type=int, nargs='+', default=RUN_WEIGHTS)
    parser.add_argument('--seed', type=int, default=secrets.randbelow(2**31))
    parser.add_argument('--length', type=int, default=94, help='2p with p an odd prime <= 47; 46 is the tested analogue')
    parser.add_argument('--append-root', action=argparse.BooleanOptionalAction, default=True,
                        help='Append verified new pairs to the project L.txt (else to the output directory)')
    parser.add_argument('--output', type=Path, default=ROOT/'results'/('L94_search_'+datetime.now().strftime('%Y%m%d_%H%M%S')))
    args = parser.parse_args()
    if args.trials < 0 or args.workers < 1 or not _groups(tuple(args.weights), args.length):
        parser.error('nonnegative trials, positive workers and weights with proven contents required')
    args.output.mkdir(parents=True, exist_ok=False)
    from solver.search_runner import append_verified_solution_if_new
    print('output:', args.output, ' seed', args.seed, ' weights', list(args.weights),
          ' trials', args.trials or 'until Ctrl+C', flush=True)
    totals, witnesses, found = Counter(), [], []
    tick, last, next_start = perf_counter(), 0.0, 0
    budget = args.trials or float('inf')

    def report(final=False):
        atomic_write_json(args.output/'report.json', dict(seed=args.seed, weights=list(args.weights),
            counts=dict(totals), elapsed=perf_counter()-tick, complete=final and totals['trials'] >= budget,
            new_l94=found, scope='Sampled supports only; exact per-support decisions; misses exclude nothing.'))
        atomic_write_json(args.output/'witnesses.json', witnesses)

    pool = ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context('spawn'), initializer=_ignore_interrupt)
    pending = set()
    try:
        while pending or next_start < budget:
            while next_start < budget and len(pending) < 2*args.workers:
                stop = int(min(next_start+BLOCK, budget))
                pending.add(pool.submit(run_block, (args.seed, next_start, stop, tuple(args.weights), args.length)))
                next_start = stop
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                counts, rows = future.result()
                totals.update(counts)
                witnesses += rows
                for r in rows:
                    if r['words'] is not None and append_verified_solution_if_new(args.length, *r['words'], ROOT if args.append_root else args.output):
                        found.append(r)
                        print('*** NEW VERIFIED L{} PQCP appended ***'.format(args.length), flush=True)
            if perf_counter()-last >= 60:
                last = perf_counter()
                report()
                speed = totals['trials']/(perf_counter()-tick)
                print('[{:%H:%M:%S}] {:,} trials ({:.0f}/s), C witnesses {}, G checks {}, new L94 {}'.format(
                    datetime.now(), totals['trials'], speed, totals['c_witnesses'], totals['g_checks'], len(found)), flush=True)
    except KeyboardInterrupt:
        print('\nStopping; finished blocks are kept in the report.', flush=True)
        workers = list((pool._processes or {}).values())  # Capture before shutdown clears it.
        pool.shutdown(wait=False, cancel_futures=True)
        for process in workers:
            process.terminate()
        for process in workers:
            process.join(timeout=30)
    else:
        pool.shutdown()
    finally:
        report(final=True)
    print('{:,} trials, C witnesses {}, G checks {}, new L94 {}; report: {}'.format(
        totals['trials'], totals['c_witnesses'], totals['g_checks'], len(found), args.output/'report.json'), flush=True)


if __name__ == '__main__':
    main()
