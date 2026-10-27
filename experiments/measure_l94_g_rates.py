"""Measure the unconditional G-sign rate at L94 next to the C rate, on the SAME support pairs.

Same proposal distribution as search_l94_supports.py (groups for C_A weights
22..25, uniform C_A support, uniform GF(2)-compatible C_B support). Every
pair gets the exact C problem AND the exact G problem of every target in its
group, independently, so p_C, p_G per target type and their joint rate are
measured directly instead of assuming p_G = p_C.
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
from experiments.search_l94_supports import FAST, _groups
from solver.checkpoint import atomic_write_json


def solve(masks, d, wanted, negative):
    from solver.ternary_signs_native import prepare_sign_enumeration, solve_sign_system
    from solver.ternary_support_search import support_sign_system
    system = support_sign_system(masks, d, wanted, negative)
    if not system.consistent:
        return 'UNSAT_FIXED_SUPPORT'
    return solve_sign_system(system, masks, d, wanted, negative, 2**25,
                             prepared=prepare_sign_enumeration(system, masks, negative), **FAST)['status']


def run_block(job):
    seed, start, stop, weights, length = job
    from solver.compressed_repair import _validated_reference
    from solver.gf2_support_family import gf2_support_ring, sample_support
    from solver.odd_lift_native import difference_sign_problem
    ring, groups = gf2_support_ring(length//2), _groups(weights, length)
    counts = Counter()
    for n in range(start, stop):
        rng = random.Random(seed*1_000_003+n)
        group = groups[n % len(groups)]
        (an, az, ap), (bn, bz, bp) = group['contents']
        a, b, d, wanted = _validated_reference((-1,)*an+(0,)*az+(1,)*ap, (-1,)*bn+(0,)*bz+(1,)*bp, group['targets'][0])
        w = an+ap
        u = sum(1 << j for j in rng.sample(range(d), w))
        v = sample_support(ring, u, min(group['targets'][0].k, d-group['targets'][0].k), d-w, rng)
        if v is None:
            continue
        masks = (u, v)
        c = solve(masks, d, wanted, (an, bn)) == 'SAT_COMPRESSED'
        tag = 'w{}'.format(w)
        counts[tag+':pairs'] += 1
        counts[tag+':C'] += c
        for target in group['targets']:
            g_masks, g_wanted, g_negative = difference_sign_problem(masks, target)
            g = solve(g_masks, d, g_wanted, g_negative) == 'SAT_COMPRESSED'
            key = '{}:k{}eta{}'.format(tag, target.k % 2, target.eta)
            counts[key+':G'] += g
            counts[key+':checks'] += 1
            counts[key+':CG'] += c and g
    return dict(counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pairs', type=int, default=3_000_000)
    parser.add_argument('--workers', type=int, default=10)
    parser.add_argument('--length', type=int, default=94)
    parser.add_argument('--weights', type=int, nargs='+', default=(22, 23, 24, 25))
    parser.add_argument('--seed', type=int, default=secrets.randbelow(2**31))
    parser.add_argument('--output', type=Path, default=ROOT/'results'/('L94_g_rates_'+datetime.now().strftime('%Y%m%d_%H%M%S')))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    jobs = [(args.seed, s, min(s+2000, args.pairs), tuple(args.weights), args.length) for s in range(0, args.pairs, 2000)]
    totals, tick, last = Counter(), perf_counter(), 0.0
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        for i, counts in enumerate(pool.map(run_block, jobs), 1):
            totals.update(counts)
            if perf_counter()-last > 60 or i == len(jobs):
                last = perf_counter()
                atomic_write_json(args.output/'report.json', dict(seed=args.seed, length=args.length, blocks=i, of=len(jobs),
                                                                   elapsed=perf_counter()-tick, counts=dict(totals)))
                pairs = sum(v for k, v in totals.items() if k.endswith(':pairs'))
                c = sum(v for k, v in totals.items() if k.endswith(':C'))
                g = sum(v for k, v in totals.items() if k.endswith(':G'))
                print('[{:%H:%M:%S}] {}/{} blocks, pairs {:,}, C-SAT {}, G-SAT {}'.format(datetime.now(), i, len(jobs), pairs, c, g), flush=True)
    print('report:', args.output/'report.json')


if __name__ == '__main__':
    main()
