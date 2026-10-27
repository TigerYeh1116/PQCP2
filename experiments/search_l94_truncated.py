"""IDE Run: low-energy truncated search for L=94 (d=47); runs until its budget or Ctrl+C.

Per C_A support (uniform, weight 22 or 24): keep the RUN_KEEP sign patterns with the
smallest sum_u R_A(u)^2, walk one C_B support per rotation orbit of the GF(2) family
and complete C_B exactly (solver/truncated_search.py). C solutions concentrate on those
patterns, so this yields about 12x more independent C witnesses per second than the
GPU+CPU support sampler on the M6 (0 false witnesses in every audit). With RUN_GPU the
C_B scan runs on the Apple GPU (solver/gpu_truncated.py; identical witnesses, 5.5 C
witnesses/s vs 0.72 for the 12-core CPU scan) and the CPU only prepares and checks. Each witness
(one per dihedral class of C_B) is re-solved exactly and its G problems are checked
for every full target sharing its C problem; verified new pairs go to 94.txt.
Sampling only: unvisited patterns and supports are never excluded.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime
import multiprocessing
from pathlib import Path
import random
import secrets
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.search_l94_supports import _groups, _ignore_interrupt, _reference
from solver.checkpoint import atomic_write_json

RUN_WEIGHTS = (22, 24)   # eta -1 contents (see search_l94_supports.py).
RUN_KEEP = 300           # Kept sign patterns per C_A support; 200..400 measured best on L94.
RUN_TRIALS = 0           # C_A supports; 0 = run until Ctrl+C.
RUN_WORKERS = 12
RUN_GPU = True           # Apple GPU scan (csrc/gpu_truncated.m): ~7x the 12-core CPU scan, identical witnesses.
BLOCK = 8                # C_A supports per worker job (~0.9 s each on L94 on the CPU path).
GPU_SUPPORTS = 48        # C_A supports per GPU dispatch (~0.5 s, ~30 MB on L94).


def _support(seed, n, groups, d):
    rng = random.Random(seed*1_000_003+n)
    group = groups[n % len(groups)]
    (an, _, ap), (bn, _, _) = group['contents']
    k = min(group['targets'][0].k, d-group['targets'][0].k)
    return group, an+ap, an, bn, k, sum(1 << j for j in rng.sample(range(d), an+ap))


def _check(seed, n, groups, d, support_b_hits, counts, witnesses):
    """Re-solve each dihedral class of C_B witnesses exactly and check its G problems."""
    from solver.pqcp_support_solver import complete_pqcp_supports
    group, w, _, _, _, support_a = _support(seed, n, groups, d)
    a, b, _, _ = _reference(group)

    def dihedral(v):
        reflected = sum(1 << ((d-j) % d) for j in range(d) if v >> j & 1)
        return min(min(((x << s) | (x >> (d-s))) & ((1 << d)-1) for s in range(d)) for x in (v, reflected))

    for support_b in sorted({dihedral(sb): sb for sb in support_b_hits}.values()):
        masks = (support_a, support_b)
        counts['c_witnesses'] += 1
        for target in group['targets']:
            full = complete_pqcp_supports(a, b, target, masks, max_assignments=2**25,
                                          native_preprocessing=False, binary_content_pruning=False)
            counts['g_checks'] += 1
            counts['g_unknown'] += full['status'] == 'UNKNOWN'
            counts['c_unconfirmed'] += full['status'] == 'UNSAT_BINARY_SUPPORT' and \
                full.get('excluded_by') == 'coarse'
            witnesses.append(dict(support=n, w=w, target=vars(target), contents=group['contents'],
                                  masks=list(masks), full_status=full['status'], words=full['words']))
            if full['verified']:
                break


def run_block(job):
    """CPU path: select, scan and check supports start..stop-1."""
    seed, start, stop, weights, length, keep = job
    from solver.truncated_search import family_target, top_signs, truncated_scan
    groups, d = _groups(weights, length), length//2
    counts, witnesses, tick = Counter(), [], perf_counter()
    for n in range(start, stop):
        group, w, an, bn, k, support_a = _support(seed, n, groups, d)
        counts['supports'] += 1
        if not family_target(d, support_a, k)[0]:
            counts['zero_branch_skipped'] += 1  # probability 2^-m; outside the paired parametrization
            continue
        patterns, _, _ = top_signs(d, support_a, an, keep)
        hits, stats = truncated_scan(d, support_a, k, d-w, bn, _reference(group)[3], patterns)
        counts['support_orbits'] += stats['weight_matches']
        counts['wide_kernel_skipped'] += stats['wide_kernel_skipped']
        _check(seed, n, groups, d, [sb for sb, _, _ in hits], counts, witnesses)
    counts['seconds'] = perf_counter()-tick
    return dict(counts), witnesses


def prepare_block(job):
    """GPU path, CPU side: the kept sign patterns and the C_B supports of supports start..stop-1."""
    seed, start, stop, weights, length, keep = job
    from solver.gpu_truncated import pack_job
    from solver.truncated_search import family_supports, family_target, top_signs
    groups, d = _groups(weights, length), length//2
    counts, items = Counter(), []
    for n in range(start, stop):
        group, w, an, bn, k, support_a = _support(seed, n, groups, d)
        counts['supports'] += 1
        if not family_target(d, support_a, k)[0]:
            counts['zero_branch_skipped'] += 1
            continue
        packed = pack_job(d, family_supports(d, support_a, k, d-w, raw=True), _reference(group)[3], bn,
                          top_signs(d, support_a, an, keep)[0])
        counts['support_orbits'] += len(packed['sb'])//8
        items.append((n, packed))
    return dict(counts), items


def check_block(job):
    """GPU path: exact re-solve and G checks of the GPU's C witnesses."""
    seed, weights, length, hits = job
    groups, d = _groups(weights, length), length//2
    counts, witnesses = Counter(), []
    for n, support_b_hits in hits:
        _check(seed, n, groups, d, support_b_hits, counts, witnesses)
    return dict(counts), witnesses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trials', type=int, default=RUN_TRIALS, help='C_A supports; 0 = until Ctrl+C')
    parser.add_argument('--workers', type=int, default=RUN_WORKERS)
    parser.add_argument('--weights', type=int, nargs='+', default=RUN_WEIGHTS)
    parser.add_argument('--keep', type=int, default=RUN_KEEP)
    parser.add_argument('--gpu', action=argparse.BooleanOptionalAction, default=RUN_GPU)
    parser.add_argument('--seed', type=int, default=secrets.randbelow(2**31))
    parser.add_argument('--length', type=int, default=94, help='94, or 46 as the tested analogue')
    parser.add_argument('--append-root', action=argparse.BooleanOptionalAction, default=True,
                        help='Append verified new pairs to the project L.txt (else to the output directory)')
    parser.add_argument('--output', type=Path,
                        default=ROOT/'results'/('L94_truncated_'+datetime.now().strftime('%Y%m%d_%H%M%S')))
    args = parser.parse_args()
    weights = tuple(args.weights)
    if args.trials < 0 or args.workers < 1 or not 1 <= args.keep <= 4096 or not _groups(weights, args.length):
        parser.error('nonnegative trials, positive workers, 1 <= keep <= 4096 and weights with proven contents required')
    args.output.mkdir(parents=True, exist_ok=False)
    from solver.search_runner import append_verified_solution_if_new
    gpu = None
    if args.gpu:
        try:
            from solver.gpu_truncated import GpuTruncated
            gpu = GpuTruncated(args.length//2)
        except Exception as error:  # no Metal device, or the kernel failed to build
            print('GPU unavailable ({}); using the CPU scan.'.format(error), flush=True)
    print('output:', args.output, ' seed', args.seed, ' weights', list(weights), ' keep', args.keep,
          ' supports', args.trials or 'until Ctrl+C', ' GPU scan' if gpu else ' CPU scan', flush=True)
    totals, witnesses, found = Counter(), [], []
    tick, last, next_start = perf_counter(), 0.0, 0
    budget = args.trials or float('inf')

    def report(final=False):
        atomic_write_json(args.output/'report.json', dict(seed=args.seed, weights=list(weights), keep=args.keep,
            method='low-energy truncated C search', gpu=bool(gpu), counts=dict(totals), elapsed=perf_counter()-tick,
            complete=final and totals['supports'] >= budget, new_l94=found,
            scope='Kept low-energy sign patterns on sampled supports; exact decisions; nothing excluded.'))
        atomic_write_json(args.output/'witnesses.json', witnesses)

    pool = ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context('spawn'), initializer=_ignore_interrupt)
    gpu_thread = ThreadPoolExecutor(1) if gpu else None
    pending, buffer = {}, []  # future -> kind; prepared (n, packed job) awaiting the GPU

    def gpu_scan(batch):  # ctypes releases the GIL while the GPU runs
        hits, wide = gpu.scan([packed for _, packed in batch])
        return batch, hits, wide

    def absorb(counts, rows):
        totals.update(counts)
        witnesses.extend(rows)
        for r in rows:
            if r['words'] is not None and append_verified_solution_if_new(
                    args.length, *r['words'], ROOT if args.append_root else args.output):
                found.append(r)
                print('*** NEW VERIFIED L{} PQCP appended ***'.format(args.length), flush=True)

    try:
        while pending or buffer or next_start < budget:
            kinds = Counter(pending.values())
            if gpu:
                while next_start < budget and kinds['prep'] < 2*args.workers and \
                        len(buffer)+BLOCK*kinds['prep'] < 3*GPU_SUPPORTS:
                    stop = int(min(next_start+BLOCK, budget))
                    pending[pool.submit(prepare_block, (args.seed, next_start, stop, weights, args.length,
                                                        args.keep))] = 'prep'
                    kinds['prep'] += 1
                    next_start = stop
                if not kinds['gpu'] and (len(buffer) >= GPU_SUPPORTS or buffer and not kinds['prep']):
                    pending[gpu_thread.submit(gpu_scan, buffer[:GPU_SUPPORTS])] = 'gpu'
                    del buffer[:GPU_SUPPORTS]
            else:
                while next_start < budget and len(pending) < 2*args.workers:
                    stop = int(min(next_start+BLOCK, budget))
                    pending[pool.submit(run_block, (args.seed, next_start, stop, weights, args.length,
                                                    args.keep))] = 'cpu'
                    next_start = stop
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                kind = pending.pop(future)
                if kind == 'prep':
                    counts, items = future.result()
                    totals['supports'] += counts.get('zero_branch_skipped', 0)  # the rest count once scanned
                    totals['zero_branch_skipped'] += counts.get('zero_branch_skipped', 0)
                    buffer.extend(items)
                elif kind == 'gpu':
                    batch, hits, wide = future.result()
                    totals['supports'] += len(batch)
                    totals['support_orbits'] += sum(len(packed['sb'])//8 for _, packed in batch)
                    totals['wide_kernel_skipped'] += wide
                    found_hits = [(n, [sb for sb, _, _ in h]) for (n, _), h in zip(batch, hits) if h]
                    if found_hits:
                        pending[pool.submit(check_block, (args.seed, weights, args.length, found_hits))] = 'check'
                else:
                    absorb(*future.result())
            if perf_counter()-last >= 60:
                last = perf_counter()
                report()
                elapsed = perf_counter()-tick
                print('[{:%H:%M:%S}] {:,} supports ({:.1f}/s), C witnesses {} ({:.2f}/s), G checks {}, new L{} {}'.format(
                    datetime.now(), totals['supports'], totals['supports']/elapsed, totals['c_witnesses'],
                    totals['c_witnesses']/elapsed, totals['g_checks'], args.length, len(found)), flush=True)
    except KeyboardInterrupt:
        print('\nStopping; finished blocks are kept in the report.', flush=True)
        workers = list((pool._processes or {}).values())
        pool.shutdown(wait=False, cancel_futures=True)
        if gpu_thread:
            gpu_thread.shutdown(wait=False, cancel_futures=True)
        for process in workers:
            process.terminate()
        for process in workers:
            process.join(timeout=30)
    else:
        pool.shutdown()
        if gpu_thread:
            gpu_thread.shutdown()
    finally:
        report(final=True)
    print('{:,} supports, C witnesses {}, G checks {}, new L{} {}; report: {}'.format(
        totals['supports'], totals['c_witnesses'], totals['g_checks'], args.length, len(found),
        args.output/'report.json'), flush=True)


if __name__ == '__main__':
    main()
