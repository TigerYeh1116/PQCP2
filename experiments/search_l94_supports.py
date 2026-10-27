"""IDE Run: weight-focused support sampling for L=94 (d=47); runs until its trial budget or Ctrl+C.

Per trial: a uniform C_A support of weight w, a uniform GF(2)-compatible C_B
support (solver/gf2_support_family.py), one exact C sign problem with mod-8
filtering enabled for d=47 (2.5x, checked against the default path), and,
for a C witness, the exact G problems of EVERY full target sharing that C
problem (k odd and k even; group_projection_targets). Verified new pairs are
appended to 94.txt. Pilot L94_pilot_c_rates_20261021_213709 measured the C
rate; expect thousands of CPU-hours per class. A miss excludes nothing.

With RUN_GPU, a second sample stream (disjoint trial indices) is prepared by
RUN_PREP_WORKERS processes and its C sign spaces are enumerated completely on
the Apple GPU (solver/gpu_signs.py); every GPU hit is re-solved by the exact
CPU solver before its G problems are checked. The other workers keep the CPU
stream. Without Metal the search falls back to the CPU stream only.
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
RUN_GPU = True                   # Apple GPU stream (verified identical verdicts to the CPU solver).
RUN_PREP_WORKERS = 8             # CPU processes preparing GPU pairs; 8 nearly saturates the M6 GPU (3855 trials/s total vs 1331 CPU-only).
BLOCK = 500
GPU_BATCH = 4096
GPU_OFFSET = 10**15              # GPU-stream trial indices never meet the CPU stream's.
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
        complete_witness(n, group, a, b, masks, counts, witnesses)
    counts['seconds'] = perf_counter()-tick
    return dict(counts), witnesses


def complete_witness(n, group, a, b, masks, counts, witnesses):
    """A C witness: solve the exact G problem of every target sharing this C problem."""
    from solver.pqcp_support_solver import complete_pqcp_supports
    counts['c_witnesses'] += 1
    for target in group['targets']:
        full = complete_pqcp_supports(a, b, target, masks, max_assignments=2**25,
                                      native_preprocessing=False, binary_content_pruning=False)
        counts['g_checks'] += 1
        counts['g_unknown'] += full['status'] == 'UNKNOWN'
        witnesses.append(dict(trial=n, w=sum(v != 0 for v in a), target=vars(target), contents=group['contents'],
                              masks=list(masks), full_status=full['status'], words=full['words']))
        if full['verified']:
            break


def _reference(group):
    from solver.compressed_repair import _validated_reference
    (an, az, ap), (bn, bz, bp) = group['contents']
    return _validated_reference((-1,)*an+(0,)*az+(1,)*ap, (-1,)*bn+(0,)*bz+(1,)*bp, group['targets'][0])


def prepare_block(job):
    """GPU stream: sample and build each pair's affine sign space; no sign enumeration here."""
    seed, start, stop, weights, length = job
    from solver.gf2_support_family import gf2_support_ring, sample_support
    from solver.gpu_signs import prepare_pair
    ring, groups = gf2_support_ring(length//2), _groups(weights, length)
    counts, items = Counter(), []
    for n in range(start, stop):
        rng = random.Random(seed*1_000_003+n)
        gi = n % len(groups)
        group = groups[gi]
        a, b, d, wanted = _reference(group)
        (an, az, ap), (bn, bz, bp) = group['contents']
        w = an+ap
        u = sum(1 << j for j in rng.sample(range(d), w))
        v = sample_support(ring, u, min(group['targets'][0].k, d-group['targets'][0].k), d-w, rng)
        if v is None:
            counts['trials'] += 1
            counts['no_weight_match'] += 1
            continue
        prepared = prepare_pair((u, v), d, wanted, (an, bn))
        if prepared is None or prepared[3] > 31:  # Already UNSAT, or too wide for the GPU kernel (never seen).
            counts['trials'] += 1
            counts['gpu_skipped_wide'] += prepared is not None
            continue
        items.append((n, gi, prepared))
    return dict(counts), items


def verify_gpu_hit(n, group, masks, length):
    """Re-solve a GPU candidate exactly on the CPU, then complete it like a CPU witness."""
    from solver.ternary_signs_native import prepare_sign_enumeration, solve_sign_system
    from solver.ternary_support_search import support_sign_system
    a, b, d, wanted = _reference(group)
    negative = (a.count(-1), b.count(-1))
    counts, witnesses = Counter(), []
    system = support_sign_system(masks, d, wanted, negative)
    out = solve_sign_system(system, masks, d, wanted, negative, 2**25,
                            prepared=prepare_sign_enumeration(system, masks, negative), **FAST)
    if out['status'] != 'SAT_COMPRESSED':
        counts['gpu_unconfirmed'] += 1
        return counts, witnesses
    complete_witness(n, group, a, b, masks, counts, witnesses)
    return counts, witnesses


def _gpu_loop(length, inbox, outbox, failure):
    """Parent thread: ctypes releases the GIL while the GPU enumerates a batch."""
    try:
        from solver.gpu_signs import GpuSigns
        gpu = GpuSigns(length//2)
    except Exception as error:  # No Metal: the CPU stream keeps running alone.
        failure.append(repr(error))
        outbox.put(None)
        return
    while True:
        batch = inbox.get()
        if batch is None:
            return
        hits, _ = gpu.check([prepared for _, _, prepared in batch])
        outbox.put((batch, hits))


def main():
    import queue
    import threading
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trials', type=int, default=RUN_TRIALS)
    parser.add_argument('--workers', type=int, default=RUN_WORKERS)
    parser.add_argument('--weights', type=int, nargs='+', default=RUN_WEIGHTS)
    parser.add_argument('--seed', type=int, default=secrets.randbelow(2**31))
    parser.add_argument('--length', type=int, default=94, help='2p with p an odd prime <= 47; 46 is the tested analogue')
    parser.add_argument('--gpu', action=argparse.BooleanOptionalAction, default=RUN_GPU)
    parser.add_argument('--prep-workers', type=int, default=RUN_PREP_WORKERS)
    parser.add_argument('--append-root', action=argparse.BooleanOptionalAction, default=True,
                        help='Append verified new pairs to the project L.txt (else to the output directory)')
    parser.add_argument('--output', type=Path, default=ROOT/'results'/('L94_search_'+datetime.now().strftime('%Y%m%d_%H%M%S')))
    args = parser.parse_args()
    groups = _groups(tuple(args.weights), args.length)
    if args.trials < 0 or args.workers < 1 or not groups or (args.gpu and not 1 <= args.prep_workers < args.workers):
        parser.error('nonnegative trials, positive workers, weights with proven contents and 1 <= prep workers < workers required')
    args.output.mkdir(parents=True, exist_ok=False)
    from solver.search_runner import append_verified_solution_if_new
    cpu_workers = args.workers-args.prep_workers if args.gpu else args.workers
    print('output:', args.output, ' seed', args.seed, ' weights', list(args.weights), ' trials', args.trials or 'until Ctrl+C',
          ' GPU stream on ({} prep + {} CPU workers)'.format(args.prep_workers, cpu_workers) if args.gpu else ' CPU only', flush=True)
    totals, witnesses, found = Counter(), [], []
    tick, last = perf_counter(), 0.0
    budget = args.trials or float('inf')
    submitted = {'cpu': 0, 'gpu': 0}
    weights = tuple(args.weights)

    def report(final=False):
        atomic_write_json(args.output/'report.json', dict(seed=args.seed, weights=list(args.weights), gpu=args.gpu,
            counts=dict(totals), elapsed=perf_counter()-tick, complete=final and totals['trials'] >= budget,
            new_l94=found, scope='Sampled supports only; exact per-support decisions; misses exclude nothing.'))
        atomic_write_json(args.output/'witnesses.json', witnesses)

    def absorb(counts, rows):
        totals.update(counts)
        witnesses.extend(rows)
        for r in rows:
            if r['words'] is not None and append_verified_solution_if_new(
                    args.length, *r['words'], ROOT if args.append_root else args.output):
                found.append(r)
                print('*** NEW VERIFIED L{} PQCP appended ***'.format(args.length), flush=True)

    spawn = multiprocessing.get_context('spawn')
    cpu_pool = ProcessPoolExecutor(cpu_workers, mp_context=spawn, initializer=_ignore_interrupt)
    prep_pool = ProcessPoolExecutor(args.prep_workers, mp_context=spawn, initializer=_ignore_interrupt) if args.gpu else None
    inbox, outbox, failure, buffer = queue.Queue(), queue.Queue(), [], []
    gpu_thread = threading.Thread(target=_gpu_loop, args=(args.length, inbox, outbox, failure), daemon=True) if args.gpu else None
    if gpu_thread:
        gpu_thread.start()
    gpu_live, gpu_in_flight = args.gpu, 0
    pending = {}

    def remaining():
        return budget-submitted['cpu']-submitted['gpu']

    try:
        while True:
            while remaining() > 0 and sum(k == 'cpu' for k in pending.values()) < 2*cpu_workers:
                stop = int(submitted['cpu']+min(BLOCK, remaining()))
                pending[cpu_pool.submit(run_block, (args.seed, submitted['cpu'], stop, weights, args.length))] = 'cpu'
                submitted['cpu'] = stop
            while (gpu_live and remaining() > 0 and len(buffer) < 3*GPU_BATCH
                   and sum(k == 'prep' for k in pending.values()) < 2*args.prep_workers):
                stop = int(submitted['gpu']+min(2000, remaining()))
                pending[prep_pool.submit(prepare_block, (args.seed, GPU_OFFSET+submitted['gpu'], GPU_OFFSET+stop,
                                                         weights, args.length))] = 'prep'
                submitted['gpu'] = stop
            if not pending and not gpu_in_flight and not buffer:
                break
            done, _ = wait(list(pending), timeout=0.2, return_when=FIRST_COMPLETED)
            for future in done:
                kind = pending.pop(future)
                if kind == 'cpu':
                    absorb(*future.result())
                else:
                    counts, items = future.result()
                    totals.update(counts)
                    buffer.extend(items)
            if gpu_live and gpu_in_flight < 2 and (len(buffer) >= GPU_BATCH or (buffer and remaining() <= 0 and not
                                                   any(k == 'prep' for k in pending.values()))):
                inbox.put(buffer[:GPU_BATCH])
                del buffer[:GPU_BATCH]
                gpu_in_flight += 1
            while True:
                try:
                    result = outbox.get_nowait()
                except queue.Empty:
                    break
                if result is None:
                    gpu_live = False
                    gpu_in_flight = 0
                    print('GPU unavailable ({}); continuing on the CPU only.'.format(failure[0] if failure else '?'), flush=True)
                    buffer.clear()
                    break
                gpu_in_flight -= 1
                batch, hits = result
                totals['trials'] += len(batch)
                totals['gpu_pairs'] += len(batch)
                for (n, gi, prepared), hit in zip(batch, hits):
                    if hit:
                        absorb(*verify_gpu_hit(n, groups[gi], prepared[0], args.length))
            if perf_counter()-last >= 60:
                last = perf_counter()
                report()
                speed = totals['trials']/(perf_counter()-tick)
                print('[{:%H:%M:%S}] {:,} trials ({:.0f}/s; GPU {:,}), C witnesses {}, G checks {}, new L94 {}'.format(
                    datetime.now(), totals['trials'], speed, totals['gpu_pairs'], totals['c_witnesses'],
                    totals['g_checks'], len(found)), flush=True)
    except KeyboardInterrupt:
        print('\nStopping; finished blocks are kept in the report.', flush=True)
        for pool in (cpu_pool, prep_pool):
            if pool is None:
                continue
            workers = list((pool._processes or {}).values())  # Capture before shutdown clears it.
            pool.shutdown(wait=False, cancel_futures=True)
            for process in workers:
                process.terminate()
            for process in workers:
                process.join(timeout=30)
    else:
        cpu_pool.shutdown()
        if prep_pool:
            prep_pool.shutdown()
    finally:
        if gpu_thread:
            inbox.put(None)
        report(final=True)
    print('{:,} trials (GPU {:,}), C witnesses {}, G checks {}, new L94 {}; report: {}'.format(
        totals['trials'], totals['gpu_pairs'], totals['c_witnesses'], totals['g_checks'], len(found),
        args.output/'report.json'), flush=True)


if __name__ == '__main__':
    main()
