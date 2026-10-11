"""Memory soak for the actual search; isolated outputs, current RSS + MPS bytes.

--source selects an unchanged source snapshot for paired measurements.
Sampling and JSON I/O are excluded from the formation step throughput timer.
Never reads formal solution archives as seeds. Output directory must be new.
"""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import threading
import tracemalloc
from time import perf_counter


def memory_snapshot(torch, device):
    row = dict(rss_bytes=int(subprocess.check_output(
        ['ps', '-o', 'rss=', '-p', str(os.getpid())], text=True).strip())*1024,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss *
                       (1 if sys.platform == 'darwin' else 1024))
    if device == 'mps':
        # Allocator counters are safe to query from the sampler. Do not
        # synchronize a stream another thread is concurrently submitting to.
        row.update(mps_active_bytes=torch.mps.current_allocated_memory(),
                   mps_driver_bytes=torch.mps.driver_allocated_memory())
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='mps', choices=('mps', 'cpu'))
    parser.add_argument('--length', type=int, default=86)
    parser.add_argument('--batch', type=int, default=16384)
    parser.add_argument('--steps', type=int, default=12000)
    parser.add_argument('--sample-every', type=int, default=500)
    parser.add_argument('--restart-steps', type=int, default=2000)
    parser.add_argument('--mode', choices=('formation', 'hybrid', 'metadata'), default='formation')
    parser.add_argument('--seconds', type=int, default=600)
    parser.add_argument('--empty-cache', action='store_true', help='diagnostic only, after every sample')
    args = parser.parse_args()
    if min(args.batch, args.steps, args.sample_every, args.restart_steps, args.seconds) < 1:
        parser.error('work and sampling limits must be positive')
    args.source = args.source.resolve(); args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.source))
    if args.mode == 'metadata':
        from solver.search_metrics import SearchMetrics
        metrics = SearchMetrics(args.output, args.length, 123)
        samples = []; started = perf_counter(); tracemalloc.start()
        for i in range(args.steps):
            key = (args.length, 2, -1, 1, (tuple((i >> j) & 1 for j in range(args.length//2)), 16, 22),
                   ((0,)*(args.length//2), 18, 21))
            metrics.matched(key)
            if (i+1) % args.sample_every == 0 or i+1 == args.steps:
                row = dict(keys=i+1, elapsed=perf_counter()-started,
                    python_bytes=tracemalloc.get_traced_memory()[0], **memory_snapshot(None, 'cpu'))
                samples.append(row); print(json.dumps(row), flush=True)
        for i in range(args.steps):
            metrics.matched((args.length, 2, -1, 1,
                (tuple((i >> j) & 1 for j in range(args.length//2)), 16, 22), ((0,)*(args.length//2), 18, 21)))
        assert metrics.snapshot()['unique_compressed_matches'] == args.steps
        tracemalloc.stop()
        if hasattr(metrics, 'close'): metrics.close()
        else: metrics.flush(force=True)
        (args.output/'report.json').write_text(json.dumps(dict(parameters={k:str(v) if isinstance(v,Path) else v
            for k,v in vars(args).items()}, samples=samples, duplicates_correct=True), indent=2))
        return
    import torch
    from solver.torch_hybrid_runner import TorchHybridConfig, _new_continuous_search, run_torch_hybrid_search
    from main import _CCompletionBridge, parse_args
    cfg = TorchHybridConfig(args.length, seed=123, root=args.output, device=args.device,
        seconds=args.seconds if args.mode == 'hybrid' else float('inf'),
        continuous_batch_size=args.batch, continuous_steps_per_restart=args.restart_steps,
        continuous_mathematical_loss='lattice', direct_compression=True, polish_mode='alternating', c_threads=6)
    samples = []; started = perf_counter(); search_seconds = 0.0
    def save(row):
        caches = {}
        for name in ('solver.compressed_pairing', 'solver.psd_pruning', 'solver.compressed_equivalence'):
            module = sys.modules.get(name)
            if module is not None:
                for key, value in vars(module).items():
                    if hasattr(value, 'cache_info'):
                        info = value.cache_info()
                        caches[name+'.'+key] = dict(size=info.currsize, limit=info.maxsize)
        samples.append(dict(elapsed=perf_counter()-started, caches=caches,
                            **row, **memory_snapshot(torch, args.device)))
        with (args.output/'samples.jsonl').open('a') as handle:
            handle.write(json.dumps(samples[-1])+'\n')
        print(json.dumps(samples[-1]), flush=True)
    bridge_args = parse_args(['--L', str(args.length), '--z3', '--torch-direct-compression',
        '--torch-polish-mode', 'alternating', '--z3-orientation-search', '--z3-lift-enumeration'])
    bridge = _CCompletionBridge(bridge_args, args.length, args.output)
    try:
        if args.mode == 'formation':
            search = _new_continuous_search(cfg)
            save(dict(step=0, search_seconds=0.0, generation=search.generation))
            for index in range(args.steps):
                tick = perf_counter()
                if search.round_step >= search.config.steps_per_restart:
                    search.generation += 1; search._initialize_batch()
                search.step()
                if search.epoch % search.config.observation_interval == 0:
                    search.observe(elite_callback=bridge.consider)
                if args.device == 'mps': torch.mps.synchronize()
                search_seconds += perf_counter()-tick
                if (index+1) % args.sample_every == 0 or index+1 == args.steps:
                    save(dict(step=index+1, search_seconds=search_seconds,
                        generation=search.generation, archive=len(search.archive),
                        results=len(bridge._results), backlog=bridge._queue.qsize()))
                    if args.empty_cache and args.device == 'mps':
                        torch.mps.empty_cache()
        else:
            errors = []; results = []
            def run():
                try:
                    results.append(run_torch_hybrid_search(cfg, elite_callback=bridge.consider, metrics=bridge.metrics))
                except BaseException as error: errors.append(error)
            worker = threading.Thread(target=run); worker.start()
            while worker.is_alive():
                worker.join(timeout=15)
                save(dict(completion_turns=bridge.trigger_count, backlog=bridge._queue.qsize()))
            if errors: raise errors[0]
            (args.output/'hybrid.json').write_text(json.dumps(asdict(results[0]), default=str, indent=2))
    finally:
        bridge.close()
    report = dict(parameters={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                  torch_version=torch.__version__, samples=samples, search_seconds=search_seconds)
    (args.output/'report.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__': main()
