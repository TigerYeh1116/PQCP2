"""Frozen full modular-set and integer-decision audit on real L86 supports.

An extra native diagnostic rebuilds every modular survivor's literal
integer syndrome. Neither a modular survivor nor a compressed SAT is a
full PQCP. Every eligible label is compared with independent Gray C.
"""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.benchmark_l86_discovery import freeze_sources
from experiments import benchmark_weight_group_pipeline as pipeline
from solver.checkpoint import atomic_write_json
from solver.compressed_repair import _validated_reference
from solver.mod8_model_reference import all_mod8_indices
from solver.mps_mod8_prefix import quadratic_coefficients
from solver import paired_signs_native as paired
from solver.projection_group_targets import group_projection_targets
from solver.ternary_system_native import prepare_native_system, run_native_system
from solver.weight_group_targets import group_weight_targets
from solver.structured_energy import integer_periodic_autocorrelation_profile as pacf
from solver.ternary_support_search import support_mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pairs', type=int, default=96)
    parser.add_argument('--seed', type=int, default=202610094)
    parser.add_argument('--carriers', type=Path,
                        help='Optional actual positive C bank; audit eligible pairs in source order')
    parser.add_argument('--_frozen-runner', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 1 <= args.pairs <= 512:
        parser.error('1..512 complete paired domains required')
    output = args.output.resolve()
    snapshot = output/'source_snapshot'
    if not args._frozen_runner:
        output.mkdir(parents=True, exist_ok=False)
        freeze_sources(snapshot)
        for name in ('benchmark_l86_discovery.py', 'benchmark_weight_group_pipeline.py',
                     'search_norm_supports.py', Path(__file__).name):
            dest = snapshot/'experiments'/name
            dest.parent.mkdir(exist_ok=True)
            shutil.copy2(ROOT/'experiments'/name, dest)
        manifest = {str(p.relative_to(snapshot)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted(snapshot.rglob('*')) if p.is_file() and p.name != 'source_manifest.json'}
        atomic_write_json(snapshot/'source_manifest.json', manifest)
        if args.carriers is not None:
            shutil.copy2(args.carriers.resolve(), output/'input_carriers.json')
        command = [sys.executable, str(snapshot/'experiments'/Path(__file__).name),
            '--_frozen-runner', '--output', str(output), '--pairs', str(args.pairs), '--seed', str(args.seed)]
        if args.carriers is not None:command.extend(('--carriers', str(output/'input_carriers.json')))
        atomic_write_json(output/'execution.json', dict(actual_frozen_execution=True,
            command=command, cwd=str(snapshot), source_manifest_sha256=hashlib.sha256(pipeline.encode(manifest).encode()).hexdigest()))
        raise SystemExit(subprocess.run(command, cwd=snapshot).returncode)
    if ROOT != snapshot:
        raise RuntimeError('actual frozen execution required')
    manifest = json.loads((snapshot/'source_manifest.json').read_text())
    if any(hashlib.sha256((snapshot/name).read_bytes()).hexdigest() != digest for name, digest in manifest.items()):
        raise RuntimeError('frozen source changed')
    import experiments.search_norm_supports as search
    search.initialize()
    groups = group_weight_targets(group_projection_targets(search.CHOICES))
    atomic_write_json(output/'target_domain.json', json.loads(pipeline.encode(groups)))
    sources = [ROOT/'csrc'/name for name in ('ternary_system.c', 'ternary_signs.c',
               'ternary_weight_signs.c', 'ternary_quadratic_signs.c')]
    library_path = output/('audit_paired.dylib' if sys.platform == 'darwin' else 'audit_paired.so')
    command = ['cc', '-O3', '-std=c11', '-Wall', '-Wextra', '-Werror', '-fPIC',
        '-DPQCP_PAIRED_AUDIT_SYNDROME', '-dynamiclib' if sys.platform == 'darwin' else '-shared',
        *map(str, sources), '-o', str(library_path)]
    subprocess.run(command, check=True, capture_output=True, timeout=30)
    diagnostic = ctypes.CDLL(str(library_path))
    model_function = diagnostic.pqcp_quadratic_paired_bit_tail_models
    p = ctypes.POINTER(ctypes.c_uint64)
    ip = ctypes.POINTER(ctypes.c_int)
    qp = ctypes.POINTER(ctypes.c_uint32)
    model_function.argtypes = [p, p, p, ctypes.c_int, ctypes.c_int, ip, ip, ctypes.c_int,
                              ctypes.c_uint64, qp, p]
    model_function.restype = ctypes.c_int
    observations = []
    source_rows = None if args.carriers is None else json.loads(args.carriers.read_text())['rows']
    total_models = 0
    total_affine = 0
    old_library = paired._library
    paired._library = lambda: diagnostic
    try:
        for trial in range(args.pairs*128 if source_rows is None else len(source_rows)):
            source_word = source_case = None
            group_index = trial % len(groups) if source_rows is None else source_rows[trial]['weight_group']
            group = groups[group_index]
            if len(group['c_groups']) != 2:
                continue
            if source_rows is None:
                u, sampled = pipeline.sample(group, trial, args.seed)
                if sampled['mask'] is None:continue
                masks = (u, sampled['mask'])
            else:
                item = source_rows[trial]
                source_word = tuple(tuple(w) for w in item['result']['attempts']['coarse']['words'])
                source_case = item['c_group']
                entry = group['c_groups'][source_case]
                if (tuple(tuple(c) for c in item['contents']) != entry['contents']
                        or item['target'] != pipeline.asdict(entry['targets'][0])):
                    raise RuntimeError('positive source target/content is outside the fresh parent domain')
                a,b,d,wanted = _validated_reference(*source_word, entry['targets'][0])
                if tuple(x+y for x,y in zip(pacf(a),pacf(b))) != wanted:
                    raise RuntimeError('positive input is not an integer C witness')
                masks = tuple(support_mask(w) for w in source_word)
                if masks != (item['u'],item['sampling']['mask']):
                    raise RuntimeError('positive input supports differ from its literal sampling ledger')
            plans = []
            for entry in group['c_groups']:
                refs = tuple(tuple([-1]*n+[0]*z+[1]*v) for n, z, v in entry['contents'])
                a, b, d, wanted = _validated_reference(*refs, entry['targets'][0])
                plans.append(prepare_native_system(masks, d, wanted, (a.count(-1), b.count(-1))))
            native_args = paired.paired_arguments(plans, 2**23)
            if native_args is None:
                continue
            q = int(plans[0]['meta'][2])
            capacity = 1 << q
            outputs = (ctypes.c_uint32*(2*capacity))()
            counts = (ctypes.c_uint64*2)()
            code = model_function(*native_args, capacity, outputs, counts)
            if code or any(count > capacity for count in counts):
                raise RuntimeError('complete diagnostic modular enumeration failed')
            for k, plan in enumerate(plans):
                expected = all_mod8_indices(quadratic_coefficients(plan), capacity=capacity)
                actual = tuple(sorted(outputs[k*capacity:k*capacity+counts[k]]))
                if not expected['complete'] or actual != expected['indices']:
                    raise RuntimeError('paired label lost or added an original modular model')
            results = paired.run_native_pair(plans, 2**23)
            baseline = tuple(run_native_system(plan, 2**23) for plan in plans)
            if any(new['status'] != old['status'] or new['words'] != old['words']
                   for new, old in zip(results, baseline)):
                raise RuntimeError('actual full paired label decisions or witnesses differ')
            if source_word is not None and results[source_case]['words'] != source_word:
                raise RuntimeError('positive source witness differs from fresh paired/baseline replay')
            observations.append(dict(trial=trial, group=group_index, masks=masks,
                q=q, full_affine_assignments=2*capacity, modular_counts=list(counts),
                statuses=[r['status'] for r in results], words=[r['words'] for r in results]))
            total_models += sum(counts)
            total_affine += 2*capacity
            if len(observations) == args.pairs:
                break
    finally:
        paired._library = old_library
    if len(observations) != args.pairs:
        raise RuntimeError('insufficient eligible actual paired supports; no complete audit claimed')
    report = dict(complete=True, actual_frozen_execution=True, pairs=len(observations),
        independent_label_domains=2*len(observations), full_affine_assignments=total_affine,
        original_modular_zero_sets_match=True, all_integer_decisions_and_witnesses_match=True,
        directly_audited_survivor_syndromes=total_models, observations=observations,
        source_positive_carriers=source_rows is not None,
        input_carriers_sha256=None if args.carriers is None else hashlib.sha256(args.carriers.read_bytes()).hexdigest(),
        diagnostic_compile_command=command,
        diagnostic_binary_sha256=hashlib.sha256(library_path.read_bytes()).hexdigest(),
        scope='All original mod8 models and bounded complete integer decisions of these actual two-label raw affine domains. No full L86 discoveries and no global exclusion.')
    atomic_write_json(output/'report.json', report)
    print(pipeline.encode({k: report[k] for k in ('complete', 'pairs', 'full_affine_assignments',
        'directly_audited_survivor_syndromes')}), flush=True)


if __name__ == '__main__':
    main()
