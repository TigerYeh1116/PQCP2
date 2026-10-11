"""IDE Run: measure exact affine consequences on archived L43 domains.

Samples compatible weight patterns, not full supports or putative solutions.
Reports mathematical ranks; it makes no runtime or global exclusion claim.
"""
import argparse
from collections import Counter
from datetime import datetime
import ctypes
import hashlib
from itertools import combinations
import json
from pathlib import Path
from random import Random
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from solver.checkpoint import atomic_write_json
from experiments.benchmark_l86_discovery import freeze_sources
from solver.quadratic_affine_consequences import affine_combinations, restrict_affine
from solver.ternary_system_native import _library, prepare_native_system
from solver.weight_prefix_native import prepare_weight_prefix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'results/L86_wide_count_full_modular_audit_20261009/report.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'results' / (
        'L86_affine_consequence_probe_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f')))
    parser.add_argument('--patterns', type=int, default=64)
    parser.add_argument('--_frozen-runner', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 1 <= args.patterns <= 1024:
        parser.error('1..1024 patterns required')
    output = args.output.resolve()
    snapshot = output / 'source_snapshot'
    if not args._frozen_runner:
        output.mkdir(parents=True, exist_ok=False)
        freeze_sources(snapshot)
        for name in (Path(__file__).name, 'benchmark_l86_discovery.py'):
            dest = snapshot / 'experiments' / name
            dest.parent.mkdir(exist_ok=True)
            shutil.copy2(ROOT / 'experiments' / name, dest)
        shutil.copy2(args.input.resolve(), output / 'input_domain.json')
        manifest = {str(p.relative_to(snapshot)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted(snapshot.rglob('*')) if p.is_file() and p.name != 'source_manifest.json'}
        atomic_write_json(snapshot / 'source_manifest.json', manifest)
        command = [sys.executable, str(snapshot / 'experiments' / Path(__file__).name),
                   '--_frozen-runner', '--input', str(output / 'input_domain.json'),
                   '--output', str(output), '--patterns', str(args.patterns)]
        atomic_write_json(output / 'execution.json', dict(actual_frozen_execution=True, command=command,
                          cwd=str(snapshot), input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest()))
        raise SystemExit(subprocess.run(command, cwd=snapshot).returncode)
    if ROOT != snapshot:
        raise RuntimeError('archived execution required')
    manifest = json.loads((snapshot / 'source_manifest.json').read_text())
    if any(hashlib.sha256((snapshot / name).read_bytes()).hexdigest() != digest for name, digest in manifest.items()):
        raise RuntimeError('frozen source changed')
    source = json.loads(args.input.read_text())
    rows = []
    lib = _library()
    total_patterns = extra_rank = contradictions = extra_contradictions = 0
    constant_contradictions = extra_constant_contradictions = 0
    for number, item in enumerate(source['observations']):
        raw = prepare_native_system(tuple(item['masks']), 43, tuple(item['wanted']), tuple(item['negative']))
        prepared = prepare_weight_prefix(raw, 2**23)
        if prepared is None:
            raise RuntimeError('archived eligible domain became ineligible')
        p, f = int(prepared['meta'][5]), int(prepared['meta'][4])
        q = p + f
        toggles = (ctypes.c_uint64 * (2 * q))(*prepared['patterns'][:2*p], *prepared['free'][:2*f])
        constant = ctypes.c_uint32()
        linear = (ctypes.c_uint32 * q)()
        quadratic = (ctypes.c_uint32 * (q*q))()
        if lib.pqcp_quadratic_plan(raw['mask_array'], prepared['base'], toggles, q, 43,
                                  raw['wanted'], raw['negative'], ctypes.byref(constant), linear, quadratic):
            raise RuntimeError('quadratic preparation failed')
        free_quadratic = tuple(quadratic[(p+i)*q+p+j] for i in range(f) for j in range(f))
        masks = affine_combinations(free_quadratic, f, 23)
        count_mask = 1 << (21 + 1 - prepared['selected'])
        count_affine = all(not value & count_mask for value in free_quadratic)
        patterns = []
        for positions in combinations(range(p), raw['negative'][prepared['selected']]):
            pattern = sum(1 << j for j in positions)
            if all(bin(pattern & prepared['checks'][j]).count('1') % 2 == prepared['rhs'][j]
                   for j in range(prepared['meta'][6])):
                patterns.append(pattern)
        chosen = Random(860150 + number).sample(patterns, min(args.patterns, len(patterns)))
        histogram = Counter()
        local_extra = local_bad = local_extra_bad = 0
        local_constant = local_extra_constant = 0
        for pattern in chosen:
            active = [j for j in range(p) if pattern >> j & 1]
            c = constant.value
            for i in active:
                c ^= linear[i]
                for j in active:
                    if i < j:
                        c ^= quadratic[i*q+j]
            lf = [linear[p+j] for j in range(f)]
            for i in active:
                for j in range(f):
                    lf[j] ^= quadratic[i*q+p+j]
            affine = restrict_affine(masks, c, lf)
            old = restrict_affine((count_mask,), c, lf) if count_affine else dict(consistent=True, rank=0)
            constant_bad = any(bin(c & mask).count('1') % 2 and
                               all(bin(value & mask).count('1') % 2 == 0 for value in lf)
                               for mask in masks)
            if constant_bad and affine['consistent']:
                raise RuntimeError('constant affine contradiction disagrees with full elimination')
            local_constant += int(constant_bad)
            local_extra_constant += int(constant_bad and old['consistent'])
            key = ('consistent' if affine['consistent'] else 'contradiction', affine['rank'], old['rank'])
            histogram[str(key)] += 1
            if not affine['consistent']:
                local_bad += 1
                local_extra_bad += int(old['consistent'])
            elif affine['rank'] > old['rank']:
                local_extra += 1
            if not old['consistent'] and affine['consistent']:
                raise RuntimeError('count consequence missing from full cancellation kernel')
        total_patterns += len(chosen)
        extra_rank += local_extra
        contradictions += local_bad
        extra_contradictions += local_extra_bad
        constant_contradictions += local_constant
        extra_constant_contradictions += local_extra_constant
        rows.append(dict(domain=number, p=p, f=f, selected=prepared['selected'],
                         affine_combination_dimension=len(masks), count_affine=count_affine,
                         compatible_patterns=len(patterns), sampled_patterns=len(chosen),
                         additional_rank_patterns=local_extra, contradictions=local_bad,
                         additional_contradictions=local_extra_bad, ranks=dict(histogram)))
        rows[-1].update(constant_only_contradictions=local_constant,
                        additional_constant_only_contradictions=local_extra_constant)
    report = dict(complete=True, actual_frozen_execution=True, input=str(args.input.resolve()), domains=len(rows),
                  sampled_patterns=total_patterns, additional_rank_patterns=extra_rank,
                  contradictions=contradictions, additional_contradictions=extra_contradictions,
                  constant_only_contradictions=constant_contradictions,
                  additional_constant_only_contradictions=extra_constant_contradictions,
                  observations=rows,
                  scope='Uniform seeded samples of compatible literal weight patterns in each archived support. Exact affine ranks only; not an exhaustive domain exclusion or speed measurement.')
    atomic_write_json(output / 'report.json', report)
    print(json.dumps({key: value for key, value in report.items() if key != 'observations'}, sort_keys=True))


if __name__ == '__main__':
    main()
