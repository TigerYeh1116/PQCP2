"""Audit Crossbred tail-linear pruning on the existing literal L43 fixtures.

Sampled prefix rejection is diagnostic; it is not a full-support exclusion,
nor an improvement in actual new L86-class discovery speed.
"""
import argparse
import hashlib
import json
from pathlib import Path
from random import Random
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from solver.checkpoint import atomic_write_json
from solver.mod8_crossbred import (
    affine_tail_indices, degree_three_tail_consequences, specialize_tail_system,
)
from solver.mod8_model_reference import all_mod8_indices
from solver.mps_mod8_prefix import evaluate, packed_coefficients, quadratic_coefficients
from solver.ternary_system_native import prepare_native_system, run_native_system


def projected_polynomial(coefficients, tail):
    _, packed, _ = packed_coefficients(coefficients, tail)
    q = len(coefficients[1])
    return (packed[0], tuple(packed[1:1+q]),
            tuple(packed[25+24*i+j] for i in range(q) for j in range(q)))


def old_tail_system(projected, prefix, tail, bits):
    constant = evaluate(projected, bits)
    return constant, tuple(evaluate(projected, bits | (1 << (prefix+j))) ^ constant
                           for j in range(tail))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT/'results/L86_mps_mod8_prefix_native_pack_audit_20261008/report.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prefix-samples', type=int, default=512)
    parser.add_argument('--tail', type=int, default=6)
    parser.add_argument('--prefix-multipliers', type=int, default=8)
    args = parser.parse_args()
    if (args.output.exists() or not 1 <= args.prefix_samples <= 65536
            or not 1 <= args.tail <= 8 or not 0 <= args.prefix_multipliers <= 24):
        parser.error('fresh output, bounded sample count, tail and multiplier count required')
    inputs = json.loads(args.input.read_text())
    if not inputs['complete']:
        parser.error('completed fixture audit required')
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for number, source in enumerate(inputs['actual_cases']):
        plan = prepare_native_system(tuple(source['masks']), 43, tuple(source['wanted']),
                                     tuple(source['negative']), conditioning=False)
        coef = quadratic_coefficients(plan)
        q = len(coef[1]); tail = min(args.tail, q); prefix = q-tail
        axes = tuple(range(min(args.prefix_multipliers, prefix))) + tuple(range(prefix, q))
        tick = perf_counter()
        derived = degree_three_tail_consequences(coef, tail, multiplier_axes=axes)
        construction_seconds = perf_counter()-tick
        reference = all_mod8_indices(coef, capacity=65536)
        if not reference['complete']:
            raise RuntimeError('full-zero-set audit overflow; no exclusion justified')
        if tuple(reference['indices']) != tuple(source['indices']):
            raise RuntimeError('input fixture changed or complete modular sets differ')
        # Check EVERY original modular zero in the full 2^q space, including
        # SAT controls. Each returned certificate was independently replayed.
        for index in reference['indices']:
            bits = index & ((1 << prefix)-1)
            if index >> prefix not in affine_tail_indices(*specialize_tail_system(derived, bits)):
                raise RuntimeError('certified necessary condition lost an original zero')
        rng = Random(860000+number)
        samples = rng.sample(range(1 << prefix), min(args.prefix_samples, 1 << prefix))
        projected = projected_polynomial(coef, tail)
        before_kept = after_kept = before_tails = after_tails = 0
        tick = perf_counter()
        for bits in samples:
            old = affine_tail_indices(*old_tail_system(projected, prefix, tail, bits))
            new = affine_tail_indices(*specialize_tail_system(derived, bits))
            if not set(new).issubset(old):
                raise RuntimeError('derived row-space failed to include the original quotient conditions')
            before_kept += bool(old); after_kept += bool(new)
            before_tails += len(old); after_tails += len(new)
        probe_seconds = perf_counter()-tick
        tick = perf_counter(); exact = run_native_system(plan, 2**23); native_seconds = perf_counter()-tick
        if exact['status'] != source['decision']:
            raise RuntimeError('original exact sign fixture decision changed')
        row = dict(number=number, masks=source['masks'], wanted=source['wanted'], negative=source['negative'],
                   original_decision=exact['status'], q=q, tail=tail, prefix_samples=len(samples),
                   quotient_kept_prefixes=before_kept, crossbred_kept_prefixes=after_kept,
                   quotient_tail_candidates=before_tails, crossbred_tail_candidates=after_tails,
                   complete_original_mod8_zero_count=reference['candidate_count'],
                   complete_original_mod8_zeros_all_retained=True,
                   construction_seconds=construction_seconds, diagnostic_seconds=probe_seconds,
                   original_native_seconds=native_seconds, derived=derived)
        rows.append(row)
        print(json.dumps({k:v for k,v in row.items() if k in (
            'number', 'original_decision', 'quotient_kept_prefixes', 'crossbred_kept_prefixes',
            'construction_seconds', 'original_native_seconds')}), flush=True)
    sources = ['solver/mod8_crossbred.py', 'solver/mps_mod8_prefix.py', 'solver/mod8_model_reference.py',
               'csrc/mod8_model_reference.c', 'experiments/probe_mod8_crossbred.py', 'tests/test_mod8_crossbred.py']
    report = dict(complete=True, rows=rows, input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
                  source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources},
                  scope='Certified necessary conditions, all original modular zeros preserved, sampled prefix comparison only.',
                  claim='No complete support exclusion from prefix samples; no new-class discovery or speed claim.')
    atomic_write_json(args.output/'report.json', report)


if __name__ == '__main__':
    main()
