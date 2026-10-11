"""Bounded unrestricted four-color refinement of verified GS ingredient seeds.

Seeds have audited original-length residual energy, never archived PQCP input.
Reaching a lower energy is not a discovery. Every reported hit passes the
independent full verifier and stronger discovery equivalence accounting.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from experiments.benchmark_l86_discovery import discovery_key
from solver.checkpoint import atomic_write_json
from solver.correlation import full_correlation_profile
from solver.search_runner import append_verified_solution_if_new
from solver.solution_equivalence import pqcp_equivalence_key
from solver.target_profiles import TargetContentProfile, pair_content
from solver.torch_joint_search import binary_joint_rows, search_joint
from solver.verifier import verify_pqcp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT/'results/L86_GS_interleave_20261007/report.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch', type=int, default=1024)
    parser.add_argument('--steps', type=int, default=32768)
    parser.add_argument('--seeds', type=int, nargs='+', default=(123, 456, 789))
    parser.add_argument('--temperatures', type=float, nargs='+', default=(.5, 2., 4.))
    parser.add_argument('--direct-transfers', action='store_true')
    args = parser.parse_args()
    if not torch.backends.mps.is_available():
        raise RuntimeError('Real MPS required')
    if args.batch < 1 or args.steps < 1:
        parser.error('positive bounded work required')
    args.output.mkdir(parents=True, exist_ok=False)
    saved, seen, starts = json.loads(args.input.read_text()), set(), []
    for row in saved['seeds']:
        words = tuple(map(tuple, row['words']))
        # Ingredient quotient is only a seed scheduling choice, not discovery
        # counting or a claim that all bounded centers have been searched.
        key = pqcp_equivalence_key(*words)
        if key in seen:
            continue
        seen.add(key)
        profile = tuple(full_correlation_profile(*words))
        if profile[43] != 0:
            raise RuntimeError('GS seed lacks required half-shift zero')
        for k in range(1, 43, 2):
            if profile[k] != 4:
                continue
            target = TargetContentProfile(86, k, 1, *pair_content(*words))
            binary_joint_rows(words, target)
            starts.append((words, target))
    if not starts:
        raise RuntimeError('no audited ingredient seeds')
    profiles = tuple(starts[i % len(starts)][1] for i in range(args.batch))
    words = tuple(starts[i % len(starts)][0] for i in range(args.batch))
    tick, rows, new_keys, first_hit = perf_counter(), [], set(), None
    for temperature in args.temperatures:
        for seed in args.seeds:
            started = perf_counter()
            out = search_joint(profiles, device=torch.device('mps'), seed=seed, steps=args.steps,
                initial_words=words, initial_temperature=temperature, final_temperature=.1,
                direct_transfers=args.direct_transfers)
            for candidate in out.candidates:
                if not candidate['verified']:
                    continue
                pair = (candidate['A'], candidate['B'])
                if not verify_pqcp(*pair).is_valid:
                    raise RuntimeError('refined candidate failed full verifier')
                key = discovery_key(*pair)
                if key not in new_keys and first_hit is None:
                    first_hit = perf_counter()-tick
                new_keys.add(key)
                append_verified_solution_if_new(86, *pair, args.output)
            row = dict(seed=seed, temperature=temperature, steps=out.steps, elapsed=perf_counter()-started,
                minimum_full_energy=int(out.energies.min().item())//2, accepted=out.accepted,
                transfer_accepted=out.transfer_accepted, candidates=out.candidates, new_classes=len(new_keys))
            rows.append(row)
            report = dict(input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
                batch=args.batch, scheduled_seed_targets=len(starts), direct_transfers=args.direct_transfers, rows=rows,
                first_verified_elapsed=first_hit, independently_verified_new_classes=len(new_keys),
                complete=len(rows) == len(args.seeds)*len(args.temperatures), elapsed=perf_counter()-tick,
                scope='Bounded joint C/G refinement from GS-43-derived, non-PQCP seeds; no convergence guarantee.')
            atomic_write_json(args.output/'report.json', report)
            print(json.dumps({k:v for k,v in row.items() if k != 'candidates'}), flush=True)


if __name__ == '__main__':
    main()
