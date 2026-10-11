"""Independent joint-half-shift search followed by exact cyclic full repair.

Never seeds from archived PQCPs. Every zero or native full witness is audited
and archived only in the isolated output. UNKNOWN/misses remain explicit.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from experiments.benchmark_l86_discovery import discovery_key
from solver.checkpoint import atomic_write_json
from solver.cyclic_support_search import repair_cyclic_supports
from solver.search_runner import append_verified_solution_if_new
from solver.target_profiles import canonical_target_content_profiles
from solver.torch_joint_search import search_joint
from solver.verifier import verify_pqcp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=(123, 456, 789))
    parser.add_argument('--batch', type=int, default=2048)
    parser.add_argument('--steps', type=int, default=8192)
    parser.add_argument('--cyclic-top', type=int, default=32)
    parser.add_argument('--cyclic-max', type=int, default=8)
    args = parser.parse_args()
    if args.batch <= 0 or args.steps <= 0 or args.cyclic_top < 0 or not 1 <= args.cyclic_max <= 4096:
        parser.error('positive bounded work required')
    args.output.mkdir(parents=True, exist_ok=False)
    if not torch.backends.mps.is_available():
        raise RuntimeError('Real MPS required')
    targets = canonical_target_content_profiles(86)
    profiles = tuple(targets[i % len(targets)] for i in range(args.batch))
    started, rows, new_keys, first_hit = perf_counter(), [], set(), None
    def record(words):
        nonlocal first_hit
        if not verify_pqcp(*words).is_valid:
            raise RuntimeError('refusing unverified joint search result')
        key = discovery_key(*words)
        if key not in new_keys and first_hit is None:
            first_hit = perf_counter()-started
        new_keys.add(key)
        return append_verified_solution_if_new(86, *words, args.output)
    for seed in args.seeds:
        tick = perf_counter()
        result = search_joint(profiles, device=torch.device('mps'), seed=seed, steps=args.steps,
                              export_limit=max(128, args.cyclic_top))
        for candidate in result.candidates:
            if candidate['verified']:
                record((candidate['A'], candidate['B']))
        c = result.coarse.int().cpu().tolist()
        energies = result.energies.int().cpu().tolist()
        repairs = []
        for lane in sorted(range(args.batch), key=lambda i: (energies[i], i))[:args.cyclic_top]:
            out = repair_cyclic_supports(tuple(c[lane][0]), tuple(c[lane][1]), profiles[lane],
                                        max_supports=args.cyclic_max, full_completion=True)
            if out['full_words'] is not None:
                record(out['full_words'])
            repairs.append(dict(lane=lane, joint_energy=energies[lane], target=asdict(profiles[lane]), result=out))
        row = dict(seed=seed, elapsed=perf_counter()-tick, steps=result.steps,
            minimum_joint_energy=min(energies), minimum_full_energy=min(energies)//2,
            accepted=result.accepted, transfer_accepted=result.transfer_accepted,
            candidates=result.candidates, repairs=repairs, cumulative_new_classes=len(new_keys))
        rows.append(row)
        report = dict(batch=args.batch, steps=args.steps, rows=rows, complete=len(rows)==len(args.seeds),
            first_verified_elapsed=first_hit, independently_verified_new_classes=len(new_keys),
            elapsed=perf_counter()-started, scope='bounded joint C/G search and ranked fixed-axis full support repair')
        atomic_write_json(args.output/'report.json', report)
        print(json.dumps({k:v for k,v in row.items() if k not in ('candidates', 'repairs')}), flush=True)


if __name__ == '__main__':
    main()
