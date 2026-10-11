"""Explore parity-compatible support blocks with all C/G signs free.

Starts are the nine independently found coarse witnesses, never full PQCPs.
Every literal parity match in a chosen block is completed at its unique
feasible content. Complete UNSAT orbits may be cached; UNKNOWN is retained.
Walking to a new coarse witness is only navigation, not a full discovery.
"""
import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.benchmark_l86_discovery import discovery_key, freeze_sources, verified_inventory
from solver.checkpoint import atomic_write_json
from solver.cyclic_support_blocks import join_support_blocks
from solver.pqcp_support_solver import complete_pqcp_supports
from solver.search_runner import append_verified_solution_if_new
from solver.structured_energy import integer_periodic_autocorrelation_profile as pacf, fold_profile, target_profile
from solver.support_orbit_cache import FULL_SUPPORT_ORBITS
from solver.target_profiles import TargetContentProfile
from solver.ternary_support_search import support_mask
from solver.torch_compressed_search import search_paired_contents


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT/'results/L86_independent_norm_long_20261007/coarse_witnesses.json')
    parser.add_argument('--gs-input', type=Path)
    parser.add_argument('--gs-only', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT/'results'/(
        'L86_support_blocks_ide_'+datetime.now().strftime('%Y%m%d_%H%M%S')))
    parser.add_argument('--block-size', type=int, default=16)
    parser.add_argument('--sweeps', type=int, default=8)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--max-assignments', type=int, default=2**23)
    parser.add_argument('--orbit-pruning', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--modulation-orbit-pruning',action=argparse.BooleanOptionalAction,default=False)
    args = parser.parse_args()
    if not 1 <= args.block_size <= 20 or args.sweeps < 1 or not 1 <= args.max_assignments <= 2**31:
        parser.error('bounded positive block size/sweeps/assignment budget required')
    if args.gs_only and args.gs_input is None:
        parser.error('--gs-only requires a GS seed bank')
    if args.modulation_orbit_pruning and not args.orbit_pruning:
        parser.error('modulation orbit pruning requires the proven UNSAT orbit cache')
    args.output.mkdir(parents=True, exist_ok=False)
    tick = perf_counter()
    snapshot = args.output/'source_snapshot'
    freeze_sources(snapshot)
    for name in ('search_support_blocks.py', 'search_joint_blocks.py', 'benchmark_l86_discovery.py'):
        destination = snapshot/'experiments'/name
        destination.parent.mkdir(exist_ok=True)
        shutil.copy2(ROOT/'experiments'/name, destination)
    manifest = {str(p.relative_to(snapshot)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(snapshot.rglob('*')) if p.is_file() and p.name != 'source_manifest.json'}
    atomic_write_json(snapshot/'source_manifest.json', manifest)
    if not args.gs_only:
        shutil.copy2(args.input, args.output/'input_coarse_witnesses.json')
    if args.gs_input is not None:
        shutil.copy2(args.gs_input, args.output/'input_gs_seeds.json')
    inventory = (ROOT/'86.txt').read_bytes() if (ROOT/'86.txt').exists() else b''
    (args.output/'initial_inventory.txt').write_bytes(inventory)
    known = verified_inventory(inventory)
    saved = {'rows': []} if args.gs_only else json.loads(args.input.read_text())
    starts = []
    for row in saved['rows']:
        target = TargetContentProfile(**row['target'])
        words = tuple(map(tuple, row['result']['attempts']['coarse']['words']))
        folded = fold_profile(target_profile(86, target.k, target.eta), 2)
        if any(value % 4 for value in folded):
            raise RuntimeError('compressed target is not integral in trit units')
        wanted = tuple(value//4 for value in folded)
        if tuple(x+y for x, y in zip(pacf(words[0]), pacf(words[1]))) != tuple(wanted):
            raise RuntimeError('starting coarse witness failed original integer audit')
        contents = search_paired_contents(target)
        by_weights = {tuple(n+p for n, z, p in pair): pair for pair in contents}
        if len(by_weights) != len(contents):
            raise RuntimeError('support weights no longer uniquely determine feasible contents')
        masks = tuple(support_mask(w) for w in words)
        starts.append(dict(target=target, masks=masks, by_weights=by_weights, trial=row['trial'],
                           visited={masks}))
    if args.gs_input is not None:
        from experiments.search_joint_blocks import normalize_start
        from solver.solution_equivalence import pqcp_equivalence_key
        from solver.torch_joint_search import binary_joint_rows
        from solver.cyclic_support_blocks import support_parity_energy_lower_bound
        seen = set()
        for number, row in enumerate(json.loads(args.gs_input.read_text())['seeds']):
            binary, target = normalize_start(row['words'], row['k'], row['sign'])
            key = pqcp_equivalence_key(*binary)
            if key in seen:
                continue
            seen.add(key)
            coarse, difference = binary_joint_rows(binary, target)
            masks = tuple(support_mask(w) for w in coarse)
            bound = support_parity_energy_lower_bound(masks, 43, min(target.k, 43-target.k))
            contents = search_paired_contents(target)
            by_weights = {tuple(n+p for n, z, p in pair): pair for pair in contents}
            if len(by_weights) != len(contents):
                raise RuntimeError('GS support weights no longer uniquely determine feasible contents')
            starts.append(dict(target=target, masks=masks, by_weights=by_weights,
                               trial='GS:%d:fixed_support_bound=%d' % (number, bound), visited={masks}))
    if not starts:
        raise ValueError('at least one independently audited coarse or binary non-solution start required')
    FULL_SUPPORT_ORBITS.clear()
    rng = random.Random(args.seed)
    records, keys, statuses, first_hit = [], set(), Counter(), None
    expected_blocks = args.sweeps*len(starts)
    with (args.output/'literal_completions.jsonl').open('w') as log:
        for sweep in range(args.sweeps):
            for lane, start in enumerate(starts):
                target = start['target']
                positions = tuple(tuple(sorted(rng.sample(range(43), args.block_size))) for side in range(2))
                joined = join_support_blocks(start['masks'], 43, min(target.k, 43-target.k),
                                             positions, tuple(start['by_weights']))
                block_statuses, new_coarse, unknown, orbit_hits = Counter(), [], 0, 0
                for masks in joined['masks']:
                    weights = tuple(bin(m).count('1') for m in masks)
                    pair = start['by_weights'][weights]
                    reference = tuple(tuple([-1]*n+[0]*z+[1]*p) for n, z, p in pair)
                    out = complete_pqcp_supports(*reference, target, masks,
                                                max_assignments=args.max_assignments,
                                                orbit_pruning=args.orbit_pruning,
                                                modulation_orbit_pruning=args.modulation_orbit_pruning)
                    statuses[out['status']] += 1
                    block_statuses[out['status']] += 1
                    unknown += int(out['status'] == 'UNKNOWN')
                    orbit_hits += int(out['excluded_by'] == 'dihedral_orbit')
                    coarse = out['attempts'].get('coarse', {})
                    if coarse.get('status') == 'SAT_COMPRESSED' and masks not in start['visited']:
                        new_coarse.append(masks)
                        start['visited'].add(masks)
                    if out['verified']:
                        key = discovery_key(*out['words'])
                        if key not in known:
                            if key not in keys and first_hit is None:
                                first_hit = perf_counter()-tick
                            keys.add(key)
                            append_verified_solution_if_new(86, *out['words'], args.output)
                    log.write(json.dumps(dict(sweep=sweep, lane=lane, target=asdict(target),
                                              masks=masks, contents=pair, result=out))+'\n')
                if new_coarse:
                    start['masks'] = rng.choice(new_coarse)
                block = dict(sweep=sweep, lane=lane, source_trial=start['trial'], positions=positions,
                             matched_literal_pairs=len(joined['masks']),
                             covered_row_configurations=joined['covered_row_configurations'],
                             retained_row_configurations=joined['row_retained_configurations'],
                             statuses=dict(block_statuses), new_coarse_supports=len(new_coarse),
                             unknown=unknown, orbit_hits=orbit_hits, next_masks=start['masks'],
                             support_join_seconds=joined['elapsed'])
                records.append(block)
                log.flush()
                report = dict(blocks=records, expected_blocks=expected_blocks,
                              complete=len(records) == expected_blocks,
                              all_selected_support_completions_decided=all(r['unknown'] == 0 for r in records),
                              statuses=dict(statuses), seed=args.seed, block_size=args.block_size,
                              orbit_pruning=args.orbit_pruning, independently_verified_new_classes=len(keys),
                              modulation_orbit_pruning=args.modulation_orbit_pruning,
                              first_verified_elapsed=first_hit, elapsed=perf_counter()-tick,
                              source_manifest_sha256=hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
                              input_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                            for p in (([] if args.gs_only else [args.input])
                                                      +([] if args.gs_input is None else [args.gs_input]))},
                              scope='All parity-compatible literal support pairs in selected blocks; all signs free. '
                                    'Warm coarse starts; supplied input generation excluded from elapsed. '
                                    'Neither misses nor coarse witnesses establish global exclusion or discovery speed.')
                atomic_write_json(args.output/'report.json', report)
                print(json.dumps(dict(completed_blocks=len(records), **block,
                                      new_classes=len(keys), elapsed=perf_counter()-tick)), flush=True)


if __name__ == '__main__':
    main()
