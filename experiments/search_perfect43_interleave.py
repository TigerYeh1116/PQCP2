"""Exact restricted interleaving: four negative flips in perfect E rows.

Negative even target: O rows remain perfect with opposite signs. Enumerate all
perfect length-43 rows from norm roots, all E flip subsets, and all relative
O rotations. Match even PACF vectors first; only then compute odd correlations.
No short-test absence or hashing collision is treated as a proof.
"""
import argparse
from collections import defaultdict
from functools import lru_cache
from itertools import combinations
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.search_perfect43_basis import perfect43_bases, rotate
from experiments.benchmark_l86_discovery import discovery_key
from solver.checkpoint import atomic_write_json
from solver.search_runner import append_verified_solution_if_new
from solver.structured_energy import target_profile
from solver.verifier import verify_pqcp


def sign_profile(mask):
    weight = bin(mask).count('1')
    return tuple(43-4*weight+4*bin(mask & rotate(mask, u)).count('1') for u in range(1, 22))


def signs(mask):
    return tuple(-1 if mask >> i & 1 else 1 for i in range(43))


def packed_profile(mask):
    """Exact additive encoding, not a probabilistic fingerprint.

    P(u)=-1 mod 4. Each normalized coefficient is in [-10,11]. A four-row
    difference has coefficients of magnitude <256, so equality of their
    signed base-256 polynomial implies equality in every coordinate.
    """
    return sum(((value+1)//4) << (8*i) for i, value in enumerate(sign_profile(mask)))


@lru_cache(maxsize=256)
def odd_vectors(e_mask, o_mask):
    e, o = signs(e_mask), signs(o_mask)
    cross = tuple(sum(e[(i+t) % 43]*o[i] for i in range(43)) for t in range(43))
    groups = defaultdict(list)
    for delta in range(43):
        vector = tuple(cross[(r+1-delta) % 43]+cross[(-r-delta) % 43] for r in range(22))
        groups[vector].append(delta)
    return dict(groups)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--eta', type=int, choices=(-1, 1), default=-1)
    parser.add_argument('--stop-after-new', type=int, default=0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    basis = perfect43_bases()
    o_masks = tuple(row['mask'] for row in basis['rows'])
    index = defaultdict(list)
    masks = set()
    for mask in o_masks:
        positions = tuple(i for i in range(43) if mask >> i & 1)
        for flipped in combinations(positions, 4):
            candidate = mask ^ sum(1 << i for i in flipped)
            if candidate in masks:
                continue
            masks.add(candidate)
            index[packed_profile(candidate)].append(candidate)
    o_words = set(o_masks)
    if args.eta == 1:
        o_words = {mask ^ (1 << i) for mask in o_masks for i in range(43) if mask >> i & 1}
    o_index = defaultdict(list)
    for mask in sorted(o_words):
        o_index[packed_profile(mask)].append(mask)
    expected = sum((1+args.eta if i == 0 else 1) << (8*i) for i in range(21))
    groups = list(o_index)
    matches, odd_tests, rows, new_keys, stopped = 0, 0, [], set(), False
    first_verified_elapsed = None
    lookups = 0
    e_items = list(index.items())
    for i, op_a in enumerate(groups):
        for op_b in groups[i:]:
            needed = expected-op_a-op_b
            for profile, a_masks in e_items:
                partner = needed-profile
                lookups += 1
                if profile > partner:
                    continue
                for a in a_masks:
                    for b in index.get(partner, ()):
                        if profile == partner and a > b:
                            continue
                        matches += 1
                        assignments = [(op_a, op_b)] if op_a == op_b else [(op_a, op_b), (op_b, op_a)]
                        for left_group, right_group in assignments:
                            for oa in o_index[left_group]:
                                left = odd_vectors(a, oa)
                                for ob in o_index[right_group]:
                                    right = odd_vectors(b, ob)
                                    odd_tests += 1
                                    for vector in left.keys() & right.keys():
                                        for da in left[vector]:
                                            for db in right[vector]:
                                                ea, eb, ova, ovb = signs(a), signs(b), signs(oa), signs(ob)
                                                x = tuple(v for i in range(43) for v in (ea[i], ova[(i+da) % 43]))
                                                y = tuple(v for i in range(43) for v in (eb[i], -ovb[(i+db) % 43]))
                                                words = tuple(tuple((1-v)//2 for v in w) for w in (x, y))
                                                check = verify_pqcp(*words)
                                                if not check.is_valid or tuple(check.profile) != target_profile(86, 2, args.eta):
                                                    raise RuntimeError('interleave witness failed independent full target verification')
                                                if first_verified_elapsed is None:
                                                    first_verified_elapsed = perf_counter()-started
                                                new = append_verified_solution_if_new(86, *words, args.output)
                                                rows.append(dict(e_masks=(a, b), o_masks=(oa, ob), offsets=(da, db), words=words, new_legacy_class=new))
                                                new_keys.add(discovery_key(*words))
                if args.stop_after_new and len(new_keys) >= args.stop_after_new:
                    stopped = True; break
            if stopped: break
        if stopped: break
        if i % 8 == 0:
            print(json.dumps(dict(o_groups_done=i+1, o_groups_total=len(groups), lookups=lookups,
                                 even_matches=matches, new_classes=len(new_keys), elapsed=perf_counter()-started)), flush=True)
    report = dict(basis_count=len(o_masks), e_configurations=len(masks), e_profile_groups=len(index), eta=args.eta,
        o_configurations=len(o_words), o_profile_groups=len(o_index), exact_profile_lookups=lookups,
        first_verified_elapsed=first_verified_elapsed, independently_verified_classes=len(new_keys),
        even_matched_pairs=matches, odd_profile_pair_tests=odd_tests, verified_models=len(rows), rows=rows,
        elapsed=perf_counter()-started, complete=not stopped,
        scope='Perfect-43 E(+4 flips)/O(0 flips if eta=-1; 1 if eta=+1), even target; not all L86 PQCPs.',
        claim='Ingredient/matching counts are not a new-discovery speed claim.')
    atomic_write_json(args.output/'report.json', report)
    print(json.dumps({k:v for k,v in report.items() if k != 'rows'}))


if __name__ == '__main__':
    main()
