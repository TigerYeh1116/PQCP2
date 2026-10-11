"""Cross-match a finite catalog of verified 43-row ingredients for odd L86.

Close every input row under unit decimation and rotation. Exact integer PACF
joins first find four-row complementary tuples; odd joins then check relative
rotations. Common-unit/individual-rotation quotients are valid for this family
because all odd target positions and all relative offsets remain available.
"""
import argparse
from collections import defaultdict
from functools import lru_cache
import hashlib
from itertools import combinations_with_replacement
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.benchmark_l86_discovery import discovery_key
from experiments.search_doptimal43_interleave import read_source as read_doptimal
from experiments.search_gs43_interleave import read_source as read_gs
from experiments.search_perfect43_basis import perfect43_bases, rotate
from experiments.search_perfect43_interleave import odd_vectors, packed_profile, signs
from solver.checkpoint import atomic_write_json
from solver.interleaving import interleave_bits, sparse_odd_matches
from solver.search_runner import append_verified_solution_if_new
from solver.structured_energy import target_profile
from solver.verifier import verify_pqcp


def rotation_key(mask):
    return min(rotate(mask, u) for u in range(43))


@lru_cache(maxsize=4096)
def unit_keys(mask):
    return tuple(rotation_key(sum(((mask >> ((q*i) % 43)) & 1) << i for i in range(43))) for q in range(1, 43))


def quad_key(quad):
    transformed = tuple(map(unit_keys, quad))
    return min((a, b, *sorted((c, d))) for a, b, c, d in zip(*transformed))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--limit', type=int, default=4096)
    args = parser.parse_args()
    if args.offset < 0 or args.limit < 1:
        parser.error('nonnegative offset and positive limit required')
    args.output.mkdir(parents=True, exist_ok=False)
    tick = perf_counter()
    source_d = ROOT/'results/L86_doptimal_source_20261007.html'
    source_g = ROOT/'results/L43_GS_source_20261007.html'
    dp, rejected = read_doptimal(source_d)
    gs = read_gs(source_g)
    rows = [row for p in dp for row in p['rows']]+[row for p in gs for row in p['rows']]
    rows += [tuple(r['signs']) for r in perfect43_bases()['rows']]
    catalogs = {16:set(), 18:set(), 21:set()}
    for row in rows:
        weight = row.count(-1)
        if weight in catalogs:
            mask = sum(int(v == -1) << i for i, v in enumerate(row))
            catalogs[weight].update(unit_keys(mask))
    ea, eb, os = (sorted(catalogs[w]) for w in (16, 18, 21))
    profiles = {m:packed_profile(m) for m in set(ea+eb+os)}
    e_index = defaultdict(list)
    for a in ea:
        for b in eb:
            e_index[profiles[a]+profiles[b]].append((a, b))
    ones = sum(1 << (8*i) for i in range(21))
    quads, lookups = set(), 0
    for oa, ob in combinations_with_replacement(os, 2):
        lookups += 1
        for a, b in e_index.get(ones-profiles[oa]-profiles[ob], ()):
            quads.add(quad_key((a, b, oa, ob)))
    quads = sorted(quads)
    atomic_write_json(args.output/'catalog.json', dict(row_counts={w:len(v) for w,v in catalogs.items()},
        quadruples=quads, exact_even_lookups=lookups, complete_catalog_join=True,
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (source_d, source_g)},
        rejected_doptimal_pairs=len(rejected), scope='Finite published-row catalog and its independent unit decimations, retaining all profile preimages. Not all binary 43 rows.'))
    trials, seeds, witnesses, keys = [], [], [], set()
    for index, (a, b, oa, ob) in enumerate(quads[args.offset:args.offset+args.limit], args.offset):
        for left_o, right_o in ((oa, ob),) if oa == ob else ((oa, ob), (ob, oa)):
            left, right = odd_vectors(a, left_o), odd_vectors(b, right_o)
            matched = 0
            for k, da, db in sparse_odd_matches(left, right):
                words = (interleave_bits(signs(a), signs(left_o), da), interleave_bits(signs(b), signs(right_o), db, -1))
                check = verify_pqcp(*words)
                if not check.is_valid or tuple(check.profile) != target_profile(86, k, 1):
                    raise RuntimeError('catalog hit failed independent original-length verification')
                keys.add(discovery_key(*words)); matched += 1
                append_verified_solution_if_new(86, *words, args.output)
                witnesses.append(dict(quad_index=index, k=k, offsets=(da, db), words=words))
            best = None
            for x, dx in left.items():
                for y, dy in right.items():
                    if x[-1] != y[-1]:
                        continue
                    difference = tuple(v-w for v, w in zip(x, y))
                    r = max(range(21), key=lambda i:difference[i])
                    residual = sum(v*v for v in difference)-8*difference[r]+16
                    if residual % 16:
                        raise RuntimeError('catalog energy lattice failed')
                    if best is None or residual < best[0]:
                        best = (residual, 2*r+1, dx[0], dy[0])
            if best is not None:
                residual, k, da, db = best
                words = (interleave_bits(signs(a), signs(left_o), da), interleave_bits(signs(b), signs(right_o), db, -1))
                check = verify_pqcp(*words)
                desired = target_profile(86, k, 1)
                if (check.profile[43] != 0 or sum((x-y)**2 for x,y in zip(check.profile[1:44],desired[1:44])) != residual):
                    raise RuntimeError('catalog seed failed independent full integer energy audit')
                seeds.append(dict(quad_index=index, k=k, sign=1, target_energy=residual//16, words=words))
                seeds.sort(key=lambda s:s['target_energy']); del seeds[64:]
            trials.append(dict(quad_index=index, o_assignment=(left_o,right_o), matches=matched,
                exact_odd_lookups=len(left)*21, minimum_full_energy=None if best is None else best[0]//16))
        if (index-args.offset+1) % 16 == 0 or index+1 == len(quads) or index+1 == args.offset+args.limit:
            report = dict(catalog_quadruples=len(quads), offset=args.offset, processed=min(index-args.offset+1,args.limit),
                complete=args.offset == 0 and index+1 == len(quads), trials=trials, seeds=seeds, witnesses=witnesses,
                independently_verified_new_classes=len(keys), elapsed=perf_counter()-tick,
                scope='Exact selected finite-catalog interleaving families, all odd shifts and relative offsets; not a global L86 exclusion.')
            atomic_write_json(args.output/'report.json',report)
            print(json.dumps({k:v for k,v in report.items() if k not in ('trials','seeds','witnesses')}),flush=True)


if __name__ == '__main__':
    main()
