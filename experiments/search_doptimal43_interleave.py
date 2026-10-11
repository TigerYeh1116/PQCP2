"""Verified D-optimal-43/perfect-43 interleaving, an exact restricted search.

The source pairs have row sums 11/7 and summed nonzero PACF 2. Perfect O
rows have sum 1 and PACF -1. Interleave E_A/O_A and E_B/-O_B: every even
sidelobe is identically zero. Join the remaining odd vectors exactly.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.benchmark_l86_discovery import discovery_key
from experiments.search_perfect43_basis import perfect43_bases
from solver.checkpoint import atomic_write_json
from solver.interleaving import interleave_bits, odd_cross_vectors, periodic_sign_profile, sparse_odd_matches
from solver.search_runner import append_verified_solution_if_new
from solver.structured_energy import target_profile
from solver.verifier import verify_pqcp

SOURCE_URL = 'https://documents.uow.edu.au/~jennie/D-optimal/order.86.html'


def read_source(path):
    text = path.read_text()
    entries = re.findall(r'([AB])_(\d+)=\{([0-9,]+)\}', text)
    if len(entries) != 34:
        raise ValueError('expected 17 complete source pairs')
    pairs, rejected = [], []
    for index in range(0, len(entries), 2):
        a, b = entries[index:index+2]
        if a[0] != 'A' or b[0] != 'B' or a[1] != b[1]:
            raise ValueError('source row pairing is ambiguous')
        supports = [tuple(map(int, item[2].split(','))) for item in (a, b)]
        if any(len(set(s)) != len(s) or any(not 0 <= i < 43 for i in s) for s in supports):
            raise ValueError('malformed source support')
        expected = (15, 21) if index < 24 else (16, 18)
        if tuple(map(len, supports)) != expected:
            raise ValueError('source support weights differ from declared parameters')
        rows = tuple(tuple(-1 if i in s else 1 for i in range(43)) for s in supports)
        profiles = tuple(map(periodic_sign_profile, rows))
        total = tuple(x+y for x, y in zip(*profiles))
        if total != (86,)+(2,)*42:
            rejected.append(dict(source_index=int(a[1]), weights=expected, rows=rows,
                observed_profile=total, reason='fails independent integer D-optimal identity'))
            continue
        pairs.append(dict(source_index=int(a[1]), weights=expected, rows=rows, origin='validated source'))
    return pairs, rejected


def repair_rejected(pairs):
    """Complete one-swap neighborhood, retaining only exact integer witnesses."""
    recovered = []
    for pair in pairs:
        for side in range(2):
            original = pair['rows'][side]
            for i in range(43):
                if original[i] != -1:
                    continue
                for j in range(43):
                    if original[j] != 1:
                        continue
                    modified = tuple(-v if k in (i, j) else v for k, v in enumerate(original))
                    rows = tuple(modified if k == side else row for k, row in enumerate(pair['rows']))
                    profiles = tuple(map(periodic_sign_profile, rows))
                    if tuple(x+y for x, y in zip(*profiles)) == (86,)+(2,)*42:
                        recovered.append(dict(source_index=pair['source_index'], weights=pair['weights'], rows=rows,
                            origin='independently verified one-swap repair', side=side, swap=(i, j)))
    return recovered


def join_index(e, o_rows):
    result = defaultdict(list)
    for index, o in enumerate(o_rows):
        for vector, offsets in odd_cross_vectors(e, o).items():
            result[vector].extend((index, offset) for offset in offsets)
    return dict(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'results/L86_doptimal_source_20261007.html')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    pairs, rejected = read_source(args.source)
    recovered = repair_rejected(rejected)
    basis = perfect43_bases()
    o_rows = tuple(tuple(row['signs']) for row in basis['rows'])
    trials, witnesses, new_keys, first_hit = [], [], set(), None
    for pair in pairs+recovered:
        if pair['weights'] != (16, 18):
            continue
        for reverse_a in (False, True):
            for reverse_b in (False, True):
                e = tuple(tuple(row[(-i) % 43] for i in range(43)) if reverse else row
                          for row, reverse in zip(pair['rows'], (reverse_a, reverse_b)))
                left, right = (join_index(row, o_rows) for row in e)
                matched = 0
                for k, (ia, da), (ib, db) in sparse_odd_matches(left, right):
                    words = (interleave_bits(e[0], o_rows[ia], da), interleave_bits(e[1], o_rows[ib], db, -1))
                    check = verify_pqcp(*words)
                    if not check.is_valid or tuple(check.profile) != target_profile(86, k, 1):
                        raise RuntimeError('interleave hit failed independent full-length verifier')
                    if first_hit is None:
                        first_hit = perf_counter()-started
                    new_keys.add(discovery_key(*words))
                    legacy_new = append_verified_solution_if_new(86, *words, args.output)
                    matched += 1
                    witnesses.append(dict(source_index=pair['source_index'], reflections=(reverse_a, reverse_b),
                        o_indices=(ia, ib), offsets=(da, db), k=k, words=words, new_legacy_class=legacy_new))
                trials.append(dict(source_index=pair['source_index'], origin=pair['origin'], reflections=(reverse_a, reverse_b),
                    left_groups=len(left), right_groups=len(right), exact_lookups=len(left)*21, matches=matched))
    report = dict(source_url=SOURCE_URL, source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), validated_source_pairs=len(pairs),
        rejected_source_pairs=rejected, independently_repaired_pairs=recovered,
        perfect_43_rotation_classes=len(o_rows), trials=trials, witnesses=witnesses, complete=True,
        first_verified_elapsed=first_hit, independently_verified_new_classes=len(new_keys),
        elapsed=perf_counter()-started, exact_lookups=sum(row['exact_lookups'] for row in trials),
        scope='Validated published D-optimal 43 row-sum 11/7 pairs and verified one-swap repairs, both independent reflections; all perfect O rows/relative rotations; positive odd target. Not all L86 PQCPs.',
        claim='No discovery speed claim without verified new full solutions and a paired baseline.')
    atomic_write_json(args.output/'report.json', report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('trials', 'witnesses')}), flush=True)


if __name__ == '__main__':
    main()
