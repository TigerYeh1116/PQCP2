"""Complete existence check of the identical one-flip-O even-positive family.

All sign assignments in every retained C support are solved, with exact full
L86 verification. Unit/rotation representatives reduce ingredients only;
every exceptional shift is retained. Never a global nonexistence claim.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.benchmark_l86_discovery import discovery_key, freeze_sources, verified_inventory
from experiments.search_perfect43_basis import perfect43_bases
from solver.checkpoint import atomic_write_json
from solver.near_perfect_odd_kernel import near_perfect43_kernel_certificate
from solver.perfect_odd_kernel import interleave_from_complementary_trits
from solver.search_runner import append_verified_solution_if_new
from solver.structured_energy import integer_periodic_autocorrelation_profile as pacf, target_profile
from solver.ternary_system_native import prepare_native_system, run_native_system
from solver.verifier import verify_pqcp


def near_perfect_ingredients():
    classes, provenance = {}, []
    source = perfect43_bases()
    for row in source['rows']:
        for p, v in enumerate(row['signs']):
            if v != -1:
                continue
            o = tuple(1 if i == p else x for i, x in enumerate(row['signs']))
            key = min(tuple(o[(q*i+r) % 43] for i in range(43))
                      for q in range(1, 43) for r in range(43))
            if key not in classes:
                classes[key] = dict(parent=tuple(row['signs']), parent_mask=row['mask'], flip=p, o=o)
            representative = classes[key]
            witness = next((q, r) for q in range(1, 43) for r in range(43)
                           if tuple(o[(q*i+r) % 43] for i in range(43)) == representative['o'])
            provenance.append(dict(parent_mask=row['mask'], flip=p,
                                   representative_mask=representative['parent_mask'],
                                   representative_flip=representative['flip'], q=witness[0], r=witness[1]))
    return dict(rows=list(classes.values()), provenance=provenance,
                perfect_basis_count=len(source['rows']), literal_flip_count=len(provenance),
                complete=source['complete'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    tick = perf_counter()
    snapshot = args.output/'source_snapshot'
    freeze_sources(snapshot)
    for name in ('search_same_near_perfect_o.py', 'search_perfect43_basis.py', 'benchmark_l86_discovery.py'):
        destination = snapshot/'experiments'/name
        destination.parent.mkdir(exist_ok=True)
        shutil.copy2(ROOT/'experiments'/name, destination)
    manifest = {str(p.relative_to(snapshot)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(snapshot.rglob('*')) if p.is_file() and p.name != 'source_manifest.json'}
    atomic_write_json(snapshot/'source_manifest.json', manifest)
    inventory = (ROOT/'86.txt').read_bytes() if (ROOT/'86.txt').exists() else b''
    (args.output/'initial_inventory.txt').write_bytes(inventory)
    known = verified_inventory(inventory)
    ingredients = near_perfect_ingredients()
    rows, certificates, keys, first_hit = [], [], set(), None
    for ingredient in ingredients['rows']:
        certificate = near_perfect43_kernel_certificate(ingredient['parent'], ingredient['flip'])
        if certificate['o'] != ingredient['o']:
            raise RuntimeError('ingredient provenance disagrees with certified row')
        certificates.append({k: v for k, v in certificate.items() if k not in ('basis', 'candidates')})
        for candidate in certificate['candidates']:
            g = candidate['word']
            mask = sum(int(v == 0) << i for i, v in enumerate(g))
            weight = bin(mask).count('1')
            pg = tuple(pacf(g))
            for k in range(1, 22):
                wanted = tuple(weight if u == 0 else -pg[u]-certificate['profile'][u]
                               +2*int(u in (k, 43-k)) for u in range(43))
                if sum(wanted) != 81:
                    raise RuntimeError('C sum-square identity failed')
                plan = prepare_native_system((mask, 0), 43, wanted, ((weight-9)//2, 0))
                out = run_native_system(plan, 2**23)
                record = dict(parent_mask=ingredient['parent_mask'], flip=ingredient['flip'],
                              terms=candidate['terms'], k=k, c_support=mask, result=out)
                if out['words'] is not None:
                    c = out['words'][0]
                    if tuple(pacf(c)) != wanted or sum(c) != 9:
                        raise RuntimeError('C witness failed independent integer audit')
                    words = interleave_from_complementary_trits(c, g, ingredient['o'])
                    check = verify_pqcp(*words)
                    if not check.is_valid or tuple(check.profile) != target_profile(86, 2*k, 1):
                        raise RuntimeError('one-flip kernel construction failed independent full target audit')
                    key = discovery_key(*words)
                    if key not in known:
                        if key not in keys and first_hit is None:
                            first_hit = perf_counter()-tick
                        keys.add(key)
                        append_verified_solution_if_new(86, *words, args.output)
                    record['full_words'] = words
                rows.append(record)
        print(json.dumps(dict(parent_mask=ingredient['parent_mask'], flip=ingredient['flip'],
                              kernel_candidates=len(certificate['candidates']), completed_jobs=len(rows))), flush=True)
    report = dict(ingredients=ingredients, certificates=certificates, rows=rows,
                  statuses=dict(Counter(row['result']['status'] for row in rows)),
                  complete=ingredients['complete'] and all(row['result']['status'] != 'UNKNOWN' for row in rows),
                  independently_verified_new_classes=len(keys), first_verified_elapsed=first_hit,
                  elapsed=perf_counter()-tick, source_manifest_sha256=hashlib.sha256(
                      json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
                  scope='ALL even-row signs, all negative-to-positive one-flip perfect-43 O ingredients, '
                        'identical odd rows up to translation; even positive target and even-row sums 9. '
                        'Existence check, not enumeration of all SAT witnesses or all L86 PQCPs.')
    atomic_write_json(args.output/'report.json', report)
    print(json.dumps({k: v for k, v in report.items() if k not in ('ingredients', 'certificates', 'rows')}), flush=True)


if __name__ == '__main__':
    main()
