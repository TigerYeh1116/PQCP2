"""Exact odd-target interleaving of independently verified GS-43 ingredients.

Four-row summed PACF zero makes all even sidelobes zero. Enumerate relative
rotations, independent reflections and the two sum-1 row assignments; join
odd profiles exactly. Also retain low-energy half-shift-zero seeds for repair.
"""
import argparse
import hashlib
from html.parser import HTMLParser
from itertools import product
import json
from pathlib import Path
import re
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.benchmark_l86_discovery import discovery_key
from solver.checkpoint import atomic_write_json
from solver.correlation import full_correlation_profile
from solver.interleaving import interleave_bits, odd_cross_vectors, periodic_sign_profile, sparse_odd_matches
from solver.search_runner import append_verified_solution_if_new
from solver.structured_energy import target_profile
from solver.verifier import verify_pqcp

SOURCE_URL = 'https://arxiv.org/html/1802.00556v1'


class SourceText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, value):
        self.parts.append(value)


def read_source(path):
    parser = SourceText()
    parser.feed(path.read_text())
    text = ''.join(parser.parts)
    start = text.rfind('7.6')
    end = text.find('7.7', start)
    if start < 0 or end < 0:
        raise ValueError('missing GS-43 appendix')
    blocks = re.findall(r'\[([0-9,]+)\]', text[start:end])
    if len(blocks) != 40:
        raise ValueError('expected ten complete GS-43 quadruples')
    families = []
    for index in range(0, len(blocks), 4):
        supports = [tuple(map(int, block.split(','))) for block in blocks[index:index+4]]
        if any(len(set(s)) != len(s) or any(not 0 <= v < 43 for v in s) for s in supports):
            raise ValueError('malformed GS support')
        rows = tuple(tuple(-1 if i in s else 1 for i in range(43)) for s in supports)
        total = tuple(map(sum, zip(*map(periodic_sign_profile, rows))))
        if total != (172,)+(0,)*42:
            raise ValueError('GS ingredient fails independent integer summed-PACF audit')
        families.append(dict(appendix_index=index//4+1, rows=rows, sums=tuple(map(sum, rows))))
    return families


def reflections(row):
    return tuple(dict.fromkeys((row, tuple(row[(-i) % 43] for i in range(43)))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'results/L43_GS_source_20261007.html')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    families = read_source(args.source)
    trials, witnesses, seeds, keys, first_hit = [], [], [], set(), None
    for family in families:
        if sorted(family['sums']) != [1, 1, 7, 11]:
            continue
        ea, eb = (family['rows'][family['sums'].index(total)] for total in (11, 7))
        os = [row for row in family['rows'] if sum(row) == 1]
        for assignment in (os, os[::-1]):
            for e_a, e_b, o_a, o_b in product(*map(reflections, (ea, eb, *assignment))):
                left, right = odd_cross_vectors(e_a, o_a), odd_cross_vectors(e_b, o_b)
                matched = 0
                for k, da, db in sparse_odd_matches(left, right):
                    words = (interleave_bits(e_a, o_a, da), interleave_bits(e_b, o_b, db, -1))
                    check = verify_pqcp(*words)
                    if not check.is_valid or tuple(check.profile) != target_profile(86, k, 1):
                        raise RuntimeError('GS interleave hit failed independent full verifier')
                    if first_hit is None:
                        first_hit = perf_counter()-started
                    keys.add(discovery_key(*words))
                    matched += 1
                    witnesses.append(dict(appendix_index=family['appendix_index'], offsets=(da, db),
                        k=k, words=words, new_legacy_class=append_verified_solution_if_new(86, *words, args.output)))
                best = None
                for a, das in left.items():
                    for b, dbs in right.items():
                        if a[-1] != b[-1]:
                            continue
                        difference = tuple(x-y for x, y in zip(a, b))
                        r = max(range(21), key=lambda i: difference[i])
                        residual = sum(v*v for v in difference)-8*difference[r]+16
                        if residual % 16:
                            raise RuntimeError('binary energy lattice failed')
                        if best is None or residual < best[0]:
                            best = (residual, 2*r+1, das[0], dbs[0])
                if best is not None:
                    residual, k, da, db = best
                    words = (interleave_bits(e_a, o_a, da), interleave_bits(e_b, o_b, db, -1))
                    profile = tuple(full_correlation_profile(*words))
                    desired = target_profile(86, k, 1)
                    if profile[43] != 0 or sum((x-y)**2 for x, y in zip(profile[1:44], desired[1:44])) != residual:
                        raise RuntimeError('nearby seed failed independent original-length audit')
                    seeds.append(dict(appendix_index=family['appendix_index'], k=k, sign=1,
                                      target_energy=residual//16, words=words))
                trials.append(dict(appendix_index=family['appendix_index'], exact_lookups=len(left)*21,
                                   matches=matched, minimum_full_energy=None if best is None else best[0]//16))
    report = dict(source_url=SOURCE_URL, source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), validated_quadruples=len(families),
        eligible_quadruples=sum(sorted(f['sums']) == [1, 1, 7, 11] for f in families),
        trials=trials, witnesses=witnesses, seeds=sorted(seeds, key=lambda s:s['target_energy']), complete=True,
        exact_lookups=sum(t['exact_lookups'] for t in trials), independently_verified_new_classes=len(keys),
        first_verified_elapsed=first_hit, elapsed=perf_counter()-started,
        scope='Published GS-43 quadruples with sums 11/7/1/1 and their independent reflections/relative rotations. Not all L86 PQCPs.',
        claim='Low-energy ingredients are seeds, not full solutions or a discovery speed claim.')
    atomic_write_json(args.output/'report.json', report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('trials', 'witnesses', 'seeds')}), flush=True)
    print(json.dumps(dict(minimum_seed_energy=min((s['target_energy'] for s in seeds), default=None))))


if __name__ == '__main__':
    main()
