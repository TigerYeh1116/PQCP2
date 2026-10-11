"""Complete perfect binary L43 ingredient generation; no L86 solution claim.

Negative support D has size 21. P_sign(u)=-1 means |D intersect (D+u)|=10.
Thus D D*=1 in F2[X]/(X43+1). Enumerate ALL cyclic norm roots modulo common
rotation, then check the full integer intersections, retaining every preimage.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from solver.cyclic_support import cyclic_norm_family, iter_family, rotation_representatives
from solver.solution_equivalence import pqcp_equivalence_key


def rotate(mask, shift, d=43):
    return ((mask << shift) | (mask >> (d-shift))) & ((1 << d)-1) if shift else mask


def perfect43_bases():
    family = cyclic_norm_family(1, 43)
    quotient = rotation_representatives(family)
    rows, eligible = [], 0
    for mask in iter_family(quotient):
        if bin(mask).count('1') != 21:
            continue
        eligible += 1
        counts = tuple(bin(mask & rotate(mask, u)).count('1') for u in range(1, 22))
        if counts != (10,)*21:
            continue
        signs = tuple(-1 if mask >> i & 1 else 1 for i in range(43))
        # Independent original sign-PACF check on retained ingredients.
        assert sum(signs) == 1 and all(sum(signs[i]*signs[(i+u) % 43] for i in range(43)) == -1
                                     for u in range(1, 43))
        bits = tuple((1-v)//2 for v in signs)
        rows.append(dict(mask=mask, signs=signs, unit_dihedral_key=pqcp_equivalence_key(bits, bits)))
    return dict(literal_norm_roots=family.size, rotation_orbits=quotient.size,
                eligible_weight21_orbits=eligible, rows=rows, complete=True,
                scope='All perfect L43 binary rows with sum +1, modulo rotation; ingredients only.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    report = perfect43_bases()
    report.update(elapsed=perf_counter()-started,
        inequivalent_under_units=len({tuple(row['unit_dihedral_key']) for row in report['rows']}),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'rows'}))


if __name__ == '__main__':
    main()
