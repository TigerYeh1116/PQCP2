"""Literal binary/PACF audit of the GS fixed-support energy lower bound."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from solver.checkpoint import atomic_write_json
from solver.correlation import full_correlation_profile
from solver.cyclic_support_blocks import support_parity_energy_lower_bound
from solver.structured_energy import target_profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT/'results/L86_GS_interleave_20261007/report.json')
    parser.add_argument('--output', type=Path, default=ROOT/'results'/(
        'L86_support_energy_bound_ide_'+datetime.now().strftime('%Y%m%d_%H%M%S')))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = json.loads(args.input.read_text())
    records = []
    for seed in source['seeds']:
        words = tuple(map(tuple, seed['words']))
        d = len(words[0])//2
        if d != 43 or any(len(w) != 86 for w in words):
            raise ValueError('full length-86 binary GS bank required')
        masks = tuple(sum(int(w[i] == w[i+d]) << i for i in range(d)) for w in words)
        if sum(bin(m).count('1') for m in masks) != d:
            raise ValueError('GS center does not satisfy half-shift zero')
        uwords = tuple(tuple(int(m >> i & 1) for i in range(d)) for m in masks)
        literal_support_pacf = tuple(sum(w[i]*w[(i+u) % d] for w in uwords for i in range(d))
                                     for u in range(d))
        full_target = target_profile(86, seed['k'], seed['sign'])
        coarse_target = tuple((full_target[u]+full_target[u+d])//4 for u in range(d))
        mismatched = tuple(u for u in range(1, 22)
                           if (literal_support_pacf[u]-coarse_target[u]) % 2)
        bound = support_parity_energy_lower_bound(masks, d, min(seed['k'], d-seed['k']))
        if bound != len(mismatched):
            raise RuntimeError('packed support bound disagrees with literal unsigned PACF')
        profile = tuple(full_correlation_profile(*words))
        residual_squared = sum((profile[u]-full_target[u])**2 for u in range(1, 44))
        if residual_squared % 16 or residual_squared < 16*bound:
            raise RuntimeError('full binary energy violates fixed-support lower bound')
        energy = residual_squared//16
        records.append(dict(words=words, k=seed['k'], eta=seed['sign'], masks=masks,
                            parity_mismatches=mismatched, unsigned_pacf=literal_support_pacf,
                            full_profile=profile, full_energy=energy, lower_bound=bound,
                            fixed_support_optimum_certified=energy == bound))
    report = dict(rows=records, complete=True,
                  all_centers_attain_fixed_support_lower_bound=all(r['fixed_support_optimum_certified'] for r in records),
                  input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  bound_module_sha256=hashlib.sha256((ROOT/'solver/cyclic_support_blocks.py').read_bytes()).hexdigest(),
                  scope='Full literal support parity and binary energies. Bound holds for every sign assignment '
                        'on these fixed supports and contents. Not a global lower bound or PQCP discovery.')
    atomic_write_json(args.output/'report.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'rows'}), flush=True)


if __name__ == '__main__':
    main()
