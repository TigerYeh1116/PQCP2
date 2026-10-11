"""Exact energy-4 error shell around audited C witnesses, never solution seeds.

The smallest nonzero G error, with fixed contents and supports, is +2/-2 at
two independent lags. Solve ALL 420 such profiles before asserting an energy
lower bound. Every near witness is audited at the original binary length.
"""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.benchmark_l86_discovery import freeze_sources
from solver.checkpoint import atomic_write_json
from solver.correlation import full_correlation_profile
from solver.near_difference_profiles import minimum_difference_error_shell
from solver.odd_lift_native import difference_sign_problem
from solver.structured_energy import fold_profile, integer_periodic_autocorrelation_profile as pacf, target_profile
from solver.target_profiles import TargetContentProfile
from solver.ternary_support_search import support_mask
from solver.ternary_system_native import prepare_native_system, run_native_system
from solver.torch_joint_blocks import joint_integer_energy
from solver.torch_joint_search import joint_binary_words
from solver.verifier import verify_pqcp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT/'results/L86_independent_norm_long_20261007/coarse_witnesses.json')
    parser.add_argument('--output', type=Path, default=ROOT/'results'/('L86_near_G_shell_ide_'+datetime.now().strftime('%Y%m%d_%H%M%S')))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    tick = perf_counter()
    snapshot = args.output/'source_snapshot'
    freeze_sources(snapshot)
    for name in ('search_near_difference_shell.py', 'benchmark_l86_discovery.py'):
        destination = snapshot/'experiments'/name
        destination.parent.mkdir(exist_ok=True)
        shutil.copy2(ROOT/'experiments'/name, destination)
    manifest = {str(p.relative_to(snapshot)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(snapshot.rglob('*')) if p.is_file() and p.name != 'source_manifest.json'}
    atomic_write_json(snapshot/'source_manifest.json', manifest)
    shutil.copy2(args.input, args.output/'input_coarse_witnesses.json')
    source = json.loads(args.input.read_text())
    records, candidates = [], []
    with (args.output/'literal_shell_jobs.jsonl').open('w') as log:
        for row in source['rows']:
            target = TargetContentProfile(**row['target'])
            coarse = tuple(map(tuple, row['result']['attempts']['coarse']['words']))
            c_target = tuple(v//4 for v in fold_profile(target_profile(86, target.k, target.eta), 2))
            if tuple(x+y for x, y in zip(pacf(coarse[0]), pacf(coarse[1]))) != c_target:
                raise RuntimeError('C starting witness failed independent full PACF audit')
            c_masks = tuple(support_mask(w) for w in coarse)
            masks, wanted, negative = difference_sign_problem(c_masks, target)
            exact = run_native_system(prepare_native_system(masks, 43, wanted, negative), 2**23)
            if exact['status'] == 'SAT_COMPRESSED':
                raise RuntimeError('source G is feasible; complete the full PQCP rather than classify it as a near miss')
            counts, found = Counter(), 0
            for plus, minus, nearby in minimum_difference_error_shell(wanted):
                out = run_native_system(prepare_native_system(masks, 43, nearby, negative), 2**23)
                counts[out['status']] += 1
                job = dict(source_trial=row['trial'], plus=plus, minus=minus, result=out)
                if out['words'] is not None:
                    g = out['words']
                    if (tuple(x+y for x, y in zip(pacf(g[0]), pacf(g[1]))) != nearby
                            or tuple(w.count(-1) for w in g) != negative
                            or tuple(support_mask(w) for w in g) != masks):
                        raise RuntimeError('near G witness failed independent sign-profile audit')
                    words = joint_binary_words(coarse, g, target)
                    profile = tuple(full_correlation_profile(*words))
                    expected = tuple(2*c_target[u % 43]+2*(-1 if u % 2 else 1)*nearby[u % 43]
                                     for u in range(86))
                    # For u>=43 the C term repeats and the alternating G
                    # sign reverses, which is exactly the antipodal identity.
                    if profile != expected or joint_integer_energy(words, target) != 8:
                        raise RuntimeError('near shell failed independent original binary energy audit')
                    if verify_pqcp(*words).is_valid:
                        raise RuntimeError('unexpected PQCP in nonzero fixed-content shell; investigate rather than miscount')
                    candidate = dict(A=words[0], B=words[1], k=target.k, sign=target.eta,
                                     target=vars(target), target_energy=4, source_trial=row['trial'],
                                     plus=plus, minus=minus, verified_pqcp=False)
                    candidates.append(candidate)
                    job['candidate'] = candidate
                    found += 1
                log.write(json.dumps(job)+'\n')
            decided = sum(counts.values()) == 420 and not counts['UNKNOWN']
            record = dict(source_trial=row['trial'], target=vars(target), masks=c_masks,
                          exact_g_status=exact['status'], shell_statuses=dict(counts),
                          complete=decided and exact['status'] != 'UNKNOWN', energy4_candidates=found,
                          full_energy_lower_bound=(8 if decided and not found and exact['status'] == 'UNSAT_FIXED_SUPPORT' else None))
            records.append(record)
            log.flush()
            report = dict(rows=records, candidates=candidates,
                          complete=len(records) == len(source['rows']) and all(r['complete'] for r in records),
                          independently_verified_new_classes=0, elapsed=perf_counter()-tick,
                          source_manifest_sha256=hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
                          scope='Exact minimum nonzero G-error shell on supplied C supports/contents. '
                                'Near candidates are NOT PQCPs. Bounds concern these fixed supports only.')
            atomic_write_json(args.output/'report.json', report)
            print(json.dumps(dict(record, elapsed=perf_counter()-tick)), flush=True)


if __name__ == '__main__':
    main()
