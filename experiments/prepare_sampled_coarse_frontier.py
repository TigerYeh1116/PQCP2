"""Frozen audit of both rows of sampled C carriers as radius-zero anchors.

Closed all-84-target points and norm/support-equivalent points can be
omitted here. This certificate never transfers to a positive-radius ball.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.benchmark_l86_discovery import freeze_sources
from experiments.complete_l86_full_rows import audited_coarse_centers
from experiments.extract_coarse_successor_anchors import closed_bank, file_sha
from solver.checkpoint import atomic_write_json
from solver.coarse_anchor_orbits import coarse_anchor_key


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--closed', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--_frozen-runner', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    source = args.input.resolve()
    output = args.output.resolve()
    banks = [p.resolve() for p in args.closed]
    snapshot = output/'source_snapshot'
    if not args._frozen_runner:
        output.mkdir(parents=True, exist_ok=False)
        freeze_sources(snapshot)
        for p in (ROOT/'experiments').glob('*.py'):
            dest = snapshot/'experiments'/p.name
            dest.parent.mkdir(exist_ok=True)
            shutil.copy2(p, dest)
        shutil.copy2(source, output/'input_carriers.json')
        shutil.copy2(source.parent/'report.json', output/'input_report.json')
        manifest = {str(p.relative_to(snapshot)): file_sha(p) for p in sorted(snapshot.rglob('*'))
                    if p.is_file() and p.name != 'source_manifest.json'}
        atomic_write_json(snapshot/'source_manifest.json', manifest)
        command = [sys.executable, str(snapshot/'experiments'/Path(__file__).name),
            '--_frozen-runner', '--input', str(source), '--output', str(output)]
        for bank in banks:
            command.extend(('--closed', str(bank)))
        atomic_write_json(output/'execution.json', dict(actual_frozen_execution=True, command=command,
            source=str(source), input_sha256=file_sha(source), cwd=str(snapshot),
            source_manifest_sha256=hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()))
        raise SystemExit(subprocess.run(command, cwd=snapshot).returncode)
    if ROOT != snapshot:
        raise RuntimeError('actual archived execution required')
    manifest = json.loads((snapshot/'source_manifest.json').read_text())
    if any(file_sha(snapshot/name) != digest for name, digest in manifest.items()):
        raise RuntimeError('archived source changed')
    packet = json.loads((output/'input_carriers.json').read_text())
    report = json.loads((output/'input_report.json').read_text())
    if (not packet['complete'] or not report['complete']
            or len(packet['rows']) != report['coarse_witnesses']
            or any(row['result'].get('verified') for row in packet['rows'])):
        raise ValueError('completed sampled compressed-carrier bank and matching report required')
    seeds = audited_coarse_centers(output/'input_carriers.json', 0)['seeds']
    closed = {}
    proofs = []
    for bank in banks:
        _, additional = closed_bank(bank)
        closed.update(additional)
        proofs.append(dict(path=str(bank), certificate_sha256=file_sha(bank/'certificate.json')))
    rows, observations = [], []
    seen = set()
    for seed, item in zip(seeds, packet['rows']):
        for side, word in enumerate(seed['coarse_rows']):
            key = coarse_anchor_key(word)
            origin = dict(source_row=seed['source_row'], side=side, trial=item['trial'])
            if key in closed:
                observations.append(dict(origin=origin, status='ALREADY_CLOSED_ALL_84_TARGETS', proof=closed[key]))
                continue
            if key in seen:
                observations.append(dict(origin=origin, status='DUPLICATE_RADIUS_ZERO_NORM_SUPPORT_POINT'))
                continue
            seen.add(key)
            pair = item['result']['attempts']['coarse']['words']
            if side == 0:
                pair = pair[::-1]
            rows.append(dict(target=item['target'], source=origin,
                result=dict(verified=False, attempts=dict(coarse=dict(status='SAT_COMPRESSED', words=pair)))))
            observations.append(dict(origin=origin, status='NEW_RADIUS_ZERO_POINT'))
    atomic_write_json(output/'coarse_witnesses.json', dict(rows=rows, planned_radius=0,
        scope='Audited both-row fixed points only. Norm/support equivalence is not a ball isometry.'))
    result = dict(complete=True, actual_frozen_execution=True, audited_carriers=len(packet['rows']),
        literal_source_points=2*len(packet['rows']), new_radius_zero_points=len(rows),
        closed_point_keys=len(closed), closed_proofs=proofs, observations=observations,
        copied_input_sha256=file_sha(output/'input_carriers.json'),
        copied_report_sha256=file_sha(output/'input_report.json'),
        claim='Eligible fixed-anchor domains only, not new full PQCPs or a throughput claim.')
    atomic_write_json(output/'report.json', result)
    print(json.dumps({k: result[k] for k in ('complete', 'audited_carriers', 'new_radius_zero_points', 'closed_point_keys')}), flush=True)


if __name__ == '__main__':
    main()
