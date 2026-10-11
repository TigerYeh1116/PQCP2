"""IDE Run: mathematically filtered two-step neighbours of both sampled C rows.

Generate the literal neighbours BEFORE using fixed-point equivalence.
Remaining G_A signs and all binary B bits stay free. A negative result
only closes this finite frontier, never all L86 PQCPs. Existing main.py
and independent norm-sampling defaults are unchanged.
"""
import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.benchmark_l86_discovery import freeze_sources, verified_inventory
from experiments.complete_l86_full_rows import audited_coarse_centers
from experiments.extract_coarse_successor_anchors import closed_bank, file_sha
from solver.checkpoint import atomic_write_json
from solver.coarse_anchor_orbits import coarse_anchor_key
from solver.coarse_character import admissible_coarse_characters, coarse_all_target_psd_obstruction
from solver.coarse_row_completion import coarse_l1_anchors

RUN_INPUT = ROOT/'results/L86_phase_grid_default_original_ten_million_20261009/coarse_witnesses.json'
RUN_RADIUS = 2  # The previous entire one-step frontier is independently closed.
RUN_CLOSED = tuple(ROOT/'results'/name for name in (
    'L86_hundred_million_coarse_radius0_closed_20261009',
    'L86_hundred_million_successor_radius0_closed_20261009',
    'L86_phase_grid_million_coarse_radius0_closed_20261009',
    'L86_phase_million_coarse_radius1_closed_20261008',
    'L86_paired_C_million_coarse_radius0_closed_20261009',
    'L86_weight_prefix_successor_radius0_closed_20261009',
    'L86_weight_prefix_successor_round2_closed_20261009',
    'L86_phase_grid_ten_million_coarse_radius0_closed_20261011',
    'L86_coarse_neighbour_radius1_closed_20261011',
))


def neighbour_frontier(seeds, closed, *, key_function=coarse_anchor_key, spectral_pruning=True, on_spectral_exclusion=None,radius=1):
    """Complete literal L1 shell one or two; all exclusions apply to leaves."""
    if type(radius) is not int or radius not in (1,2):raise ValueError('exact L1 shell one or two required')
    points, seen, stats = [], set(), Counter()
    for parent, seed in enumerate(seeds):
        for side, original in enumerate(seed['coarse_rows']):
            stats['literal_centres'] += 1
            for row, edits in coarse_l1_anchors(original, radius):
                if sum(abs(old-new) for _,old,new in edits)!=radius:
                    continue  # Exactly this shell, not its centre/interior.
                stats['literal_neighbours'] += 1
                characters = admissible_coarse_characters(row)
                if not characters:
                    stats['scalar_impossible'] += 1
                    continue
                stats['scalar_admissible'] += 1
                key = key_function(row)
                if key in closed:
                    stats['already_closed_leaves'] += 1
                    continue
                if key in seen:
                    stats['duplicate_leaves'] += 1
                    continue
                seen.add(key)
                if spectral_pruning:
                    proof = coarse_all_target_psd_obstruction(row)
                    if proof is not None:
                        stats['spectral_impossible'] += 1
                        if on_spectral_exclusion is not None:
                            on_spectral_exclusion(dict(row=row, proof=proof, parent=parent, side=side, edits=edits))
                        continue
                points.append(dict(row=row, admissible_characters=characters,
                    provenance=dict(parent=parent, source_row=seed.get('source_row', parent),
                                    side=side, edits=edits)))
    stats['new_fixed_points'] = len(points)
    assert stats['literal_neighbours'] == (stats['scalar_impossible']+stats['already_closed_leaves']
                                          +stats['duplicate_leaves']+stats['spectral_impossible']+len(points))
    return points, dict(stats)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=RUN_INPUT)
    parser.add_argument('--closed', type=Path, action='append', help='Audited complete all-target fixed-point bank; defaults to existing nine banks')
    parser.add_argument('--radius',type=int,choices=(1,2),default=RUN_RADIUS)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--spectral-pruning', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--top', type=int, default=0, help='0 searches the entire prepared frontier; positive values explicitly select a bounded prefix')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--_frozen-runner', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.top < 0:
        parser.error('nonnegative top required')
    source = args.input.resolve()
    banks = tuple(p.resolve() for p in (args.closed if args.closed is not None else RUN_CLOSED))
    output = (args.output or ROOT/'results'/('L86_coarse_neighbours_'+datetime.now().strftime('%Y%m%d_%H%M%S'))).resolve()
    snapshot = output/'source_snapshot'
    if not args._frozen_runner:
        output.mkdir(parents=True, exist_ok=False)
        freeze_sources(snapshot)
        for path in (ROOT/'experiments').glob('*.py'):
            target = snapshot/'experiments'/path.name
            target.parent.mkdir(exist_ok=True)
            shutil.copy2(path, target)
        shutil.copy2(source, output/'input_carriers.json')
        shutil.copy2(source.parent/'report.json', output/'input_report.json')
        raw = (ROOT/'86.txt').read_bytes() if (ROOT/'86.txt').exists() else b''
        verified_inventory(raw)
        (snapshot/'86.txt').write_bytes(raw)
        manifest = {str(p.relative_to(snapshot)): file_sha(p) for p in sorted(snapshot.rglob('*'))
                    if p.is_file() and p.name != 'source_manifest.json'}
        atomic_write_json(snapshot/'source_manifest.json', manifest)
        command = [sys.executable, str(snapshot/'experiments'/Path(__file__).name),
                   '--_frozen-runner', '--input', str(source), '--output', str(output), '--top', str(args.top),'--radius',str(args.radius)]
        for bank in banks:
            command.extend(('--closed', str(bank)))
        if args.prepare_only:
            command.append('--prepare-only')
        command.append('--spectral-pruning' if args.spectral_pruning else '--no-spectral-pruning')
        atomic_write_json(output/'execution.json', dict(actual_frozen_execution=True, command=command,
            copied_input_sha256=file_sha(output/'input_carriers.json'),
            copied_report_sha256=file_sha(output/'input_report.json'),
            top=args.top, prepare_only=args.prepare_only, spectral_pruning=args.spectral_pruning, source_shell_radius=args.radius, complete=False))
        code = subprocess.run(command, cwd=snapshot).returncode
        state = json.loads((output/'execution.json').read_text())
        state.update(exit_code=code, runner_finished=True)
        if (output/'coverage/report.json').exists():
            coverage = json.loads((output/'coverage/report.json').read_text())
            state.update(complete=bool(code == 0 and coverage['complete']),
                         independently_verified_new_classes=coverage['independently_verified_new_classes'])
        atomic_write_json(output/'execution.json', state)
        raise SystemExit(code)
    if ROOT != snapshot:
        raise RuntimeError('actual archived execution required')
    manifest = json.loads((snapshot/'source_manifest.json').read_text())
    if any(file_sha(snapshot/name) != digest for name, digest in manifest.items()):
        raise RuntimeError('archived source changed')
    packet = json.loads((output/'input_carriers.json').read_text())
    report = json.loads((output/'input_report.json').read_text())
    if (not packet['complete'] or not report['complete'] or len(packet['rows']) != report['coarse_witnesses']
            or any(row['result'].get('verified') for row in packet['rows'])):
        raise ValueError('completed sampled carrier bank with matching report required')
    seeds = audited_coarse_centers(output/'input_carriers.json', 0)['seeds']
    closed, proofs = {}, []
    for bank in banks:
        _, keys = closed_bank(bank)
        closed.update(keys)
        proofs.append(dict(path=str(bank), certificate_sha256=file_sha(bank/'certificate.json')))
    spectral_receipts = []
    points, stats = neighbour_frontier(seeds, closed, spectral_pruning=args.spectral_pruning,
                                      on_spectral_exclusion=spectral_receipts.append,radius=args.radius)
    atomic_write_json(output/'spectral_exclusions.json', dict(rows=spectral_receipts,
        scope='Strict outward integer certificate of PSD(C_A)>45 at one nonzero frequency; excludes all 84 targets at this fixed point.'))
    atomic_write_json(output/'coarse_points.json', dict(format='literal_coarse_points_v1', planned_radius=0,
        points=points, source_shell_radius=args.radius,
        scope='Literal L1 shell generated before character/closed-point/equivalence filtering. Each retained row is a fixed point, not a compressed pair or a whole ball.'))
    atomic_write_json(output/'frontier_report.json', dict(complete=True, actual_frozen_execution=True,
        audited_carriers=len(seeds), closed_point_keys=len(closed), closed_proofs=proofs, stats=stats,
        search_prefix_selected=args.top, spectral_pruning=args.spectral_pruning,source_shell_radius=args.radius,
        claim='Prepared finite search region only; no full new PQCP or speedup asserted.'))
    print(json.dumps(stats), flush=True)
    if args.prepare_only or not points:
        return
    command = [sys.executable, str(ROOT/'experiments/complete_l86_full_rows.py'),
        '--mode', 'coarse-row', '--coarse-anchors', str(output/'coarse_points.json'),
        '--radius', '0', '--anchor-side', '0', '--top', str(args.top), '--output', str(output/'coverage'),
        '--two-adic', '--dihedral-support', '--norm-unsat-cache', '4096',
        '--norm-carrier-cache', '4096', '--unary-batch-free', '12',
        '--prebatch-dihedral','--unary-linear-certificate','--native-support-audit']
    atomic_write_json(output/'search_execution.json', dict(actual_frozen_execution=True, command=command,
        input_sha256=file_sha(output/'coarse_points.json'), selected_top=args.top,
        scope='All 84 strict targets per selected fixed C_A. G_A and the other binary row remain free.'))
    raise SystemExit(subprocess.run(command, cwd=ROOT).returncode)


if __name__ == '__main__':
    main()
