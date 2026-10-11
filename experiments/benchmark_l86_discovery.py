"""Paired full-pipeline runs in frozen code/isolated archives, retaining zeros.

Compares the new mathematically constrained observation path with its rollback.
Only independent full verifier hits count. Alternate modulation is additionally
quotiented for measurement, so transformed models cannot inflate discovery rate.
This experiment does not initialize from archived solutions.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from solver.l86_binary_content import RECORDS
from solver.l86_content_certificate import WEIGHT7_RECORDS
from solver.solution_equivalence import pqcp_equivalence_key
from solver.verifier import verify_pqcp


def discovery_key(a, b):
    a, b = tuple(map(int, a)), tuple(map(int, b))
    if len(a) != 86 or len(b) != 86 or not verify_pqcp(a, b).is_valid:
        raise ValueError('only fully verified L86 pairs enter discovery accounting')
    return min(pqcp_equivalence_key(a, b), pqcp_equivalence_key(
        tuple(v ^ (i % 2) for i, v in enumerate(a)),
        tuple(v ^ (i % 2) for i, v in enumerate(b))))


def verified_inventory(raw):
    text = raw.decode('utf-8')
    pairs = re.findall(r'^a=([01]+)\r?\nb=([01]+)\r?$', text, re.MULTILINE)
    if len(pairs) != len(re.findall(r'^a=', text, re.MULTILINE)) or len(pairs) != len(re.findall(r'^b=', text, re.MULTILINE)):
        raise ValueError('malformed solution inventory')
    return {discovery_key(a, b) for a, b in pairs}


def freeze_sources(destination):
    destination.mkdir()
    shutil.copy2(ROOT/'main.py', destination/'main.py')
    for name in ('solver', 'csrc'):
        shutil.copytree(ROOT/name, destination/name,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.DS_Store'))
    proof_paths = ['results/'+row[0] for row in WEIGHT7_RECORDS]
    proof_paths += ['results/'+row[0]+'/report.json' for row in RECORDS]
    for name in proof_paths:
        path = destination/name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, path)
    manifest = {str(path.relative_to(destination)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(destination.rglob('*')) if path.is_file()}
    (destination/'source_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()


def run_process(command, cwd, log_path, budget):
    timed_out = False
    with log_path.open('w') as log:
        process = subprocess.Popen(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=budget+180)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGINT)
            try:
                code = process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    code = process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    code = process.wait()
    return code, timed_out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=(123, 456, 789))
    parser.add_argument('--seconds', type=float, default=300)
    parser.add_argument('--batch', type=int, default=16384)
    parser.add_argument('--comparison',choices=('projection','alternation'),default='projection')
    parser.add_argument('--inventory',type=Path)
    args = parser.parse_args()
    if args.seconds <= 0 or args.batch <= 0 or len(set(args.seeds)) != len(args.seeds):
        parser.error('positive budgets and distinct seeds required')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source_sha = freeze_sources(output/'source')
    inventory_path=args.inventory if args.inventory is not None else ROOT/'86.txt'
    inventory = inventory_path.read_bytes() if inventory_path.exists() else b''
    reference_keys = verified_inventory(inventory)
    (output/'initial_inventory.txt').write_bytes(inventory)
    rows = []
    for index, seed in enumerate(args.seeds):
        for variant in (('rollback', 'candidate') if index % 2 == 0 else ('candidate', 'rollback')):
            destination = output/('{}_seed{}'.format(variant, seed))
            destination.mkdir()
            if inventory:
                (destination/'86.txt').write_bytes(inventory)
            command = [sys.executable, str(output/'source/main.py'), '--L', '86', '--seed', str(seed),
                '--seconds', str(args.seconds), '--torch-device', 'mps', '--torch-batch-size', str(args.batch),
                '--z3', '--torch-direct-compression', '--torch-polish-mode', 'alternating']
            if args.comparison == 'projection' and variant == 'rollback':
                command.append('--no-torch-half-shift-observation-projection')
            if args.comparison == 'alternation':
                command.append('--torch-alternation-orbit-pruning' if variant == 'candidate'
                               else '--no-torch-alternation-orbit-pruning')
            print(json.dumps(dict(event='start', variant=variant, seed=seed, budget=args.seconds)), flush=True)
            started = perf_counter()
            code, timed_out = run_process(command, destination, destination/'console.log', args.seconds)
            wall = perf_counter()-started
            final_raw = (destination/'86.txt').read_bytes() if (destination/'86.txt').exists() else b''
            new_keys = verified_inventory(final_raw)-reference_keys
            metrics = [json.loads(path.read_text()) for path in sorted(destination.glob('results/search_metrics/*.json'))]
            row = dict(variant=variant, comparison=args.comparison, seed=seed, budget=args.seconds, wall_seconds=wall,
                returncode=code, timed_out=timed_out, independently_verified_new_classes=len(new_keys),
                new_class_keys=sorted(new_keys), metrics=metrics)
            rows.append(row)
            report = dict(source_manifest_sha256=source_sha, comparison=args.comparison,
                inventory_sha256=hashlib.sha256(inventory).hexdigest(), batch=args.batch,
                rows=rows, complete=len(rows)==2*len(args.seeds) and all(r['returncode']==0 and not r['timed_out'] for r in rows),
                claim='No discovery-speed claim if both variants find zero new classes.',
                scope=('Full production pipeline; simultaneous alternating-sign target orbit reduction versus rollback; '
                       'same fixed time budget and other mathematics. No speed claim from case counts alone.'
                       if args.comparison=='alternation' else
                       'Full production pipeline; latest observation method versus its rollback; all other mathematics identical.'))
            (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
            print(json.dumps({k:v for k,v in row.items() if k != 'metrics'}), flush=True)
            if code or timed_out:
                return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
