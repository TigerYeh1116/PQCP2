"""IDE-runnable exact arbitrary-position repair of fresh non-solution centers.

Default: best audited centers from the latest root L86 torch log, up to four
flips per row, allowing all parity contents. If no log exists, use GS centers.
Centers are quotiented only by Hamming isometries. This experiment never
seeds from an already verified PQCP solution.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys
from time import perf_counter

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
# Edit this one value when using the IDE Run button. Five is a larger exact
# neighborhood with a temporary index upper bound below 1 GB at L86.
DEFAULT_RADIUS=4
from experiments.benchmark_l86_discovery import discovery_key,freeze_sources,verified_inventory
from solver.checkpoint import atomic_write_json
from solver.search_runner import append_verified_solution_if_new
from solver.solution_equivalence import pqcp_equivalence_key
from solver.correlation import full_correlation_profile,normalize_binary_sequence
from solver.structured_energy import target_profile
from solver.unrestricted_flip_join import exact_unrestricted_flip_repair
from solver.verifier import verify_pqcp


def audited_log_centers(path,top):
    """Select fresh non-solutions heuristically; no region is mathematically excluded."""
    candidates={};skipped_solutions=0
    with path.open() as handle:
        for number,line in enumerate(handle,1):
            if not line.strip():continue
            row=json.loads(line)
            if row.get('L')!=86 or 'A' not in row or 'B' not in row:continue
            words=tuple(tuple(normalize_binary_sequence(row[name])) for name in ('A','B'))
            if tuple(map(len,words))!=(86,86):raise ValueError('log row length mismatch')
            check=verify_pqcp(*words)
            if check.is_valid:
                skipped_solutions+=1;continue
            profile=tuple(full_correlation_profile(*words))
            if 'profile' in row and tuple(row['profile'])!=profile:
                raise ValueError('logged full profile failed independent integer audit at line '+str(number))
            k,eta=row.get('k'),row.get('sign')
            if type(k) is not int or not 1<=k<43 or eta not in (-1,1):continue
            wanted=target_profile(86,k,eta)
            energy=sum(((profile[u]-wanted[u])//4)**2 for u in range(1,44))
            record=dict(words=words,line=number,k=k,eta=eta,independent_full_energy=energy,profile=profile)
            key=pqcp_equivalence_key(*words)
            if key not in candidates or energy<candidates[key]['independent_full_energy']:
                candidates[key]=record
    ordered=sorted(candidates.values(),key=lambda r:(r['independent_full_energy'],r['line']))
    selected=[];cases=set()
    # First give each observed parity/sign family a center, then fill by energy.
    for row in ordered:
        case=(row['k']%2,row['eta'])
        if case not in cases and len(selected)<top:selected.append(row);cases.add(case)
    selected.extend(row for row in ordered if row not in selected)
    return dict(seeds=selected[:top],log=str(path),literal_logged_solutions_skipped=skipped_solutions,
                audited_non_solution_classes=len(candidates),selection='Lowest exact energy per observed '
                'parity/sign family, then by exact energy. Bounded heuristic center selection, no global exclusion.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--input',type=Path,help='JSON with a seeds/words list')
    group.add_argument('--log',type=Path,help='L86 torch JSONL; default is newest file in root logs')
    parser.add_argument('--top',type=int,default=16,help='maximum audited centers selected from a log')
    parser.add_argument('--output',type=Path,default=ROOT/'results'/(
        'L86_unrestricted_flip_repair_'+datetime.now().strftime('%Y%m%d_%H%M%S')))
    parser.add_argument('--radius',type=int,default=DEFAULT_RADIUS)
    parser.add_argument('--content-grouping',action=argparse.BooleanOptionalAction,default=True)
    args=parser.parse_args()
    if not 0<=args.radius<=5:parser.error('radius must be 0..5')
    if args.top<1:parser.error('top must be positive')
    if args.input is None and args.log is None:
        logs=tuple((ROOT/'logs').glob('torch_L86_seed*.jsonl'))
        if logs:args.log=max(logs,key=lambda p:p.stat().st_mtime)
        else:args.input=ROOT/'results/L86_GS_interleave_20261007/report.json'
    inputs=audited_log_centers(args.log,args.top) if args.log is not None else json.loads(args.input.read_text())
    if not inputs['seeds']:raise ValueError('no non-solution L86 centers to repair')
    args.output.mkdir(parents=True,exist_ok=False)
    tick=perf_counter();snapshot=args.output/'source_snapshot';freeze_sources(snapshot)
    for name in ('repair_unrestricted_flips.py','benchmark_l86_discovery.py'):
        destination=snapshot/'experiments'/name;destination.parent.mkdir(exist_ok=True)
        shutil.copy2(ROOT/'experiments'/name,destination)
    manifest={str(p.relative_to(snapshot)):hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(snapshot.rglob('*')) if p.is_file() and p.name!='source_manifest.json'}
    atomic_write_json(snapshot/'source_manifest.json',manifest)
    source=args.log if args.log is not None else args.input
    shutil.copy2(source,args.output/('input_log.jsonl' if args.log is not None else 'input_source.json'))
    atomic_write_json(args.output/'input_centers.json',inputs)
    raw=(ROOT/'86.txt').read_bytes() if (ROOT/'86.txt').exists() else b''
    (args.output/'initial_inventory.txt').write_bytes(raw)
    known=verified_inventory(raw);centers={}
    for seed in inputs['seeds']:
        words=seed['words']
        if tuple(map(len,words))!=(86,86):raise ValueError('L86 centers required')
        centers.setdefault(pqcp_equivalence_key(*words),words)
    print(json.dumps(dict(input=str(source),centers=len(centers),radius=args.radius)),flush=True)
    rows=[];keys=set();first_hit=None
    for index,words in enumerate(centers.values()):
        result=exact_unrestricted_flip_repair(*words,radius=args.radius,content_grouping=args.content_grouping)
        for hit in result['witnesses']:
            key=discovery_key(*hit['words'])
            if key not in known:
                if key not in keys and first_hit is None:first_hit=perf_counter()-tick
                keys.add(key);append_verified_solution_if_new(86,*hit['words'],args.output)
        rows.append(dict(center=words,result=result))
        report=dict(complete=len(rows)==len(centers) and all(row['result']['existence_decided'] for row in rows),
                    rows=rows,radius_per_row=args.radius,content_grouping=args.content_grouping,expected_centers=len(centers),
                    independently_verified_new_classes=len(keys),first_verified_elapsed=first_hit,
                    elapsed=perf_counter()-tick,source_manifest_sha256=hashlib.sha256(
                        json.dumps(manifest,sort_keys=True).encode()).hexdigest(),
                    scope='Existence decisions in ALL arbitrary-position radius-r per-row product balls '
                          'of these non-solution centers. No fixed-content restriction. '
                          'All literal signature preimages retained; no global exclusion or speed claim.')
        atomic_write_json(args.output/'report.json',report)
        print(json.dumps(dict(center=index,status=result['status'],
                              indexed=result['indexed_configurations'],mod8_pairs=result['mod8_candidate_pairs'],
                              checks=result['full_coordinate_checks'],new_classes=len(keys),
                              elapsed=report['elapsed'])),flush=True)


if __name__=='__main__':main()
