"""Try the other full target above the SAME folded C witness.

k and 43-k have identical C norms but opposite G target signs. They have
different full-length gcd orbits. No old G-UNSAT is reused for its sibling.
Only original binary full verification and class accounting constitute hits.
"""
import argparse
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.benchmark_l86_discovery import freeze_sources,discovery_key,verified_inventory
from solver.checkpoint import atomic_write_json
from solver.compressed_repair import _validated_reference
from solver.pqcp_support_solver import complete_pqcp_supports
from solver.search_runner import append_verified_solution_if_new
from solver.structured_energy import integer_periodic_autocorrelation_profile as pacf
from solver.target_profiles import TargetContentProfile
from solver.ternary_support_search import support_mask


def sibling_profile(words,original):
    if original.L!=86 or not 1<=original.k<=42:raise ValueError('strict L86 target required')
    k=43-original.k;epsilon=original.eta*(-1 if k%2 else 1)
    magnitudes=(3,6) if epsilon==1 else (4,5)
    counts=[]
    for word in words:
        weight=sum(v!=0 for v in word);zero=43-weight;sigma=sum(word)
        g=next(m for m in magnitudes if m%2==zero%2)
        if g>zero:return None
        even=(43-sigma-g)//2;odd=(43-sigma+g)//2
        if not 0<=even<=43 or not 0<=odd<=43:return None
        counts.extend((even,odd))
    target=TargetContentProfile(86,k,original.eta,*counts)
    _validated_reference(*words,target)
    return target


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=ROOT/'results/L86_independent_norm_phase_ten_million_20261008/coarse_witnesses.json')
    p.add_argument('--output',type=Path,default=ROOT/'results'/('L86_coarse_sibling_lifts_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')))
    p.add_argument('--_worker',action='store_true',help=argparse.SUPPRESS)
    args=p.parse_args();out=args.output.resolve()
    if not args._worker:
        out.mkdir(parents=True,exist_ok=False);source=out/'source_snapshot';freeze_sources(source)
        (source/'experiments').mkdir()
        for name in ('lift_coarse_target_siblings.py','benchmark_l86_discovery.py'):
            shutil.copy2(ROOT/'experiments'/name,source/'experiments'/name)
        manifest={str(f.relative_to(source)):hashlib.sha256(f.read_bytes()).hexdigest()
                  for f in sorted(source.rglob('*')) if f.is_file() and f.name!='source_manifest.json'}
        atomic_write_json(source/'source_manifest.json',manifest)
        shutil.copy2(args.input,out/'input_coarse_witnesses.json')
        raw=(ROOT/'86.txt').read_bytes() if (ROOT/'86.txt').exists() else b''
        verified_inventory(raw);(out/'initial_inventory.txt').write_bytes(raw)
        return subprocess.run([sys.executable,str(source/'experiments/lift_coarse_target_siblings.py'),
            '--_worker','--input',str(out/'input_coarse_witnesses.json'),'--output',str(out)],cwd=source).returncode
    tick=perf_counter();known=verified_inventory((out/'initial_inventory.txt').read_bytes());keys=set();rows=[];first_hit=None
    inputs=json.loads(args.input.read_text())['rows']
    for item in inputs:
        old=TargetContentProfile(**item['target']);words=tuple(tuple(w) for w in item['result']['attempts']['coarse']['words'])
        _,_,_,expected=_validated_reference(*words,old)
        if tuple(x+y for x,y in zip(pacf(words[0]),pacf(words[1])))!=expected:raise RuntimeError('input is not an exact C witness')
        target=sibling_profile(words,old)
        if target is None:result=dict(status='UNSAT_SIBLING_CONTENT',verified=False,words=None)
        else:
            result=complete_pqcp_supports(*words,target,tuple(support_mask(w) for w in words),even_sign_orbit=True)
            if result['verified']:
                key=discovery_key(*result['words'])
                if key not in known and key not in keys:
                    if first_hit is None:first_hit=perf_counter()-tick
                    keys.add(key);append_verified_solution_if_new(86,*result['words'],out)
        rows.append(dict(source_trial=item['trial'],original_target=asdict(old),
                         sibling_target=asdict(target) if target is not None else None,result=result))
        report=dict(rows=rows,tested=len(rows),planned=len(inputs),complete=len(rows)==len(inputs),
            all_siblings_decided=all(r['result']['status']!='UNKNOWN' for r in rows),
            independently_verified_new_classes=len(keys),first_verified_elapsed=first_hit,elapsed=perf_counter()-tick,
            input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
            scope='Other full target over each actual C witness; original G-UNSAT not reused. No global nonexistence or discovery-speed claim from misses.')
        atomic_write_json(out/'report.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='rows'}),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
