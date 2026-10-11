"""Complete restricted odd-positive L86 interleave family via integer spheres.

IDE Run defaults to all eight perfect-43 rotation representatives and all 21
odd target pairs, with sum(E_A)=11, sum(E_B)=7. A missing integer certificate
raises; native UNKNOWN remains UNKNOWN. No restriction of the general solver.
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

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.benchmark_l86_discovery import discovery_key,freeze_sources,verified_inventory
from experiments.search_perfect43_basis import perfect43_bases
from solver.checkpoint import atomic_write_json
from solver.perfect_affine_kernel import perfect_affine_kernel_certificate,perfect_affine_trits
from solver.perfect_odd_kernel import interleave_from_complementary_trits
from solver.search_runner import append_verified_solution_if_new
from solver.structured_energy import integer_periodic_autocorrelation_profile as pacf,target_profile
from solver.ternary_system_native import prepare_native_system,run_native_system
from solver.verifier import verify_pqcp


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'results'/(
        'L86_perfect_affine_O_'+datetime.now().strftime('%Y%m%d_%H%M%S')))
    parser.add_argument('--max-assignments',type=int,default=2**23)
    args=parser.parse_args()
    if not 1<=args.max_assignments<=2**31:parser.error('invalid exact sign budget')
    args.output.mkdir(parents=True,exist_ok=False)
    tick=perf_counter();snapshot=args.output/'source_snapshot'
    freeze_sources(snapshot)
    for name in ('search_perfect_affine_o.py','search_perfect43_basis.py','benchmark_l86_discovery.py'):
        destination=snapshot/'experiments'/name
        destination.parent.mkdir(exist_ok=True)
        shutil.copy2(ROOT/'experiments'/name,destination)
    manifest={str(p.relative_to(snapshot)):hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(snapshot.rglob('*')) if p.is_file() and p.name!='source_manifest.json'}
    atomic_write_json(snapshot/'source_manifest.json',manifest)
    raw=(ROOT/'86.txt').read_bytes() if (ROOT/'86.txt').exists() else b''
    (args.output/'initial_inventory.txt').write_bytes(raw)
    known=verified_inventory(raw);keys=set();first_hit=None
    source=perfect43_bases();certificates=[];fibers=[];jobs=[]
    for ingredient in source['rows']:
        certificate=perfect_affine_kernel_certificate(ingredient['signs'])
        certificates.append(dict(parent_mask=ingredient['mask'],certificate=certificate))
        for t in range(1,22):
            fiber=perfect_affine_trits(certificate,t,34)
            fibers.append(dict(parent_mask=ingredient['mask'],**fiber))
            for candidate in fiber['candidates']:
                g=candidate['g'];weight=43-candidate['norm']
                if weight<9 or (weight-9)%2:raise RuntimeError('C sum-nine content obstruction')
                mask=sum(int(v==0)<<i for i,v in enumerate(g));pg=tuple(pacf(g))
                wanted=(weight,)+tuple(1-pg[u] for u in range(1,43))
                if sum(wanted)!=81:raise RuntimeError('C sum-square identity failed')
                result=run_native_system(prepare_native_system((mask,0),43,wanted,((weight-9)//2,0)),
                                         args.max_assignments)
                row=dict(parent_mask=ingredient['mask'],t=t,g=g,c_support=mask,wanted=wanted,result=result)
                if result['words'] is not None:
                    c=result['words'][0]
                    if sum(c)!=9 or tuple(pacf(c))!=wanted:raise RuntimeError('C witness failed audit')
                    words=interleave_from_complementary_trits(c,g,ingredient['signs'])
                    check=verify_pqcp(*words)
                    if not check.is_valid or tuple(check.profile)!=target_profile(86,43-2*t,1):
                        raise RuntimeError('affine interleave failed full independent target audit')
                    row['full_words']=words;key=discovery_key(*words)
                    if key not in known:
                        if key not in keys and first_hit is None:first_hit=perf_counter()-tick
                        keys.add(key);append_verified_solution_if_new(86,*words,args.output)
                jobs.append(row)
        report=dict(complete=len(certificates)==len(source['rows']) and len(fibers)==21*len(source['rows'])
                    and all(row['result']['status']!='UNKNOWN' for row in jobs),
                    ingredient_count=len(source['rows']),certificates=certificates,fibers=fibers,jobs=jobs,
                    statuses=dict(Counter(row['result']['status'] for row in jobs)),
                    independently_verified_new_classes=len(keys),first_verified_elapsed=first_hit,
                    elapsed=perf_counter()-tick,source_manifest_sha256=hashlib.sha256(
                        json.dumps(manifest,sort_keys=True).encode()).hexdigest(),
                    scope='All perfect O of sum one modulo rotation, all odd positive targets, '
                          'identical opposite odd rows and even-row sums 11 and 7. '
                          'Existence decision per fixed G; not global L86 exclusion or all-SAT enumeration.')
        atomic_write_json(args.output/'report.json',report)
        print(json.dumps(dict(parent_mask=ingredient['mask'],fibers=len(fibers),jobs=len(jobs),
                              statuses=report['statuses'],new_classes=len(keys),elapsed=report['elapsed'])),flush=True)


if __name__=='__main__':main()
