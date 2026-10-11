"""IDE-runnable complete g-weight-four, arbitrary-O centered construction.

All three-plus/one-minus g of sum two, normalized minus at zero. Common
reflection quotient retains the entire union up to PQCP equivalence. Every
complementary C of sum nine, every O of sum one, every exact integer lag.
Only the declared construction is excluded by a complete negative result.
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
from solver.checkpoint import atomic_write_json
from solver.norm_trace_completion import reverse_mask
from solver.same_o_weight4 import normalized_g_rows,c_affine_certificate,required_o_intersections,odd_integer_match
from solver.same_o_weight4_native import complete_weight4_c_fiber,complete_weight2_c_fiber
from solver.sds_norm_completion import intersection_profile
from solver.perfect_odd_kernel import interleave_from_complementary_trits
from solver.search_runner import append_verified_solution_if_new
from solver.structured_energy import target_profile
from solver.verifier import verify_pqcp

DEFAULT_MAX_G=0  # Zero: the entire normalized g bank, with no sampled cutoff.


def main(default_g_weight=4):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-g',type=int,default=DEFAULT_MAX_G)
    parser.add_argument('--g-weight',type=int,choices=(2,4),default=default_g_weight)
    parser.add_argument('--reflection-quotient',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.max_g<0:parser.error('nonnegative max-g required')
    if args.output is None:args.output=ROOT/'results'/('L86_same_O_weight'+str(args.g_weight)+'_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    if args.g_weight==4:
        pool=tuple((p,m) for p,m in normalized_g_rows() if not args.reflection_quotient or p<reverse_mask(p,43))
        expected_bank=5740 if args.reflection_quotient else 11480
    else:
        pool=tuple(((1<<s)|(1<<(43-s)),0) for s in range(1,22));expected_bank=21
    if len(pool)!=expected_bank:raise RuntimeError('normalized g bank coverage failed')
    if args.max_g:pool=pool[:args.max_g]
    args.output.mkdir(parents=True,exist_ok=False);tick=perf_counter()
    snapshot=args.output/'source_snapshot';freeze_sources(snapshot);(snapshot/'experiments').mkdir(exist_ok=True)
    for path in (ROOT/'experiments').glob('*.py'):shutil.copy2(path,snapshot/'experiments'/path.name)
    manifest={str(p.relative_to(snapshot)):hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(snapshot.rglob('*')) if p.is_file() and p.name!='source_manifest.json'}
    atomic_write_json(snapshot/'source_manifest.json',manifest)
    atomic_write_json(args.output/'input_g_rows.json',dict(rows=pool,reflection_quotient=args.reflection_quotient))
    raw=(ROOT/'86.txt').read_bytes() if (ROOT/'86.txt').exists() else b''
    (args.output/'initial_inventory.txt').write_bytes(raw);known=verified_inventory(raw);keys=set();first_hit=None
    totals=Counter();ranks=Counter();full_hits=[]
    print(json.dumps(dict(expected_fibers=len(pool),entire_bank=len(pool)==expected_bank)),flush=True)
    for number,(plus,minus) in enumerate(pool):
        certificate=c_affine_certificate(plus,minus,negative_weight=17-args.g_weight//2,g_weight=args.g_weight)
        out=(complete_weight4_c_fiber if args.g_weight==4 else complete_weight2_c_fiber)(certificate)
        if not out['complete']:raise RuntimeError('C/O fiber incomplete')
        if certificate['inconsistent']:totals['inconsistent_c_fibers']+=1
        else:ranks[certificate['rank']]+=1
        totals['declared_raw_affine_assignments']+=out['raw_affine_assignments']
        totals.update(out['stats'])
        for negative,o_mask in out['hits']:
            if (intersection_profile(o_mask,43)!=required_o_intersections(negative,plus,minus)
                    or not odd_integer_match(o_mask,plus,minus)):
                raise RuntimeError('native hit failed independent integer E/O audit')
            c=tuple(0 if (plus|minus)>>i&1 else 1-2*((negative>>i)&1) for i in range(43))
            g=tuple(int(plus>>i&1)-int(minus>>i&1) for i in range(43))
            o=tuple(1-2*((o_mask>>i)&1) for i in range(43));words=interleave_from_complementary_trits(c,g,o)
            check=verify_pqcp(*words)
            if not check.is_valid or tuple(check.profile)!=target_profile(86,41,1):
                raise RuntimeError('native weight-four hit failed independent full L86 verifier')
            full_hits.append(dict(plus=plus,minus=minus,c_negative=negative,o_mask=o_mask,words=words))
            key=discovery_key(*words)
            if key not in known:
                if key not in keys and first_hit is None:first_hit=perf_counter()-tick
                keys.add(key);append_verified_solution_if_new(86,*words,args.output)
        with (args.output/'completion_fibers.jsonl').open('a') as handle:
            handle.write(json.dumps(dict(index=number,certificate=certificate,result=out))+'\n')
        if (number+1)%50==0 or number+1==len(pool):
            report=dict(complete=number+1==len(pool),entire_normalized_bank=len(pool)==expected_bank,
                expected_fibers=len(pool),completed_fibers=number+1,unquotiented_g_bank=11480 if args.g_weight==4 else 21,
                g_weight=args.g_weight,
                reflection_quotient=args.reflection_quotient,ranks=dict(ranks),totals=dict(totals),
                full_hits=full_hits,independently_verified_new_classes=len(keys),first_verified_elapsed=first_hit,
                elapsed=perf_counter()-tick,source_manifest_sha256=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest(),
                scope='Selected declared g bank; each g has ALL complementary C of sum nine and ALL sum-one O. '
                      'No skew/perfect requirement, no O rotation quotient, target 41/45 with +4. '
                      'Whole-bank reflection quotient preserves solution classes. Restricted construction only.',
                claim='No full-new-discovery-speed claim from candidate pruning, raw affine sizes or zero discoveries.')
            atomic_write_json(args.output/'report.json',report)
            print(json.dumps({k:report[k] for k in ('complete','completed_fibers','independently_verified_new_classes','elapsed')}),flush=True)


if __name__=='__main__':main()
