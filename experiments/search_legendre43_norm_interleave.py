"""Complete finite Legendre/D-optimal construction expansion for strict L86.

Odd ingredients need only have PACF SUM -2, not individually be perfect.
ALL fixed-side norm completions of the stated trade pool are decided, and
the even ingredient bank is closed under common units/independent dihedral
actions BEFORE quotienting odd ingredient pairs. Both odd row assignments
and all relative phases survive. Only full verified L86 classes count.
"""
import argparse
from datetime import datetime
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import shutil
import sys
from time import perf_counter

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.benchmark_l86_discovery import discovery_key,freeze_sources,verified_inventory
from experiments.expand_doptimal43_norm import fixed_side_candidates
from experiments.search_perfect43_basis import perfect43_bases
from solver.checkpoint import atomic_write_json
from solver.interleaving import (interleave_bits,periodic_sign_profile,sparse_odd_matches,
                                best_sparse_odd_candidate)
from solver.sds_norm_completion import complete_sds_support,native_odd_cross_vectors
from solver.search_runner import append_verified_solution_if_new
from solver.solution_equivalence import pqcp_equivalence_key
from solver.structured_energy import target_profile
from solver.verifier import verify_pqcp


def bits(mask):return tuple((mask>>i)&1 for i in range(43))
def signs(mask):return tuple(1-2*v for v in bits(mask))


def even_ingredient_closure(bank):
    """Explicitly close E before an O unit/reflection quotient can be used."""
    seen={};all_bits=(1<<43)-1
    def rotation_key(mask):
        return min(((mask<<r)|(mask>>(43-r)))&all_bits for r in range(43))
    for index,item in enumerate(bank):
        e=tuple(tuple(w) for w in item['rows'])
        if tuple(map(sum,e))!=(11,7) or tuple(x+y for x,y in zip(*(periodic_sign_profile(w) for w in e)))!=(86,)+(2,)*42:
            raise ValueError('independently verified D-optimal43 sums 11/7 required')
        for q in range(1,43):
            for ra in (-1,1):
                for rb in (-1,1):
                    masks=tuple(rotation_key(sum(int(row[(q*r*i)%43]==-1)<<i for i in range(43)))
                                for row,r in zip(e,(ra,rb)))
                    seen.setdefault(masks,dict(masks=masks,parent=index,unit=q,reflections=(ra,rb)))
    for masks in seen:
        if tuple(x+y for x,y in zip(*(periodic_sign_profile(signs(m)) for m in masks)))!=(86,)+(2,)*42:
            raise RuntimeError('even ingredient closure violated full integer D-optimal identity')
    return tuple(seen.values())


@lru_cache(maxsize=4096)
def vectors(e,o):return native_odd_cross_vectors(e,o)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--even-input',type=Path,default=ROOT/'results/L86_new_doptimal_norm_trades2_20261007/report.json')
    parser.add_argument('--output',type=Path,default=ROOT/'results'/(
        'L86_legendre_norm_interleave_'+datetime.now().strftime('%Y%m%d_%H%M%S')))
    parser.add_argument('--trades',type=int,default=1)
    args=parser.parse_args()
    if not 0<=args.trades<=2:parser.error('trades must be 0..2')
    args.output.mkdir(parents=True,exist_ok=False);tick=perf_counter()
    snapshot=args.output/'source_snapshot';freeze_sources(snapshot)
    (snapshot/'experiments').mkdir(exist_ok=True)
    for source in (ROOT/'experiments').glob('*.py'):shutil.copy2(source,snapshot/'experiments'/source.name)
    manifest={str(p.relative_to(snapshot)):hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(snapshot.rglob('*')) if p.is_file() and p.name!='source_manifest.json'}
    atomic_write_json(snapshot/'source_manifest.json',manifest)
    shutil.copy2(args.even_input,args.output/'input_even_bank.json')
    bank=json.loads(args.even_input.read_text())
    if not bank['complete']:raise ValueError('completed even-ingredient bank required')
    even=even_ingredient_closure(bank['sds']);atomic_write_json(args.output/'even_closure.json',even)
    ingredients=perfect43_bases()
    parents=[dict(rows=(r['signs'],),source_index=i) for i,r in enumerate(ingredients['rows'])]
    pool=fixed_side_candidates(parents,args.trades)
    atomic_write_json(args.output/'fixed_odd_candidates.json',pool)
    raw=(ROOT/'86.txt').read_bytes() if (ROOT/'86.txt').exists() else b''
    (args.output/'initial_inventory.txt').write_bytes(raw);known=verified_inventory(raw)
    print(json.dumps(dict(event='generated',odd_fixed_sides=len(pool['rows']),even_pairs=len(even),
                          elapsed=perf_counter()-tick)),flush=True)
    seen=set();odd_bank=[];keys=set();first_hit=None;seeds=[];seed_keys=set()
    totals=dict(norm_visited=0,weight_matches=0,exact_odd_sds_models=0,odd_lookups=0,interleave_families=0)
    for number,item in enumerate(pool['rows']):
        out=complete_sds_support(item['fixed'],43,21,20)
        with (args.output/'completion_jobs.jsonl').open('a') as handle:
            handle.write(json.dumps(dict(candidate_index=number,fixed=item['fixed'],**out))+'\n')
        totals['norm_visited']+=out['visited'];totals['weight_matches']+=out['weight_matches']
        totals['exact_odd_sds_models']+=len(out['masks'])
        for v in out['masks']:
            pair=(item['fixed'],v);key=pqcp_equivalence_key(*(bits(m) for m in pair))
            if key in seen:continue
            seen.add(key)
            profile=tuple(x+y for x,y in zip(*(periodic_sign_profile(signs(m)) for m in pair)))
            if profile!=(86,)+(-2,)*42:raise RuntimeError('Legendre pair failed full original43 audit')
            odd_bank.append(dict(masks=pair,profile=profile,
                                 individually_perfect=all(periodic_sign_profile(signs(m))==(43,)+(-1,)*42 for m in pair)))
            for oa,ob in (pair,) if pair[0]==pair[1] else (pair,pair[::-1]):
                for e_item in even:
                    ea,eb=e_item['masks'];left,right=vectors(ea,oa),vectors(eb,ob)
                    totals['interleave_families']+=1;totals['odd_lookups']+=21*len(left)
                    for k,da,db in sparse_odd_matches(left,right):
                        words=(interleave_bits(signs(ea),signs(oa),da),interleave_bits(signs(eb),signs(ob),db,-1))
                        check=verify_pqcp(*words)
                        if not check.is_valid or tuple(check.profile)!=target_profile(86,k,1):
                            raise RuntimeError('Legendre/D-optimal hit failed independent full L86 target audit')
                        discovery=discovery_key(*words)
                        if discovery not in known:
                            if discovery not in keys and first_hit is None:first_hit=perf_counter()-tick
                            keys.add(discovery);append_verified_solution_if_new(86,*words,args.output)
                    best=best_sparse_odd_candidate(left,right)
                    if best is None or best['full_energy']==0:continue
                    if len(seeds)>=64 and best['full_energy']>=seeds[-1]['target_energy']:continue
                    words=(interleave_bits(signs(ea),signs(oa),best['left_preimage']),
                           interleave_bits(signs(eb),signs(ob),best['right_preimage'],-1))
                    check=verify_pqcp(*words);wanted=target_profile(86,best['k'],1)
                    if check.profile[43] or sum(((x-y)//4)**2 for x,y in zip(check.profile[1:44],wanted[1:44]))!=best['full_energy']:
                        raise RuntimeError('Legendre near-seed failed full original-length energy audit')
                    center_key=pqcp_equivalence_key(*words)
                    if center_key in seed_keys:continue
                    seed_keys.add(center_key)
                    seeds.append(dict(words=words,k=best['k'],sign=1,target_energy=best['full_energy'],
                                      even_masks=(ea,eb),odd_masks=(oa,ob),profile=check.profile))
                    seeds.sort(key=lambda r:r['target_energy']);del seeds[64:]
        if (number+1)%20==0 or number+1==len(pool['rows']):
            report=dict(complete=number+1==len(pool['rows']),completed_fixed_sides=number+1,
                        expected_fixed_sides=len(pool['rows']),even_pair_count=len(even),trades=args.trades,
                        odd_bank=odd_bank,seeds=seeds,**totals,
                        independently_verified_new_classes=len(keys),first_verified_elapsed=first_hit,
                        minimum_seed_energy=None if not seeds else seeds[0]['target_energy'],
                        elapsed=perf_counter()-tick,source_manifest_sha256=hashlib.sha256(
                            json.dumps(manifest,sort_keys=True).encode()).hexdigest(),
                        scope='All fixed-side Legendre completions in reported trade pool and ALL closed '
                              'even-ingredient/odd-assignment/relative-phase construction families. '
                              'Seed selection bounded; no global L86 exclusion or timing speed claim.')
            atomic_write_json(args.output/'report.json',report)
            print(json.dumps({k:report[k] for k in ('complete','completed_fixed_sides','interleave_families',
                  'minimum_seed_energy','independently_verified_new_classes','elapsed')},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
