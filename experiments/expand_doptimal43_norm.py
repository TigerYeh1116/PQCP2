"""New D-optimal-43 ingredients by complete norm completion, then L86 joins.

IDE default expands every single swap of the verified published row-sum 11/7
parents. For each distinct U, ALL V of weight 18 with constant intersection
sum 13 are decided. Every exact SDS is interleaved against ALL perfect43 O
ingredients and all relative phases. Only independently verified full L86
classes count as discoveries. No global exclusion or speed guarantee.
"""
import argparse
from collections import defaultdict
from datetime import datetime
import hashlib
from itertools import combinations
import json
from pathlib import Path
import shutil
import sys
from time import perf_counter

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.benchmark_l86_discovery import discovery_key,freeze_sources,verified_inventory
from experiments.search_doptimal43_interleave import read_source
from experiments.search_perfect43_basis import perfect43_bases
from solver.checkpoint import atomic_write_json
from solver.interleaving import (interleave_bits,odd_cross_vectors,periodic_sign_profile,
                                sparse_odd_matches,best_sparse_odd_candidate)
from solver.sds_norm_completion import complete_sds_support,sds_ingredient_key
from solver.search_runner import append_verified_solution_if_new
from solver.solution_equivalence import pqcp_equivalence_key
from solver.structured_energy import target_profile
from solver.verifier import verify_pqcp

DEFAULT_TRADES=1


def fixed_side_candidates(parents,trades):
    candidates={};literal=0
    for parent in parents:
        row=parent['rows'][0];mask=sum(int(v==-1)<<i for i,v in enumerate(row))
        negative=tuple(i for i,v in enumerate(row) if v==-1)
        positive=tuple(i for i,v in enumerate(row) if v==1)
        for count in range(trades+1):
            for removed in combinations(negative,count):
                for added in combinations(positive,count):
                    literal+=1
                    fixed=mask^sum(1<<i for i in removed+added)
                    key=sds_ingredient_key(fixed,43)
                    candidates.setdefault(key,dict(fixed=fixed,parent_source_index=parent['source_index'],
                                                  removed=removed,added=added,ingredient_key=key))
    return dict(rows=list(candidates.values()),literal_trades=literal,
                scope='All weight-preserving U swaps up to selected radius from these parents, '
                      'quotiented by common units/translations with ALL V completions retained.')


def interleave_sds(e,o_rows):
    def index(row):
        out=defaultdict(list)
        for i,o in enumerate(o_rows):
            for vector,offsets in odd_cross_vectors(row,o).items():out[vector].extend((i,s) for s in offsets)
        return dict(out)
    left,right=(index(row) for row in e)
    hits=[]
    for k,(ia,da),(ib,db) in sparse_odd_matches(left,right):
        words=(interleave_bits(e[0],o_rows[ia],da),interleave_bits(e[1],o_rows[ib],db,-1))
        check=verify_pqcp(*words)
        if not check.is_valid or tuple(check.profile)!=target_profile(86,k,1):
            raise RuntimeError('new SDS interleave failed independent full L86 target audit')
        hits.append(dict(words=words,k=k,eta=1,o_indices=(ia,ib),offsets=(da,db)))
    best=best_sparse_odd_candidate(left,right)
    if best is not None:
        ia,da=best['left_preimage'];ib,db=best['right_preimage']
        words=(interleave_bits(e[0],o_rows[ia],da),interleave_bits(e[1],o_rows[ib],db,-1))
        check=verify_pqcp(*words);wanted=target_profile(86,best['k'],1)
        if check.profile[43] or sum(((x-y)//4)**2 for x,y in zip(check.profile[1:44],wanted[1:44]))!=best['full_energy']:
            raise RuntimeError('best SDS seed failed original full L86 energy audit')
        best=dict(words=words,k=best['k'],sign=1,target_energy=best['full_energy'],
                  o_indices=(ia,ib),offsets=(da,db),profile=check.profile)
    return hits,len(left)*21,best


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'results/L86_doptimal_source_20261007.html')
    parser.add_argument('--output',type=Path,default=ROOT/'results'/(
        'L86_new_doptimal_norm_'+datetime.now().strftime('%Y%m%d_%H%M%S')))
    parser.add_argument('--trades',type=int,default=DEFAULT_TRADES)
    args=parser.parse_args()
    if not 0<=args.trades<=2:parser.error('trades must be 0..2')
    args.output.mkdir(parents=True,exist_ok=False);tick=perf_counter()
    snapshot=args.output/'source_snapshot';freeze_sources(snapshot)
    for name in ('expand_doptimal43_norm.py','search_perfect43_basis.py','search_doptimal43_interleave.py','benchmark_l86_discovery.py'):
        destination=snapshot/'experiments'/name;destination.parent.mkdir(exist_ok=True)
        shutil.copy2(ROOT/'experiments'/name,destination)
    manifest={str(p.relative_to(snapshot)):hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(snapshot.rglob('*')) if p.is_file() and p.name!='source_manifest.json'}
    atomic_write_json(snapshot/'source_manifest.json',manifest)
    shutil.copy2(args.source,args.output/'input_source.html')
    raw=(ROOT/'86.txt').read_bytes() if (ROOT/'86.txt').exists() else b''
    (args.output/'initial_inventory.txt').write_bytes(raw);known=verified_inventory(raw)
    source,rejected=read_source(args.source);parents=[r for r in source if r['weights']==(16,18)]
    initial_sds={pqcp_equivalence_key(*(tuple((1-v)//2 for v in row) for row in p['rows'])) for p in parents}
    o_rows=tuple(tuple(r['signs']) for r in perfect43_bases()['rows'])
    candidates=fixed_side_candidates(parents,args.trades)
    atomic_write_json(args.output/'fixed_side_candidates.json',candidates)
    print(json.dumps(dict(event='generated',distinct_fixed=len(candidates['rows']),
                          literal_trades=candidates['literal_trades'],elapsed=perf_counter()-tick)),flush=True)
    jobs=[];sds=[];seen_sds=set();keys=set();first_hit=None;lookups=0
    for number,row in enumerate(candidates['rows']):
        out=complete_sds_support(row['fixed'],43,18,13)
        for v in out['masks']:
            e=tuple(tuple(-1 if mask>>i&1 else 1 for i in range(43)) for mask in (row['fixed'],v))
            if (tuple(map(sum,e))!=(11,7) or tuple(x+y for x,y in zip(*(periodic_sign_profile(w) for w in e)))!=(86,)+(2,)*42):
                raise RuntimeError('native SDS failed full original43 sign-PACF audit')
            sds_key=pqcp_equivalence_key(*(tuple((1-x)//2 for x in w) for w in e))
            if sds_key in seen_sds:continue
            seen_sds.add(sds_key)
            hits,queries,best=interleave_sds(e,o_rows);lookups+=queries
            sds.append(dict(masks=(row['fixed'],v),rows=e,new_relative_to_published_parents=sds_key not in initial_sds,
                            hits=hits,odd_lookups=queries,best_seed=best))
            for hit in hits:
                key=discovery_key(*hit['words'])
                if key not in known:
                    if key not in keys and first_hit is None:first_hit=perf_counter()-tick
                    keys.add(key);append_verified_solution_if_new(86,*hit['words'],args.output)
        jobs.append(dict(candidate_index=number,**out))
        if (number+1)%1000==0 or number+1==len(candidates['rows']):
            report=dict(complete=len(jobs)==len(candidates['rows']),trades=args.trades,expected_jobs=len(candidates['rows']),
                        jobs=jobs,sds=sds,seeds=[r['best_seed'] for r in sds if r['best_seed'] is not None],
                        rejected_source_pairs=rejected,unique_sds=len(sds),
                        new_sds_relative_to_published_parents=sum(r['new_relative_to_published_parents'] for r in sds),
                        exact_odd_lookups=lookups,independently_verified_new_classes=len(keys),
                        first_verified_elapsed=first_hit,elapsed=perf_counter()-tick,
                        source_manifest_sha256=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest(),
                        scope='Complete SDS V completions for reported U trade pool, then ALL perfect O/relative-phase '
                              'interleavings per SDS symmetry class. New SDS ingredients are NOT full L86 discoveries.')
            atomic_write_json(args.output/'report.json',report)
            print(json.dumps(dict({k:report[k] for k in ('complete','unique_sds','new_sds_relative_to_published_parents',
                 'independently_verified_new_classes','elapsed')},completed_jobs=len(jobs))),flush=True)


if __name__=='__main__':main()
