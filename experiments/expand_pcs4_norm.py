"""Complete fourth-row PCS43 norm lifts with all strict L86 odd-phase joins.

Three fixed rows may have arbitrary correlations; only FOUR-row sum is zero.
Every V of weight 21 with the required integer profile is decided. Full
relative reflections, O assignments and phase preimages are retained before
counting verified L86 discoveries. No homometric fixed triples are merged.
"""
import argparse
from collections import OrderedDict
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
from experiments.search_gs43_interleave import read_source
from solver.checkpoint import atomic_write_json
from solver.interleaving import periodic_sign_profile
from solver.pcs4_interleave import mask_signs,pcs4_ingredient_key,search_pcs4_interleave
from solver.sds_norm_completion import complete_binary_intersection_profile,intersection_profile
from solver.search_runner import append_verified_solution_if_new
from solver.solution_equivalence import pqcp_equivalence_key
from solver.support_orbit_cache import row_dihedral_mask

DEFAULT_TRADES=1


def load_roots(gs_path,catalog_path,extra_paths=()):
    roots={}
    def add(quad,origin):
        quad=tuple(quad)
        if tuple(bin(m).count('1') for m in quad)!=(16,18,21,21):raise ValueError('PCS43 16/18/21/21 rows required')
        profiles=tuple(periodic_sign_profile(mask_signs(m,43)) for m in quad)
        if tuple(map(sum,zip(*profiles)))!=(172,)+(0,)*42:raise ValueError('root failed full original43 summed PACF audit')
        roots.setdefault(pcs4_ingredient_key(quad),dict(masks=quad,origin=origin))
    for family in read_source(gs_path):
        if sorted(family['sums'])!=[1,1,7,11]:continue
        rows=tuple(family['rows'][family['sums'].index(s)] for s in (11,7))+tuple(w for w in family['rows'] if sum(w)==1)
        add(tuple(sum(int(v==-1)<<i for i,v in enumerate(w)) for w in rows),'GS appendix '+str(family['appendix_index']))
    if catalog_path is not None:
        raw=json.loads(catalog_path.read_text())
        if not raw['complete_catalog_join']:raise ValueError('completed row catalog required')
        for number,quad in enumerate(raw['quadruples']):add(quad,'previous catalog '+str(number))
    for path in extra_paths:
        raw=json.loads(path.read_text())
        if not raw['complete']:raise ValueError('completed extra PCS4 bank required')
        for number,item in enumerate(raw['quads']):add(item['masks'],str(path)+' quad '+str(number))
    return tuple(roots.values())


def triple_pool(roots,trades,free_rows=(0,1,2,3)):
    pool={};literal=0
    for root_index,root in enumerate(roots):
        for free in free_rows:
            fixed_sides=tuple(i for i in range(4) if i!=free)
            fixed=tuple(root['masks'][i] for i in fixed_sides)
            for position,side in enumerate(fixed_sides):
                mask=fixed[position];negative=tuple(i for i in range(43) if mask>>i&1)
                positive=tuple(i for i in range(43) if not mask>>i&1)
                for count in range(0 if position==0 else 1,trades+1):
                    for removed in combinations(negative,count):
                        for added in combinations(positive,count):
                            literal+=1;changed=list(fixed);changed[position]^=sum(1<<i for i in removed+added)
                            key=tuple(row_dihedral_mask(m,43) for m in changed)
                            pool.setdefault((free,*key),dict(masks=key,free=free,fixed_sides=fixed_sides,
                                parent=root_index,side=side,removed=removed,added=added))
    return dict(rows=list(pool.values()),literal_trades=literal,
                scope='All up-to-selected swaps on one fixed row of reported roots; all fourth rows free. '
                      'Independent fixed-row dihedral quotient is covered by relative-reflection phase images.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gs-source',type=Path,default=ROOT/'results/L43_GS_source_20261007.html')
    parser.add_argument('--catalog',type=Path,default=ROOT/'results/L86_interleave_catalog_20261007/catalog.json')
    parser.add_argument('--include-catalog',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--extra-roots',type=Path,nargs='+',default=())
    parser.add_argument('--trades',type=int,default=DEFAULT_TRADES)
    parser.add_argument('--free-rows',type=int,nargs='+',default=(0,1,2,3))
    parser.add_argument('--output',type=Path,default=ROOT/'results'/(
        'L86_pcs4_norm_'+datetime.now().strftime('%Y%m%d_%H%M%S')))
    args=parser.parse_args()
    if not 0<=args.trades<=2:parser.error('trades must be 0..2')
    if len(set(args.free_rows))!=len(args.free_rows) or any(i not in range(4) for i in args.free_rows):
        parser.error('distinct free rows 0..3 required')
    args.output.mkdir(parents=True,exist_ok=False);tick=perf_counter()
    snapshot=args.output/'source_snapshot';freeze_sources(snapshot);(snapshot/'experiments').mkdir(exist_ok=True)
    for path in (ROOT/'experiments').glob('*.py'):shutil.copy2(path,snapshot/'experiments'/path.name)
    manifest={str(p.relative_to(snapshot)):hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(snapshot.rglob('*')) if p.is_file() and p.name!='source_manifest.json'}
    atomic_write_json(snapshot/'source_manifest.json',manifest)
    shutil.copy2(args.gs_source,args.output/'input_gs.html')
    if args.include_catalog:shutil.copy2(args.catalog,args.output/'input_catalog.json')
    for number,path in enumerate(args.extra_roots):shutil.copy2(path,args.output/('input_extra_roots_'+str(number)+'.json'))
    roots=load_roots(args.gs_source,args.catalog if args.include_catalog else None,args.extra_roots)
    pool=triple_pool(roots,args.trades,args.free_rows)
    atomic_write_json(args.output/'roots.json',roots);atomic_write_json(args.output/'fixed_triples.json',pool)
    raw=(ROOT/'86.txt').read_bytes() if (ROOT/'86.txt').exists() else b''
    (args.output/'initial_inventory.txt').write_bytes(raw);known=verified_inventory(raw)
    original={pcs4_ingredient_key(r['masks']) for r in roots}
    cache=OrderedDict();seen=set();quads=[];keys=set();first_hit=None;seeds=[];seed_keys=set()
    totals=dict(actual_norm_visited=0,logical_norm_covered=0,profile_cache_hits=0,
                fixed_triples_with_completions=0,odd_lookups=0)
    print(json.dumps(dict(event='generated',roots=len(roots),fixed_triples=len(pool['rows']),
                          literal_trades=pool['literal_trades'],elapsed=perf_counter()-tick)),flush=True)
    for number,item in enumerate(pool['rows']):
        fixed=tuple(item['masks']);profiles=tuple(intersection_profile(m,43) for m in fixed)
        wanted=((16,18,21,21)[item['free']],)+tuple(33-sum(p[u] for p in profiles) for u in range(1,43))
        reused=wanted in cache
        if reused:
            totals['profile_cache_hits']+=1;out=cache.pop(wanted);cache[wanted]=out
        else:
            out=complete_binary_intersection_profile(wanted);cache[wanted]=out
            if len(cache)>512:cache.popitem(last=False)
            totals['actual_norm_visited']+=out['visited']
        totals['logical_norm_covered']+=out['visited'];totals['fixed_triples_with_completions']+=bool(out['masks'])
        with (args.output/'completion_jobs.jsonl').open('a') as handle:
            handle.write(json.dumps(dict(candidate_index=number,reused_profile=reused,result=out))+'\n')
        for v in out['masks']:
            quad=[0]*4;quad[item['free']]=v
            for index,mask in zip(item['fixed_sides'],fixed):quad[index]=mask
            quad=tuple(quad);ingredient=pcs4_ingredient_key(quad)
            if ingredient in seen:continue
            seen.add(ingredient);match=search_pcs4_interleave(quad)
            totals['odd_lookups']+=match['odd_lookups']
            quads.append(dict(masks=quad,candidate_index=number,new_relative_to_input_roots=ingredient not in original,
                              match=match))
            for hit in match['witnesses']:
                discovery=discovery_key(*hit['words'])
                if discovery not in known:
                    if discovery not in keys and first_hit is None:first_hit=perf_counter()-tick
                    keys.add(discovery);append_verified_solution_if_new(86,*hit['words'],args.output)
            for seed in match['seeds']:
                if len(seeds)>=64 and seed['target_energy']>=seeds[-1]['target_energy']:continue
                key=pqcp_equivalence_key(*seed['words'])
                if key in seed_keys:continue
                seed_keys.add(key);seeds.append(dict(seed,quad_masks=quad))
                seeds.sort(key=lambda r:r['target_energy']);del seeds[64:]
        if (number+1)%1000==0 or number+1==len(pool['rows']):
            report=dict(complete=number+1==len(pool['rows']),trades=args.trades,free_rows=args.free_rows,roots=len(roots),
                        completed_fixed_triples=number+1,expected_fixed_triples=len(pool['rows']),
                        quads=quads,seeds=seeds,**totals,
                        new_ingredient_classes=sum(r['new_relative_to_input_roots'] for r in quads),
                        independently_verified_new_classes=len(keys),first_verified_elapsed=first_hit,
                        minimum_seed_energy=None if not seeds else seeds[0]['target_energy'],
                        elapsed=perf_counter()-tick,source_manifest_sha256=hashlib.sha256(
                            json.dumps(manifest,sort_keys=True).encode()).hexdigest(),
                        scope='Complete fourth-row lifts of reported fixed triples, all required relative reflections '
                              'and phases. Scalar E/O pair sums need not be constant. Cached profiles retain '
                              'every literal fixed-triple preimage. No global exclusion or discovery-speed claim.')
            atomic_write_json(args.output/'report.json',report)
            print(json.dumps({k:report[k] for k in ('complete','completed_fixed_triples','new_ingredient_classes',
                  'minimum_seed_energy','independently_verified_new_classes','elapsed')},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
