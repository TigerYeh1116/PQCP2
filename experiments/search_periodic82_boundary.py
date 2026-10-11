"""Complete L86 append-four search over two printed periodic-Golay82 orbits.

The mod-8 boundary projection is derived here for the strict nonzero target.
No aperiodic-cancellation assumption is imported from Golay-80 experiments.
Raw source data are independently audited at length 82 before any search.
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
from experiments.benchmark_l86_discovery import discovery_key, freeze_sources, verified_inventory
from solver.boundary_tail_search import search_boundary_tail_masks
from solver.checkpoint import atomic_write_json
from solver.correlation import full_correlation_profile
from solver.search_runner import append_verified_solution_if_new

PAPER='https://arxiv.org/pdf/1310.5773'
H=(1,37,51,57,59)
PARENTS=(((1,2,11,12,15,17,22,23,30),(1,4,10,12,17,22,23,41)),
         ((1,2,3,6,8,12,17,23,30),(3,5,6,12,17,22,30,41)))


def periodic82_parents():
    rows=[]
    for j,k in PARENTS:
        supports=tuple(sorted({h*x%82 for h in H for x in subset}) for subset in (j,k))
        if tuple(map(len,supports))!=(45,36):raise RuntimeError('printed orbit support sizes disagree')
        words=tuple(tuple(int(i in support) for i in range(82)) for support in supports)
        profile=tuple(full_correlation_profile(*words))
        if profile!=(164,)+(0,)*81:
            raise RuntimeError('printed length-82 source is not independently periodic complementary')
        rows.append(dict(j=j,k=k,supports=supports,words=words,
                         masks=tuple(sum(v << i for i,v in enumerate(w)) for w in words),profile=profile))
    return tuple(rows)


def rotation_masks(mask,length=82):
    all_bits=(1 << length)-1
    return tuple(((mask << r)|(mask >> (length-r))) & all_bits for r in range(length))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'results'/(
        'L86_periodic82_boundary_'+datetime.now().strftime('%Y%m%d_%H%M%S')))
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    tick=perf_counter()
    snapshot=args.output/'source_snapshot'
    freeze_sources(snapshot)
    for name in ('search_periodic82_boundary.py','benchmark_l86_discovery.py'):
        destination=snapshot/'experiments'/name
        destination.parent.mkdir(exist_ok=True)
        shutil.copy2(ROOT/'experiments'/name,destination)
    manifest={str(p.relative_to(snapshot)):hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(snapshot.rglob('*')) if p.is_file() and p.name!='source_manifest.json'}
    atomic_write_json(snapshot/'source_manifest.json',manifest)
    inventory=(ROOT/'86.txt').read_bytes() if (ROOT/'86.txt').exists() else b''
    (args.output/'initial_inventory.txt').write_bytes(inventory)
    known=verified_inventory(inventory)
    parents=periodic82_parents()
    units=tuple(range(1,41,2))
    records,keys,first_hit=[],set(),None
    for parent_index,parent in enumerate(parents):
        for q in units:
            decimated=tuple(tuple(w[q*i%82] for i in range(82)) for w in parent['words'])
            if tuple(full_correlation_profile(*decimated))!=(164,)+(0,)*81:
                raise RuntimeError('common unit decimation failed independent source audit')
            phases=tuple(rotation_masks(sum(v << i for i,v in enumerate(w))) for w in decimated)
            totals=Counter()
            ranks=Counter()
            witnesses=[]
            for ra,a in enumerate(phases[0]):
                for rb,b in enumerate(phases[1]):
                    out=search_boundary_tail_masks((a,b),86,4)
                    if not out['complete']:raise RuntimeError('boundary query failed complete tail coverage')
                    ranks[out['rank']]+=1
                    totals['prefix_pairs']+=1
                    for name in ('covered_tail_assignments','target_groups','excluded_groups',
                                 'surviving_tail_assignments','correlation_checks'):
                        totals[name]+=out[name]
                    for hit in out['witnesses']:
                        key=discovery_key(*hit['words'])
                        if key not in known:
                            if key not in keys and first_hit is None:first_hit=perf_counter()-tick
                            keys.add(key)
                            append_verified_solution_if_new(86,*hit['words'],args.output)
                        witnesses.append(dict(rotation_a=ra,rotation_b=rb,**hit))
            row=dict(parent=parent_index,common_unit=q,totals=dict(totals),ranks=dict(ranks),witnesses=witnesses)
            records.append(row)
            report=dict(parents=parents,rows=records,complete=len(records)==len(parents)*len(units),
                        expected_prefix_pairs=2*20*82**2,
                        independently_verified_new_classes=len(keys),first_verified_elapsed=first_hit,
                        elapsed=perf_counter()-tick,paper=PAPER,
                        source_manifest_sha256=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest(),
                        scope='ALL arbitrary four-bit tails on both rows over common units and independent '
                              'rotations of these TWO printed periodic-82 parents. Other symmetry images '
                              'covered via full-L86 row complement/reversal/shift/swap and simultaneous alternation. '
                              'Not all periodic82 parents or general L86 PQCPs.',
                        speed_claim='No new-class speed claim from excluded groups or finite-family exhaustion.')
            atomic_write_json(args.output/'report.json',report)
            print(json.dumps(dict(parent=parent_index,q=q,**dict(totals),new_classes=len(keys),
                                  elapsed=perf_counter()-tick)),flush=True)


if __name__=='__main__':main()
