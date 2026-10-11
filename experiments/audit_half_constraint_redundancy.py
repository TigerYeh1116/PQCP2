"""Exact constraint-equivalence audit of an entire four-half radius-three region.

For one unknown half, the other full row enters only through its whole PACF.
The known companion can rotate/reverse/complement with an inverse change of
the entirely free half. Common unit decimation acts on both constraints.
Equal keys prove identical SAT/UNSAT status, not equivalence of every full
pair: a homometric fixed full row may produce a distinct novel PQCP.
Only completely UNSAT fibers could safely be skipped using this relation.
"""
import argparse
from datetime import datetime
from functools import lru_cache
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys
from time import perf_counter

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.complete_two_l86_halves import packed_profile
from experiments.benchmark_l86_discovery import freeze_sources
from solver.checkpoint import atomic_write_json
from solver.solution_equivalence import _row_key
from solver.three_half_row_completion import four_half_row_fibers,four_half_triple_count


@lru_cache(maxsize=30000)
def canonical_full_profile(even,odd):
    row=tuple(v for pair in zip(even,odd) for v in pair)
    profile=tuple(v//2 for v in packed_profile((row,row)))
    length=len(row);d=length//2;choices=[]
    for q in range(1,d,2):
        values=bytes(profile[q*u%length]+128 for u in range(1,d+1))
        choices.append((values,q))
    smallest=min(v for v,q in choices)
    return smallest,tuple(q for v,q in choices if v==smallest)


@lru_cache(maxsize=20000)
def companion_key(row,q):
    return _row_key(''.join(str(row[q*i%len(row)]) for i in range(len(row))))


def constraint_key(halves,free):
    d=len(halves[(free+1)%4]);side=free//2;other=1-side
    profile,qs=canonical_full_profile(halves[2*other],halves[2*other+1])
    known=halves[2*side+(1-free%2)]
    return d,profile,min(companion_key(known,q) for q in qs)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=ROOT/'results/L86_latest_two_halves_radius2_top8_20261008/input_centers.json')
    parser.add_argument('--output',type=Path,default=ROOT/'results'/('L86_constraint_orbit_audit_'+datetime.now().strftime('%Y%m%d_%H%M%S')))
    args=parser.parse_args();inputs=json.loads(args.input.read_text());tick=perf_counter()
    args.output.mkdir(parents=True,exist_ok=False);shutil.copy2(args.input,args.output/'input_centers.json')
    digest=freeze_sources(args.output/'source_snapshot');shutil.copy2(Path(__file__),args.output/'audit_source.py')
    keys={};rows=[];eligible=0;duplicate_count=0
    with gzip.open(args.output/'duplicate_constraint_certificates.jsonl.gz','wt',compresslevel=1) as ledger:
        for index,item in enumerate(inputs['seeds']):
            count=raw=0;local=set()
            for job in four_half_row_fibers(*item['words'],radius=3):
                raw+=1
                if job['kind']!='ELIGIBLE':continue
                count+=1;k=constraint_key(job['halves'],job['free']);local.add(k)
                origin=dict(center=index,ordinal=job['ordinal'],free=job['free'],fixed_masks=job['fixed_masks'])
                if k in keys:
                    duplicate_count+=1
                    ledger.write(json.dumps(dict(first=keys[k],duplicate=origin,
                         key=dict(d=k[0],full_row_canonical_pacf_hex=k[1].hex(),known_companion=k[2])))+'\n')
                else:keys[k]=origin
            if raw!=four_half_triple_count(43,3):raise RuntimeError('audit did not traverse its entire declared region')
            eligible+=count;rows.append(dict(center=index,raw=raw,eligible=count,distinct_constraints=len(local)))
            print(json.dumps(dict(rows[-1],elapsed=perf_counter()-tick)),flush=True)
    report=dict(complete=True,radius=3,varied_halves=3,rows=rows,eligible=eligible,
        distinct_constraints=len(keys),duplicate_constraints=duplicate_count,
        redundancy_fraction=duplicate_count/eligible,elapsed=perf_counter()-tick,source_manifest_sha256=digest,
        audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='All four free roles and all radius-three nonzero masks in the other three physical halves at frozen input centers. '
              'Keys use the entire fixed full-row PACF and literal known companion modulo justified phase/sign/decimation mappings. '
              'No profile truncation or hash-only equality. This is a redundancy audit, not an exclusion or new-class discovery benchmark.',
        caution='Do not discard SAT fibers: different homometric fixed full rows can yield different novel full PQCP classes.')
    atomic_write_json(args.output/'report.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
