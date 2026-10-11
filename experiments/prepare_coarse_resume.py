"""Audit a TERMINAL coarse run's fully closed anchor prefix for continuation.

Partial last-anchor work is not credited. Original ledgers/sources remain
in their original directory. No sign-model/proof cache is imported.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.complete_l86_full_rows import anchor_candidates
from solver.checkpoint import atomic_write_json


def file_sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def words_sha(inputs):
    return hashlib.sha256(json.dumps([s.get('words',s.get('coarse_rows')) for s in inputs['seeds']],sort_keys=True).encode()).hexdigest()


def closed_target_stats(rows,records):
    """Audit every whole target of the declared contiguous closed prefix."""
    n=len(rows);stats=dict(target_fibers_decided=0,support_orbits=0,sign_fibers=0,assignments=0,coarse_witness_supports=0)
    fresh=[r for r in rows if not r.get('closed_prefix_parent')]
    for row in rows:
        if row.get('closed_prefix_parent'):
            if 'proof_target_stats' not in row:raise ValueError('inherited anchor requires independently audited original target statistics')
            part=row['proof_target_stats']
            if part['target_fibers_decided']!=84:raise ValueError('inherited anchor proof does not cover every strict target')
            for key in stats:stats[key]+=part[key]
        else:row['proof_target_stats']={key:0 for key in stats}
    expected=0
    for record in records:
        if record['anchor']>=n:continue
        ordinal,within=divmod(expected,84);k,side=divmod(within,2)
        if ordinal>=len(fresh):raise ValueError('extra target record outside audited closed prefix')
        anchor=fresh[ordinal]['anchor']
        if (record['anchor']!=anchor or record['k']!=k+1 or record['eta']!=(-1,1)[side]
                or not record['complete'] or record['status']!='UNSAT_FIXED_SIGNED_C_A'
                or record['words'] is not None or record.get('unknown_sign_fibers',0)):
            raise ValueError('closed prefix target is missing, reordered, SAT, UNKNOWN or not exhaustive')
        row=rows[anchor]
        if any(record[key]!=row[key] for key in ('parent','side','flips')):
            raise ValueError('closed target literal anchor metadata disagrees with anchor ledger')
        stats['target_fibers_decided']+=1
        for key in ('support_orbits','sign_fibers','assignments','coarse_witness_supports'):stats[key]+=record.get(key,0)
        row['proof_target_stats']['target_fibers_decided']+=1
        for key in ('support_orbits','sign_fibers','assignments','coarse_witness_supports'):
            row['proof_target_stats'][key]+=record.get(key,0)
        expected+=1
    if expected!=84*len(fresh) or stats['target_fibers_decided']!=84*n:raise ValueError('closed prefix does not have every strict target')
    return stats


def audited_run_rows(parent):
    """Recursively audit inherited closed anchors without importing norm caches."""
    parent=Path(parent)
    with gzip.open(parent/'completed_anchors.jsonl.gz','rt') as stream:rows=[json.loads(line) for line in stream]
    for index,row in enumerate(rows):
        if row['anchor']!=index or row['target_fibers_decided']!=84 or row['statuses']!={'UNSAT_FIXED_SIGNED_C_A':84}:
            raise ValueError('only a contiguous fully exhausted UNSAT anchor prefix can be credited')
    if any(r.get('closed_prefix_parent') for r in rows):
        certificate=json.loads((parent/'closed_prefix_certificate.json').read_text());previous_parent=Path(certificate['parent'])
        if previous_parent.resolve()==parent.resolve():raise ValueError('cyclic closed proof reference')
        for name,digest in certificate['parent_sha256'].items():
            if file_sha(previous_parent/name)!=digest:raise ValueError('inherited original proof changed')
        previous,_=audited_run_rows(previous_parent)
        for row in rows:
            if not row.get('closed_prefix_parent'):continue
            original=row.get('closed_proof_original_anchor',row['anchor'])
            if (Path(row['closed_prefix_parent']).resolve()!=previous_parent.resolve() or not 0<=original<len(previous)
                    or row['anchor_key']!=previous[original]['anchor_key']):
                raise ValueError('inherited anchor is not the same certified literal coarse row')
            row['proof_target_stats']=dict(previous[original]['proof_target_stats'])
    with gzip.open(parent/'full_row_target_fibers.jsonl.gz','rt') as stream:
        stats=closed_target_stats(rows,(json.loads(line) for line in stream))
    snapshot=parent/'source_snapshot';manifest=json.loads((snapshot/'source_manifest.json').read_text())
    for relative,expected in manifest.items():
        if file_sha(snapshot/relative)!=expected:raise ValueError('frozen parent source manifest no longer matches')
    return rows,stats


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args();parent=args.parent.resolve()
    report=json.loads((parent/'report.json').read_text());inputs=json.loads((parent/'input_centers.json').read_text())
    if report['mode']!='coarse-row' or report['radius'] not in (0,1,2):
        raise ValueError('audited fixed coarse row region required')
    # Reading through gzip EOF requires a complete footer. Stop the owned
    # worker gracefully first; live/truncated gzip files fail this audit.
    rows,stats=audited_run_rows(parent)
    if report['complete'] and (len(rows)!=report['distinct_anchors'] or any(stats[key]!=report[key] for key in stats)):
        raise ValueError('terminal whole-region report differs from complete recursively audited anchor/target evidence')
    seen=set();count=0
    for center,item in enumerate(inputs['seeds']):
        for side,flips,_,_,key in anchor_candidates(item,'coarse-row',report['radius'],report.get('anchor_side')):
            if key in seen:continue
            seen.add(key)
            if count==len(rows):break
            row=rows[count]
            if (row['parent'],row['side'],row['flips'],row['anchor_key'])!=(center,side,json.loads(json.dumps(flips)),list(key)):
                raise ValueError('closed prefix differs from the frozen literal input/order')
            count+=1
        if count==len(rows):break
    if count!=len(rows):raise ValueError('too many closed anchors for input region')
    paths=('completed_anchors.jsonl.gz','full_row_target_fibers.jsonl.gz','signed_support_fibers.jsonl.gz',
           'input_centers.json','report.json','source_snapshot/source_manifest.json')
    hashes={name:file_sha(parent/name) for name in paths}
    args.output.mkdir(parents=True,exist_ok=False)
    atomic_write_json(args.output/'closed_anchors.json',rows)
    certificate=dict(parent=str(parent),mode='coarse-row',radius=report['radius'],
        direct_coarse_input=bool(report.get('direct_coarse_input',False)),anchor_side=report.get('anchor_side'),
        input_words_sha256=words_sha(inputs),closed_anchors=len(rows),stats=stats,parent_sha256=hashes,
        closed_anchors_sha256=file_sha(args.output/'closed_anchors.json'),
        scope='Whole literal anchor prefix; every 84-target UNSAT result audited. Unfinished last-anchor work not credited. Fresh norm proof cache required.')
    atomic_write_json(args.output/'certificate.json',certificate)
    print(json.dumps(certificate),flush=True)


if __name__=='__main__':main()
