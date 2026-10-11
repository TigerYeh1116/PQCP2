"""Audit new C_B carriers as successor C_A anchors, with exact exclusions.

A complete all-k fixed-C_A exclusion is invariant under common odd units,
independent A rotation, and replacement by a same-norm/same-support word.
Every selected row remains only a compressed witness, never a full PQCP.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from solver.checkpoint import atomic_write_json
from solver.coarse_anchor_orbits import coarse_anchor_key,coarse_word_orbit_key
from solver.coarse_row_completion import coarse_target,coarse_l1_anchors,validate_coarse_row
from solver.structured_energy import integer_periodic_autocorrelation_profile as pacf


def file_sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def closed_bank(path):
    certificate=json.loads((path/'certificate.json').read_text())
    parent=Path(certificate['parent']);report=json.loads((parent/'report.json').read_text())
    if (not report['complete'] or report['unknown_target_fibers'] or report['sat_target_fibers']
            or certificate['mode']!='coarse-row'):
        raise ValueError('complete, independently audited all-target coarse exclusion required')
    for name,digest in certificate['parent_sha256'].items():
        if file_sha(parent/name)!=digest:raise RuntimeError('closed proof parent file changed after audit')
    closed_rows=json.loads((path/'closed_anchors.json').read_text())
    if (file_sha(path/'closed_anchors.json')!=certificate['closed_anchors_sha256']
            or len(closed_rows)!=certificate['closed_anchors']):
        raise RuntimeError('closed literal anchor proof changed')
    closed_keys={}
    for row in closed_rows:
        if row['target_fibers_decided']!=84 or any(str(k).startswith(('UNKNOWN','SAT')) for k in row['statuses']):
            raise RuntimeError('one closed anchor lacks all 84 complete exclusions')
        c=tuple(row['anchor_key']);closed_keys.setdefault(coarse_anchor_key(c),
            dict(closed_anchor=row['anchor'],closed_proof=str(path.resolve())))
    return parent,closed_keys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--closed',type=Path,default=ROOT/'results/L86_phase_million_coarse_radius1_closed_20261008')
    parser.add_argument('--also-closed',type=Path,action='append',default=[])
    parser.add_argument('--radius',type=int,choices=(0,1,2),default=1,
                        help='Planned successor L1 balls. A closed centre alone never excludes its neighbours.')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('fresh output required')
    parent,closed_keys=closed_bank(args.closed)
    for path in args.also_closed:
        _,additional=closed_bank(path);closed_keys.update(additional)
    centers=json.loads((parent/'input_centers.json').read_text())['seeds']
    trace=parent/'signed_support_fibers.jsonl.gz';observations=[];rows=[];retained={};lines=0;sat=0;closed=0;duplicate=0
    args.output.mkdir(parents=True,exist_ok=False)
    with gzip.open(trace,'rb') as stream:
        for line in stream:
            lines+=1
            if b'SAT_COMPRESSED' not in line:continue
            source=json.loads(line);attempts=source.get('attempts',())
            carriers=[a['result'] for a in attempts if a.get('kind')=='C_B' and a['result']['status']=='SAT_COMPRESSED']
            if not carriers:continue
            if len(carriers)!=1:raise RuntimeError('ambiguous C_B carrier ledger entry')
            sat+=1
            ca=list(centers[source['parent']]['coarse_rows'][source['side']])
            for position,old,new in source['flips']:
                if ca[position]!=old:raise RuntimeError('anchor edit provenance mismatch')
                ca[position]=new
            ca=tuple(ca);cb=validate_coarse_row(carriers[0]['words'][0]);spec=coarse_target(ca,source['k'],source['eta'])
            if (spec is None or tuple(pacf(cb))!=tuple(spec['wanted_c_b'])
                    or cb.count(-1)!=spec['c_b_negative']
                    or sum((v!=0)<<j for j,v in enumerate(cb))!=source['mask']):
                raise RuntimeError('C_B carrier failed independent all-shift norm/count/support audit')
            tau=source['k']*22%43
            expected=(43,)+tuple(source['eta']*int(u in (tau,-tau%43)) for u in range(1,43))
            if tuple(x+y for x,y in zip(pacf(ca),pacf(cb)))!=expected:
                raise RuntimeError('combined signed C norm audit failed')
            # Norm/support equivalence transfers a FIXED point exclusion.
            # Whole balls require a literal signed-word isometry instead.
            key=coarse_anchor_key(cb) if args.radius==0 else coarse_word_orbit_key(cb)
            label={k:source[k] for k in ('anchor','parent','side','flips','k','eta')}
            covered=[]
            for point,_ in coarse_l1_anchors(cb,args.radius):
                proof=closed_keys.get(coarse_anchor_key(point))
                if proof is None:break
                covered.append(proof)
            else:
                closed+=1;observations.append(dict(source=label,status='ALREADY_CLOSED_ALL_K_L1_BALL',
                    radius=args.radius,covered_literal_points=len(covered),point_proofs=covered));continue
            if key in retained:
                duplicate+=1;observations.append(dict(source=label,status='DUPLICATE_SUCCESSOR_ANCHOR',source_row=retained[key]));continue
            retained[key]=len(rows)
            old_pair=tuple(tuple(word[22*j%43] for j in range(43)) for word in (ca,cb))
            original=(43,)+tuple(source['eta']*int(u in (source['k'],43-source['k'])) for u in range(1,43))
            if tuple(x+y for x,y in zip(pacf(old_pair[0]),pacf(old_pair[1])))!=original:
                raise RuntimeError('CRT-to-folded witness conversion failed')
            rows.append(dict(target=dict(L=86,k=source['k'],eta=source['eta']),
                result=dict(attempts=dict(coarse=dict(status='SAT_COMPRESSED',words=old_pair)),verified=False),
                successor_crt_row=cb,source=label))
            observations.append(dict(source=label,status='NEW_SUCCESSOR_ANCHOR',source_row=len(rows)-1))
    atomic_write_json(args.output/'coarse_witnesses.json',dict(rows=rows,planned_radius=args.radius,
        scope='Audited C_B successor centres. Whole-ball duplicates use only signed affine word isometries; no full binary solutions.'))
    sources=['experiments/extract_coarse_successor_anchors.py','solver/coarse_anchor_orbits.py','csrc/coarse_anchor_orbits.c']
    result=dict(complete=True,signed_ledger_lines=lines,audited_coarse_carriers=sat,
        already_closed_all_k_carriers=closed,duplicate_successor_carriers=duplicate,new_successor_anchors=len(rows),
        planned_radius=args.radius,centre_dedup_equivalence='NORM_SUPPORT_POINT' if args.radius==0 else 'SIGNED_AFFINE_WORD_ISOMETRY',
        closed_anchor_keys=len(closed_keys),closed_proof=str(args.closed.resolve()),closed_proof_sha256=file_sha(args.closed/'certificate.json'),
        additional_closed_proofs=[dict(path=str(path.resolve()),certificate_sha256=file_sha(path/'certificate.json')) for path in args.also_closed],
        observations=observations,source_sha256={name:file_sha(ROOT/name) for name in sources},
        claim='New mathematically eligible anchor domains only; no full PQCP discovery or speed claim.')
    atomic_write_json(args.output/'report.json',result)
    print(json.dumps({k:result[k] for k in ('complete','audited_coarse_carriers','already_closed_all_k_carriers','duplicate_successor_carriers','new_successor_anchors')}),flush=True)


if __name__=='__main__':main()
