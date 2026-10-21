"""IDE Run: one complete row anchored, all 86 bits in the other row free.

Newest root long-test log, eight independently audited non-solution pairs,
each A/B anchor and its one-bit neighbors; all strict k/eta. Frozen inputs
and sources, full support/sign ledgers, first independently new full class.
"""
import argparse
from collections import Counter,deque
from contextlib import nullcontext
from datetime import datetime
import gzip
import hashlib
from itertools import combinations
import json
from math import gcd
from pathlib import Path
import shutil
import sys
from time import perf_counter

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.benchmark_l86_discovery import freeze_sources,verified_inventory,discovery_key
from experiments.complete_l86_half_rows import audited_log_centers
from experiments.complete_two_l86_halves import select_json_centers
from solver.checkpoint import atomic_write_json
from solver.full_row_completion import complete_fixed_binary_row_target,crt_trit_rows
from solver.coarse_row_completion import complete_fixed_coarse_row_target,CompleteUnaryNormUnsatCache,coarse_l1_anchors,validate_coarse_row
from solver.unary_norm_carrier_cache import UnaryNormCarrierCache
from solver.structured_energy import integer_periodic_autocorrelation_profile as pacf
from solver.search_runner import append_verified_solution_if_new
from solver.solution_equivalence import _row_key


def anchor_orbit_key(row):
    # B is wholly free and ALL strict targets are included. Common units
    # can therefore normalize A with an inverse action on every B model.
    length=len(row)
    return min(_row_key(''.join(str(row[q*i%length]) for i in range(length)))
               for q in range(1,length) if gcd(q,length)==1)


def audited_coarse_centers(path,top):
    """Audit actual signed compressed witnesses, never label them full PQCPs."""
    data=json.loads(path.read_text());seeds=[]
    for index,item in enumerate(data['rows'][:top or None]):
        old=tuple(tuple(r) for r in item['result']['attempts']['coarse']['words'])
        target=item['target'];k=target['k'];eta=target['eta']
        if (target['L']!=86 or len(old)!=2 or any(len(r)!=43 or any(type(v) is not int or v not in (-1,0,1) for v in r) for r in old)
                or type(k) is not int or not 1<=k<=42 or eta not in (-1,1)):
            raise ValueError('literal compressed 43-point witnesses and strict original target required')
        norm=tuple(x+y for x,y in zip(pacf(old[0]),pacf(old[1])))
        expected=(43,)+tuple(eta*int(u in (k,43-k)) for u in range(1,43))
        if norm!=expected:raise ValueError('archived coarse pair failed independent all-shift integer norm audit')
        crt=tuple(tuple(r[2*j%43] for j in range(43)) for r in old)
        tau=k*22%43
        if tuple(x+y for x,y in zip(pacf(crt[0]),pacf(crt[1])))!=(43,)+tuple(eta*int(u in (tau,-tau%43)) for u in range(1,43)):
            raise RuntimeError('old-to-CRT coarse indexing audit failed')
        seeds.append(dict(coarse_rows=crt,source_row=index,original_target=target,
            source_full_verified=bool(item['result'].get('verified',False))))
    if not seeds:raise ValueError('coarse input has no audited witnesses')
    return dict(seeds=seeds,source=str(path),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                scope='Actual signed compressed norm witnesses; no full binary PQCP assumption.')


def anchor_candidates(item,mode,radius,anchor_side=None):
    if anchor_side is not None and (type(anchor_side) is not int or anchor_side not in (0,1)):
        raise ValueError('source anchor side must be 0, 1 or None')
    if 'coarse_rows' in item:
        for side,original in enumerate(item['coarse_rows']):
            if anchor_side is not None and side!=anchor_side:continue
            for coarse,edits in coarse_l1_anchors(original,radius):yield side,edits,None,coarse,coarse
        return
    for side,original in enumerate(item['words']):
        if anchor_side is not None and side!=anchor_side:continue
        for distance in range(radius+1):
            for flips in combinations(range(86),distance):
                a=list(original)
                for i in flips:a[i]^=1
                a=tuple(a);coarse=crt_trit_rows(a)[0]
                yield side,flips,a,coarse,anchor_orbit_key(a) if mode=='full-row' else coarse


def audited_coarse_points(path, top):
    """Literal ternary points need no fictitious C_B witness to be searched."""
    packet = json.loads(path.read_text())
    if packet.get('format') != 'literal_coarse_points_v1' or packet.get('planned_radius') != 0:
        raise ValueError('radius-zero literal coarse point bank required')
    seeds = []
    for index, item in enumerate(packet['points'][:top or None]):
        row = validate_coarse_row(item['row'])
        if len(row) != 43:
            raise ValueError('length-43 literal coarse anchor required')
        seeds.append(dict(coarse_rows=(row,), source_point=index, provenance=item.get('provenance')))
    if not seeds:
        raise ValueError('literal coarse point bank is empty')
    return dict(seeds=seeds, source=str(path), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                scope='Literal fixed C_A points only; no compressed C_B witness or full PQCP assumed.')


def main(*,default_mode='full-row',default_radius=1):
    parser=argparse.ArgumentParser(description=__doc__);source=parser.add_mutually_exclusive_group()
    source.add_argument('--input',type=Path);source.add_argument('--log',type=Path);source.add_argument('--coarse-input',type=Path)
    source.add_argument('--coarse-anchors',type=Path,help='Literal C_A point bank; no C_B witness required, radius must be zero')
    parser.add_argument('--top',type=int,default=8);parser.add_argument('--radius',type=int,default=default_radius)
    parser.add_argument('--anchor-side',type=int,choices=(0,1),default=None,
                        help='Select one source row as the fixed anchor; default searches both. This selects a domain, not an exclusion of the other row ball.')
    parser.add_argument('--mode',choices=('full-row','coarse-row'),default=default_mode)
    parser.add_argument('--support-bounds',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--two-adic',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--norm-unsat-cache',type=int,default=4096 if default_mode=='coarse-row' else 0,help='Bounded, in-process SINGLE C_B norm exhaustion proofs (coarse mode only).')
    parser.add_argument('--norm-carrier-cache',type=int,default=0,help='Bounded support slots for C_B carriers and every unresolved C_B support; G is always freshly solved.')
    parser.add_argument('--unary-batch-free',type=int,default=-1,help='Complete independently audited C_B sign fibers in 256-support batches through this free dimension (0..12); -1 rolls back.')
    parser.add_argument('--dihedral-support',action=argparse.BooleanOptionalAction,default=default_mode=='coarse-row')
    parser.add_argument('--cut-projection',action=argparse.BooleanOptionalAction,default=False)
    parser.add_argument('--resume-closed',type=Path,help='Certificate directory from a terminal, independently audited coarse run at the same or smaller radius; only fully closed literal anchors credited.')
    parser.add_argument('--certified-psd',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--all-target-psd',action=argparse.BooleanOptionalAction,default=False,
                        help='Share one cached exact C_A spectral envelope across all target locations (coarse mode)')
    parser.add_argument('--prebatch-dihedral',action=argparse.BooleanOptionalAction,default=default_mode=='coarse-row',
                        help='Reuse complete single C_B dihedral exclusions before native batch/fallback work; never G-only exclusions')
    parser.add_argument('--unary-affine-audit',action=argparse.BooleanOptionalAction,default=False,
                        help='Certify equality of independently constructed affine spaces before reusing complete unary exhaustion')
    parser.add_argument('--unary-linear-certificate',action=argparse.BooleanOptionalAction,default=None,
                        help='Verify necessary equations, affine base, independent spanning directions and exact dimension before transferring complete literal enumeration')
    parser.add_argument('--native-support-audit',action=argparse.BooleanOptionalAction,default=default_mode=='coarse-row',
                        help='Independently audit every retained support using full GF2 polynomial convolution in native code')
    parser.add_argument('--prime-norm',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--stop-on-new',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--support-ledger',action=argparse.BooleanOptionalAction,default=True,
                        help='Write every signed support fiber; per-target and per-anchor ledgers are always written')
    parser.add_argument('--output',type=Path,default=ROOT/'results'/('L86_full_row_completion_'+datetime.now().strftime('%Y%m%d_%H%M%S')))
    args=parser.parse_args()
    if args.unary_linear_certificate is None:
        args.unary_linear_certificate=default_mode=='coarse-row' and not args.unary_affine_audit
    if args.unary_linear_certificate and args.unary_affine_audit:
        parser.error('choose the linear certificate or dense affine-space audit, not both')
    if args.coarse_input is not None and args.mode!='coarse-row':parser.error('direct coarse input requires coarse-row mode')
    if args.coarse_anchors and (args.mode!='coarse-row' or args.radius!=0 or args.anchor_side==1):
        parser.error('literal coarse anchors require coarse-row mode, radius zero and side 0 or both')
    if args.top<0 or args.radius not in (0,1,2) or not 0<=args.norm_unsat_cache<=65536 or not 0<=args.norm_carrier_cache<=65536 or not -1<=args.unary_batch_free<=12:
        parser.error('nonnegative top, whole-anchor radius 0..2 and norm proof cache 0..65536 required')
    cache=CompleteUnaryNormUnsatCache(args.norm_unsat_cache) if args.norm_unsat_cache and args.mode=='coarse-row' else None
    carrier_cache=UnaryNormCarrierCache(args.norm_carrier_cache) if args.norm_carrier_cache and args.mode=='coarse-row' else None
    if args.input is None and args.log is None and args.coarse_input is None and args.coarse_anchors is None:
        args.log=max((ROOT/'logs').glob('torch_L86_seed*.jsonl'),key=lambda p:p.stat().st_mtime)
    inputs=(audited_coarse_points(args.coarse_anchors,args.top) if args.coarse_anchors else
            audited_coarse_centers(args.coarse_input,args.top) if args.coarse_input else
            audited_log_centers(args.log,args.top if args.top else sys.maxsize) if args.log else
            select_json_centers(json.loads(args.input.read_text()),args.top))
    prefix=[];prefix_certificate=None
    if args.resume_closed:
        from experiments.prepare_coarse_resume import file_sha,words_sha
        prefix_certificate=json.loads((args.resume_closed/'certificate.json').read_text())
        if (args.mode!='coarse-row' or bool(args.coarse_input)!=bool(prefix_certificate.get('direct_coarse_input',False))
                or prefix_certificate['mode']!=args.mode or prefix_certificate['radius']>args.radius
                or prefix_certificate.get('anchor_side')!=args.anchor_side
                or prefix_certificate['input_words_sha256']!=words_sha(inputs)):
            raise ValueError('closed prefix belongs to a different literal input/mode/radius')
        if file_sha(args.resume_closed/'closed_anchors.json')!=prefix_certificate['closed_anchors_sha256']:
            raise ValueError('closed anchor certificate changed after audit')
        parent=Path(prefix_certificate['parent'])
        for name,digest in prefix_certificate['parent_sha256'].items():
            if file_sha(parent/name)!=digest:raise ValueError('original proof ledger/source reference changed after prefix audit')
        prefix=json.loads((args.resume_closed/'closed_anchors.json').read_text())
        if len(prefix)!=prefix_certificate['closed_anchors']:raise ValueError('closed prefix certificate count mismatch')
    closed_by_key={tuple(r['anchor_key']):r for r in prefix};unseen_closed=set(closed_by_key)
    if len(closed_by_key)!=len(prefix):raise ValueError('duplicate literal anchor in closed proof bank')
    args.output.mkdir(parents=True,exist_ok=False);tick=perf_counter();snapshot=args.output/'source_snapshot'
    freeze_sources(snapshot);(snapshot/'experiments').mkdir(exist_ok=True)
    for path in (ROOT/'experiments').glob('*.py'):shutil.copy2(path,snapshot/'experiments'/path.name)
    sources={str(p.relative_to(snapshot)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(snapshot.rglob('*'))
             if p.is_file() and p.name!='source_manifest.json'}
    atomic_write_json(snapshot/'source_manifest.json',sources);atomic_write_json(args.output/'input_centers.json',inputs)
    if args.coarse_input:shutil.copy2(args.coarse_input,args.output/'input_coarse_source.json')
    if args.coarse_anchors:shutil.copy2(args.coarse_anchors,args.output/'input_coarse_points.json')
    if prefix_certificate:
        atomic_write_json(args.output/'closed_prefix_certificate.json',prefix_certificate)
        atomic_write_json(args.output/'closed_prefix_anchors.json',prefix)
    raw=(ROOT/'86.txt').read_bytes() if (ROOT/'86.txt').exists() else b''
    (args.output/'initial_inventory.txt').write_bytes(raw);known=verified_inventory(raw)
    seen=set();new=set();rows=deque(maxlen=16);anchor_count=literal_anchors=unknown=0;stopped=False
    norm_orbits=sign_fibers=assignments=target_count=sat_targets=coarse_supports=0
    if prefix_certificate:
        stats=prefix_certificate['stats'];norm_orbits=stats['support_orbits'];sign_fibers=stats['sign_fibers']
        assignments=stats['assignments'];target_count=stats['target_fibers_decided'];coarse_supports=stats['coarse_witness_supports']
    def save(exhausted):
        report=dict(complete=exhausted and not unknown and not stopped and not sat_targets,
                    all_planned_anchors_exhausted=exhausted,stopped_on_new=stopped,radius=args.radius,
                    parents=len(inputs['seeds']),anchor_side=args.anchor_side,literal_anchors=literal_anchors,distinct_anchors=anchor_count+len(unseen_closed),
                    generated_distinct_anchors=anchor_count,
                    distinct_anchor_orbits=anchor_count if args.mode=='full-row' else None,mode=args.mode,
                    target_fibers_decided=target_count,unknown_target_fibers=unknown,recent_completed_anchors=list(rows),sat_target_fibers=sat_targets,
                    completed_anchor_ledger='completed_anchors.jsonl.gz',coarse_witness_supports=coarse_supports,
                    support_orbits=norm_orbits,sign_fibers=sign_fibers,assignments=assignments,
                    certified_psd=args.certified_psd,prime_norm=args.prime_norm,
                    all_target_psd=args.all_target_psd,
                    prebatch_dihedral=args.prebatch_dihedral,
                    unary_affine_audit=args.unary_affine_audit,
                    unary_linear_certificate=args.unary_linear_certificate,
                    native_support_audit=args.native_support_audit,
                    support_ledger=args.support_ledger,
                    support_bounds=args.support_bounds,
                    two_adic=args.two_adic,
                    norm_unsat_cache_capacity=args.norm_unsat_cache,norm_unsat_cache_hits=cache.hits if cache else 0,
                    norm_carrier_cache_slot_budget=args.norm_carrier_cache,norm_carrier_cache_hits=carrier_cache.hits if carrier_cache else 0,
                    norm_carrier_cache_live_slots=carrier_cache.slots if carrier_cache else 0,
                    unary_batch_free=args.unary_batch_free,
                    dihedral_support=args.dihedral_support,
                    cut_projection=args.cut_projection,
                    direct_coarse_input=bool(args.coarse_input),
                    literal_coarse_points_input=bool(args.coarse_anchors),
                    credited_closed_anchors=len(prefix),closed_prefix_parent=prefix_certificate['parent'] if prefix_certificate else None,
                    closed_proof_kind=('same_radius_closed_prefix' if prefix_certificate['radius']==args.radius else 'smaller_radius_closed_anchor_bank') if prefix_certificate else None,
                    independently_verified_new_classes=len(new),elapsed=perf_counter()-tick,
                    source_manifest_sha256=hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest(),
                    scope=('Anchors from audited non-solution pairs and their complete radius-r whole-row balls; '
                           'other full binary row has all 86 bits free. Anchor quotient only exact row dihedral/complement/common-unit mappings. '
                           if args.mode=='full-row' else ('Explicit literal signed C_A fixed points; no compressed C_B witness assumed. '
                           if args.coarse_anchors else 'Signed C_A anchors from independently audited actual compressed norm witnesses and their '
                           'complete radius-r L1 balls. ' if args.coarse_input else 'Signed C_A anchors from audited non-solution pairs and all radius-r binary-row edits, '
                           'equivalently the complete radius-r L1 ball in C_A. '
                           )+'All G_A signs and all binary B bits free. '
                           'Only identical literal C_A anchors deduplicated, with all models remaining available. ')+
                          'All k=1..42 and eta=+/-1. '
                          +('Both source-row anchor balls. ' if args.anchor_side is None else 'Only source row '+str(args.anchor_side)+' anchor balls; no exclusion of the other source-row ball. ')+
                          'SAT retains first verified model, not every full equivalence class; capped sign branches remain UNKNOWN.',
                    claim='No >=20% new-discovery claim unless actual fully new L86 classes and appropriate complete comparison establish it.')
        atomic_write_json(args.output/'report.json',report)
        print(json.dumps({k:report[k] for k in ('complete','distinct_anchors','target_fibers_decided','independently_verified_new_classes','elapsed')}),flush=True)
    with (gzip.open(args.output/'signed_support_fibers.jsonl.gz','wt',compresslevel=1) if args.support_ledger else nullcontext()) as ledger, \
         gzip.open(args.output/'full_row_target_fibers.jsonl.gz','wt',compresslevel=1) as target_ledger, \
         gzip.open(args.output/'completed_anchors.jsonl.gz','wt',compresslevel=1) as anchor_ledger:
        def flush():
            if ledger is not None:ledger.flush()
            target_ledger.flush();anchor_ledger.flush()
        last_checkpoint=perf_counter()
        for parent,item in enumerate(inputs['seeds']):
            for side,flips,a,coarse,key in anchor_candidates(item,args.mode,args.radius,args.anchor_side):
                literal_anchors+=1
                a=tuple(a) if a is not None else None
                if key in seen:continue
                seen.add(key);anchor_count+=1
                if key in closed_by_key:
                    previous=closed_by_key[key];unseen_closed.remove(key)
                    row=dict(previous,anchor=anchor_count-1,parent=parent,side=side,flips=flips,
                             closed_proof_original_anchor=previous['anchor'],closed_prefix_parent=prefix_certificate['parent'])
                    rows.append(row);anchor_ledger.write(json.dumps(row)+'\n');continue
                anchor_tick=perf_counter();statuses=Counter();decided=0
                def anchor_row():return dict(anchor=anchor_count-1,parent=parent,side=side,flips=flips,
                    anchor_key=key,target_fibers_decided=decided,statuses=dict(statuses),elapsed=perf_counter()-anchor_tick)
                for k in range(1,43):
                    for eta in (-1,1):
                        label=dict(anchor=anchor_count-1,parent=parent,side=side,flips=flips,k=k,eta=eta)
                        def record(out,label=label):ledger.write(json.dumps(dict(label,**out))+'\n')
                        solver=complete_fixed_binary_row_target if args.mode=='full-row' else complete_fixed_coarse_row_target
                        out=solver(a if args.mode=='full-row' else coarse,k=k,eta=eta,certified_psd=args.certified_psd,
                                    prime_norm=args.prime_norm,on_support=record if ledger is not None else None,support_bounds=args.support_bounds,two_adic=args.two_adic,cut_projection=args.cut_projection,
                                    **(dict(norm_unsat_cache=cache,norm_carrier_cache=carrier_cache,dihedral_support=args.dihedral_support,unary_batch_free=args.unary_batch_free,all_target_psd=args.all_target_psd,prebatch_dihedral=args.prebatch_dihedral,unary_affine_audit=args.unary_affine_audit,unary_linear_certificate=args.unary_linear_certificate,native_support_audit=args.native_support_audit) if args.mode=='coarse-row' else {}))
                        norm_orbits+=out['support_orbits'];sign_fibers+=out['sign_fibers'];assignments+=out['assignments']
                        unknown+=out['status'].startswith('UNKNOWN');target_count+=1;decided+=1
                        statuses[out['status']]+=1;sat_targets+=out['status']=='SAT_FULL_PQCP'
                        coarse_supports+=out.get('coarse_witness_supports',0)
                        target_ledger.write(json.dumps(dict(label,**out))+'\n')
                        if out['words']:
                            hit=discovery_key(*out['words'])
                            if hit not in known:
                                new.add(hit);append_verified_solution_if_new(86,*out['words'],args.output)
                                if args.stop_on_new:
                                    stopped=True;row=anchor_row();rows.append(row);anchor_ledger.write(json.dumps(row)+'\n')
                                    flush();save(False);return
                row=anchor_row();rows.append(row);anchor_ledger.write(json.dumps(row)+'\n')
                if anchor_count%256==0 or perf_counter()-last_checkpoint>=30:
                    flush();save(False);last_checkpoint=perf_counter()
    if unseen_closed:raise ValueError('input region did not regenerate every already certified literal anchor')
    save(True)


if __name__=='__main__':main()
