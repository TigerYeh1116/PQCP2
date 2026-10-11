"""Replay real unary fibres against archived independently enumerated outcomes.

No old sign/proof cache is imported. Orbit proofs are built solely by the
fresh replay. SAT carriers still receive independent integer PACF audits.
"""
import argparse
import base64
import gzip
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.benchmark_l86_discovery import freeze_sources
from experiments.extract_coarse_successor_anchors import file_sha
from solver.checkpoint import atomic_write_json
from solver.unary_norm_batch import complete_unary_norm_batch,CompleteUnarySupportOrbitUnsatCache
from solver.cyclic_support_native import audit_literal_support_norms


def replay(path,limit,linear_certificate=False,norm_audit_only=False):
    context=None;cache=None;supports=batches=sat=deferred=orbit_hits=old_audits=new_audits=0
    with gzip.open(path,'rt') as stream:
        for line in stream:
            entry=json.loads(line)
            if entry.get('kind')!='C_B_BATCH':continue
            current=(entry['anchor'],entry['k'],entry['eta'])
            if current!=context:context=current;cache=CompleteUnarySupportOrbitUnsatCache()
            original=tuple(struct.iter_unpack('=QQQQQ',base64.b64decode(entry['certificate_base64'])))
            masks=tuple(r[0] for r in original)
            desired=sum((v%2)<<u for u,v in enumerate(entry['wanted']))
            if not audit_literal_support_norms(masks,len(entry['wanted']),desired,entry['wanted'][0]):
                raise RuntimeError('native full polynomial norm/weight audit differs from original literal oracle')
            if norm_audit_only:
                supports+=len(masks);batches+=1
                if supports>=limit:break
                continue
            result=complete_unary_norm_batch(masks,len(entry['wanted']),entry['wanted'],entry['negative'],
                2**31,free_limit=entry['free_limit'],dihedral_unsat_cache=cache,affine_audit=not linear_certificate,linear_certificate=linear_certificate)
            actual={r[0]:(r[1]&255,r[2]) for r in struct.iter_unpack('=QQQQQ',base64.b64decode(result['certificate_base64']))}
            for _,mask,source,flags,seen,audit in struct.iter_unpack('=QQQQQQ',base64.b64decode(result['dihedral_unsat_receipts_base64'])):
                if flags&255:raise RuntimeError('orbit source was not a complete single-norm exclusion')
                actual[mask]=(0,0)
            if len(actual)!=len(original):raise RuntimeError('replay did not partition every original literal support')
            for mask,flags,word,seen,audit in original:
                if actual[mask]!=(flags&255,word):raise RuntimeError('literal support classification/witness differs from archived original auditor')
                old_audits+=audit
            supports+=len(masks);batches+=1;sat+=result['sat_supports'];deferred+=result['deferred_supports']
            orbit_hits+=result['dihedral_unsat_supports'];new_audits+=result['auditor_candidates_inspected']
            if supports>=limit:break
    if not supports:raise RuntimeError('no actual native batch records found')
    return dict(source=str(path),source_sha256=file_sha(path),literal_supports=supports,batches=batches,
        identical_classification_and_first_witness=not norm_audit_only,native_full_norm_audits=supports,
        sat_carriers=sat,deferred_supports=deferred,
        new_complete_orbit_exclusions=orbit_hits,original_auditor_assignments=old_audits,
        new_auditor_assignments=new_audits)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ledger',type=Path,action='append',required=True)
    p.add_argument('--supports-per-ledger',type=int,default=250000)
    p.add_argument('--linear-certificate',action='store_true')
    p.add_argument('--norm-audit-only',action='store_true',help='Audit full polynomial norm/weight only; no sign classification claim')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--_frozen-runner',action='store_true',help=argparse.SUPPRESS)
    args=p.parse_args();output=args.output.resolve();ledgers=tuple(x.resolve() for x in args.ledger)
    if args.supports_per_ledger<1:p.error('positive support count required')
    snapshot=output/'source_snapshot'
    if not args._frozen_runner:
        output.mkdir(parents=True,exist_ok=False);freeze_sources(snapshot)
        for path in (ROOT/'experiments').glob('*.py'):
            dest=snapshot/'experiments'/path.name;dest.parent.mkdir(exist_ok=True);shutil.copy2(path,dest)
        manifest={str(x.relative_to(snapshot)):file_sha(x) for x in sorted(snapshot.rglob('*'))
                  if x.is_file() and x.name!='source_manifest.json'}
        atomic_write_json(snapshot/'source_manifest.json',manifest)
        command=[sys.executable,str(snapshot/'experiments'/Path(__file__).name),'--_frozen-runner',
                 '--supports-per-ledger',str(args.supports_per_ledger),'--output',str(output)]
        for ledger in ledgers:command.extend(('--ledger',str(ledger)))
        if args.linear_certificate:command.append('--linear-certificate')
        if args.norm_audit_only:command.append('--norm-audit-only')
        atomic_write_json(output/'execution.json',dict(actual_frozen_execution=True,command=command,
            source_ledger_sha256={str(x):file_sha(x) for x in ledgers}))
        raise SystemExit(subprocess.run(command,cwd=snapshot).returncode)
    if ROOT!=snapshot:raise RuntimeError('actual archived execution required')
    manifest=json.loads((snapshot/'source_manifest.json').read_text())
    if any(file_sha(snapshot/name)!=digest for name,digest in manifest.items()):raise RuntimeError('archived source changed')
    before={str(x):file_sha(x) for x in ledgers};rows=[replay(x,args.supports_per_ledger,args.linear_certificate,args.norm_audit_only) for x in ledgers]
    if before!={str(x):file_sha(x) for x in ledgers}:raise RuntimeError('original proof ledger changed during replay')
    result=dict(complete=True,actual_frozen_execution=True,linear_certificate=args.linear_certificate,norm_audit_only=args.norm_audit_only,rows=rows,
        claim=('Every archived literal support received an independent native full GF2 norm/weight audit; sign classification not replayed.'
               if args.norm_audit_only else 'Exact classifications and SAT carriers match actual archived unary fibres; no new full L86 or speed claim.'))
    atomic_write_json(output/'report.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
