"""Copy complete JSON records from an interrupted, single-member gzip log.

Original files remain unchanged. Recovery alone is NOT an UNSAT proof;
prepare_coarse_resume must independently audit all 84 targets, input order
and frozen source hashes before any closed anchor can be credited.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zlib

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.prepare_coarse_resume import file_sha
from solver.checkpoint import atomic_write_json


def recover_records(source,destination):
    """Accept complete JSON lines; reject malformed deflate, CRC or JSON."""
    source,destination=Path(source),Path(destination)
    original_sha=file_sha(source);decoder=zlib.decompressobj(16+zlib.MAX_WBITS)
    pending=b'';records=0;plaintext=hashlib.sha256()
    with source.open('rb') as stream,gzip.open(destination,'wb',compresslevel=1) as output:
        for chunk in iter(lambda:stream.read(128*1024),b''):
            decoded=decoder.decompress(chunk)
            if decoder.unused_data:
                raise ValueError('multiple gzip members or trailing data require a separate audit')
            pieces=(pending+decoded).split(b'\n');pending=pieces.pop()
            for line in pieces:
                if not line:raise ValueError('empty ledger record')
                json.loads(line)
                output.write(line+b'\n');plaintext.update(line+b'\n');records+=1
        if decoder.eof and pending:
            raise ValueError('complete stream contains an unterminated ledger record')
    if file_sha(source)!=original_sha:raise ValueError('original ledger changed during recovery')
    return dict(source=str(source.resolve()),original_sha256=original_sha,
        original_gzip_footer_verified=decoder.eof,complete_json_records=records,
        discarded_unterminated_bytes=len(pending),complete_record_bytes_sha256=plaintext.hexdigest(),
        recovered_sha256=file_sha(destination))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();parent=args.parent.resolve();output=args.output.resolve()
    report=json.loads((parent/'report.json').read_text())
    if report['mode']!='coarse-row':raise ValueError('coarse ledgers required')
    output.mkdir(parents=True,exist_ok=False)
    copied={}
    for name in ('report.json','input_centers.json'):
        before=file_sha(parent/name);shutil.copy2(parent/name,output/name)
        if file_sha(parent/name)!=before or file_sha(output/name)!=before:
            raise ValueError('input/report changed during recovery')
        copied[name]=before
    shutil.copytree(parent/'source_snapshot',output/'source_snapshot')
    for name in ('input_coarse_source.json','initial_inventory.txt','closed_prefix_certificate.json','closed_prefix_anchors.json'):
        if (parent/name).exists():shutil.copy2(parent/name,output/name)
    ledgers={name:recover_records(parent/name,output/name) for name in
             ('completed_anchors.jsonl.gz','full_row_target_fibers.jsonl.gz','signed_support_fibers.jsonl.gz')}
    atomic_write_json(output/'recovery_receipt.json',dict(original_parent=str(parent),copied_input_sha256=copied,
        ledgers=ledgers,proof_status='NOT_A_CERTIFICATE',
        scope='Complete serialized records only. Missing gzip footers remain disclosed. Independent closed-prefix audit required; no partial last anchor, new solution, or sign-model cache credited.'))
    print(json.dumps({name:row['complete_json_records'] for name,row in ledgers.items()}),flush=True)


if __name__=='__main__':main()
