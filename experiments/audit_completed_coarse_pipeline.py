"""Read-only independent audit of terminal paired coarse child ledgers."""
import argparse
import gzip
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.prepare_coarse_resume import closed_target_stats,file_sha
from experiments.benchmark_l86_discovery import verified_inventory
from solver.checkpoint import atomic_write_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args();rows=[]
    for path in sorted(args.input.glob('*/result/report.json')):
        report=json.loads(path.read_text());folder=path.parent
        if not report['complete']:continue
        with gzip.open(folder/'completed_anchors.jsonl.gz','rt') as stream:anchors=[json.loads(line) for line in stream]
        if len(anchors)!=report['distinct_anchors'] or any(r['anchor']!=j or r['target_fibers_decided']!=84 or
                r['statuses']!={'UNSAT_FIXED_SIGNED_C_A':84} for j,r in enumerate(anchors)):
            raise ValueError('completed child has partial/SAT/UNKNOWN anchor evidence')
        with gzip.open(folder/'full_row_target_fibers.jsonl.gz','rt') as stream:
            stats=closed_target_stats(anchors,(json.loads(line) for line in stream))
        if any(stats[k]!=report[k] for k in stats):raise ValueError('target ledger totals disagree with report')
        snapshot=folder/'source_snapshot';manifest=json.loads((snapshot/'source_manifest.json').read_text())
        for name,digest in manifest.items():
            if file_sha(snapshot/name)!=digest:raise ValueError('source snapshot changed')
        classes=verified_inventory((folder/'86.txt').read_bytes() if (folder/'86.txt').exists() else b'')
        if classes or report['independently_verified_new_classes']:raise ValueError('exhaustion should not contain full new models')
        rows.append(dict(child=folder.parent.name,anchors=len(anchors),all_target_stats=stats,
                         report_sha256=file_sha(path),source_manifest_sha256=file_sha(snapshot/'source_manifest.json')))
    atomic_write_json(args.output,dict(completed_children_audited=rows,
        scope='Only already terminal complete children; every anchor and target status/count and frozen source checked. Live children not credited.'))
    print(json.dumps(dict(completed_children=len(rows),children=[r['child'] for r in rows])),flush=True)


if __name__=='__main__':main()
