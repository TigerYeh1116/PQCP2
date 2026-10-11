"""Audit terminal radius-four children and reconcile a failed coordinator.

Never launches/restarts a search. A failed manager report stays archived.
Every credited child must independently close its exact center and region,
match immutable source hashes and independently verified solution inventory.
"""
import argparse
import hashlib
import json
from math import comb
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.benchmark_l86_discovery import verified_inventory
from solver.checkpoint import atomic_write_json
from solver.three_half_row_completion import three_half_triple_count,four_half_triple_count


def audit_manifest(folder,expected_digest):
    manifest=json.loads((folder/'source_manifest.json').read_text())
    digest=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    if digest!=expected_digest:raise RuntimeError('report source manifest digest changed')
    for name,value in manifest.items():
        path=(folder/name).resolve()
        if folder.resolve() not in path.parents or hashlib.sha256(path.read_bytes()).hexdigest()!=value:
            raise RuntimeError('frozen source file changed: '+name)
    return manifest


def audit_child(initial,inputs,index,folder):
    out=json.loads((folder/'result/report.json').read_text());d=43;varied=initial['varied_halves']
    expected=(12*sum(comb(d,r) for r in range(5)) if varied==1 else
              three_half_triple_count(d,4) if varied==2 else four_half_triple_count(d,4))
    if (not out['complete'] or out['completed_centers']!=1 or out['centers']!=1 or out['radius']!=4
            or out['varied_halves']!=varied or out['generated_triples']!=expected
            or sum(out[k] for k in ('content_excluded','identical_triples','eligible_jobs'))!=expected
            or out['actual_fibers_solved']+out['cache_hits']!=out['eligible_jobs']):
        raise RuntimeError('child did not close its entire declared region')
    literal=tuple(map(tuple,inputs['seeds'][index]['words']))
    for name in ('input.json','result/input_centers.json'):
        child=json.loads((folder/name).read_text())
        if len(child['seeds'])!=1 or tuple(map(tuple,child['seeds'][0]['words']))!=literal:
            raise RuntimeError('child literal center changed')
    child_sources=audit_manifest(folder/'result/source_snapshot',out['source_manifest_sha256'])
    coordinator_sources=json.loads((folder.parent.parent/'source_snapshot/source_manifest.json').read_text())
    # The coordinator's empty/known 86.txt is runtime input, while a child
    # stores it outside source_snapshot as initial_inventory.txt. All actual
    # source files and copied mathematical proofs must match exactly.
    coordinator_sources.pop('86.txt',None)
    if child_sources!=coordinator_sources:raise RuntimeError('child used different source content from the pinned coordinator runtime')
    if (folder/'result/initial_inventory.txt').read_bytes()!=(folder.parent.parent/'initial_inventory.txt').read_bytes():
        raise RuntimeError('child initial solution inventory changed')
    solution=folder/'result/86.txt';keys=verified_inventory(solution.read_bytes() if solution.exists() else b'')
    known=verified_inventory((folder.parent.parent/'initial_inventory.txt').read_bytes())
    if len(keys-known)!=out['independently_verified_new_classes']:
        raise RuntimeError('child novel-class count differs from independent full-verifier inventory')
    return out,keys-known


def reconcile(root):
    root=Path(root).resolve();initial=json.loads((root/'report.json').read_text())
    if initial['closed_parent_prefix']!=0:raise ValueError('this recovery audits fresh jobs only; do not merge an unaudited parent prefix')
    inputs=json.loads((root/'input_centers.json').read_text())
    audit_manifest(root/'source_snapshot',initial['source_manifest_sha256'])
    children={};new=set()
    for index in range(len(inputs['seeds'])):
        out,keys=audit_child(initial,inputs,index,root/'jobs'/('center_'+str(index)))
        children[str(index)]=out;new.update(keys)
    original=root/'manager_report_before_reconciliation.json'
    if original.exists():raise ValueError('existing reconciliation archive; refusing to overwrite evidence')
    shutil.copy2(root/'report.json',original)
    shutil.copy2(Path(__file__),root/'reconciliation_source.py')
    out=dict(initial,complete=True,credited_center_indices=list(range(len(inputs['seeds']))),
             completed_parallel_centers=len(children),children=children,
             independently_verified_new_classes=len(new),
             reconciliation=dict(reason='Coordinator compared JSON list against in-memory tuple; all search children already terminal.',
                 original_coordinator_complete=False,all_literal_inputs_and_source_hashes_audited=True,
                 all_declared_child_counts_audited=True,all_solution_inventories_independently_full_verified=True,
                 source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 timing='Original coordinator elapsed was incomplete; do not use it as completed-run timing.'),
             elapsed=None,
             generated_triples=sum(c['generated_triples'] for c in children.values()),
             eligible_jobs=sum(c['eligible_jobs'] for c in children.values()),
             norm_visited=sum(c['norm_visited'] for c in children.values()))
    atomic_write_json(root/'report.json',out)
    return out


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--experiment',type=Path,required=True)
    result=reconcile(parser.parse_args().experiment)
    print(json.dumps({k:result[k] for k in ('complete','generated_triples','eligible_jobs','independently_verified_new_classes')}))
