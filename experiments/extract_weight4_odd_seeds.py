"""Extract ALL PSD/norm-admissible odd-integer survivors of a complete g bank.

Audit the entire original input/fiber ledger, then replay every nonempty odd
fiber with the new complete seed buffer. Original full-search stats must be
identical. These non-solutions are seeds only, never full PQCP discoveries.
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

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.benchmark_l86_discovery import freeze_sources
from solver.checkpoint import atomic_write_json
from solver.same_o_weight4_native import complete_weight4_c_fiber,complete_weight2_c_fiber
from solver.same_o_weight4 import required_o_intersections,odd_integer_match
from solver.sds_norm_completion import intersection_profile
from solver.perfect_odd_kernel import interleave_from_complementary_trits
from solver.solution_equivalence import pqcp_equivalence_key
from solver.structured_energy import full_error_energy,target_profile
from solver.verifier import verify_pqcp


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent',type=Path,default=ROOT/'results/L86_same_O_weight4_complete_20261007')
    parser.add_argument('--output',type=Path,default=ROOT/'results'/('L86_weight4_odd_seeds_'+datetime.now().strftime('%Y%m%d_%H%M%S')))
    args=parser.parse_args();parent=json.loads((args.parent/'report.json').read_text())
    if not parent['complete'] or not parent['entire_normalized_bank']:raise ValueError('complete entire-bank parent required')
    if parent['totals'].get('full_hits',0):raise ValueError('this non-solution seed extractor requires a no-full-hit parent')
    inputs=json.loads((args.parent/'input_g_rows.json').read_text())['rows'];records=[];totals=Counter();selected=[]
    for index,line in enumerate((args.parent/'completion_fibers.jsonl').open()):
        record=json.loads(line);certificate=record['certificate'];out=record['result']
        if (record['index']!=index or index>=len(inputs) or [certificate['plus'],certificate['minus']]!=inputs[index]
                or not out['complete']):raise RuntimeError('parent ledger is missing, duplicated, misordered or incomplete')
        records.append(record);totals.update(out['stats'])
        if out['stats'].get('even_integer_excluded',0)+out['stats'].get('full_hits',0)>0:selected.append(record)
    if len(records)!=len(inputs) or len(records)!=parent['expected_fibers']:raise RuntimeError('parent input coverage incomplete')
    for name,value in totals.items():
        if parent['totals'][name]!=value:raise RuntimeError('parent aggregate counters differ from closed ledger')
    args.output.mkdir(parents=True,exist_ok=False);tick=perf_counter();snapshot=args.output/'source_snapshot';freeze_sources(snapshot)
    (snapshot/'experiments').mkdir(exist_ok=True)
    for path in (ROOT/'experiments').glob('*.py'):shutil.copy2(path,snapshot/'experiments'/path.name)
    manifest={str(p.relative_to(snapshot)):hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(snapshot.rglob('*')) if p.is_file() and p.name!='source_manifest.json'}
    atomic_write_json(snapshot/'source_manifest.json',manifest)
    for name in ('report.json','input_g_rows.json','completion_fibers.jsonl'):
        shutil.copy2(args.parent/name,args.output/('parent_'+name))
    seeds={};literal_models=0;replayed=0;histogram=Counter()
    for record in selected:
        c=record['certificate'];plus,minus=c['plus'],c['minus']
        complete=complete_weight2_c_fiber if c.get('g_weight',4)==2 else complete_weight4_c_fiber
        out=complete(c,seed_capacity=1)
        for name,value in record['result']['stats'].items():
            if out['stats'][name]!=value:raise RuntimeError('replay changed complete parent fiber counters')
        expected=record['result']['stats'].get('even_integer_excluded',0)+record['result']['stats'].get('full_hits',0)
        if len(out['odd_integer_models'])!=expected:raise RuntimeError('seed replay omitted an odd-integer preimage')
        for negative,o_mask,deviation in out['odd_integer_models']:
            wanted=required_o_intersections(negative,plus,minus);actual=intersection_profile(o_mask,43)
            if (sum((actual[u]-wanted[u])**2 for u in range(1,22))!=deviation
                    or not odd_integer_match(o_mask,plus,minus)):raise RuntimeError('native seed deviation/odd equations failed Python audit')
            c_row=tuple(0 if (plus|minus)>>i&1 else 1-2*((negative>>i)&1) for i in range(43))
            g=tuple(int(plus>>i&1)-int(minus>>i&1) for i in range(43));o=tuple(1-2*((o_mask>>i)&1) for i in range(43))
            words=interleave_from_complementary_trits(c_row,g,o);check=verify_pqcp(*words)
            target=target_profile(86,41,1)
            if any(check.profile[u]!=target[u] for u in range(1,86,2)):raise RuntimeError('odd seed failed independent original full odd lags')
            raw_energy=full_error_energy(check.profile,41,1)
            if raw_energy!=64*deviation:raise RuntimeError('seed full profile energy differs from native even-deviation score')
            literal_models+=1;energy=raw_energy//16;histogram[energy]+=1
            key=pqcp_equivalence_key(*words)
            if key not in seeds:
                seeds[key]=dict(words=words,k=41,eta=1,target_energy=energy,is_full_solution=check.is_valid,
                                parent_index=record['index'],plus=plus,minus=minus,c_negative=negative,o_mask=o_mask)
        replayed+=1
    expected_models=parent['totals'].get('even_integer_excluded',0)+parent['totals'].get('full_hits',0)
    if literal_models!=expected_models:raise RuntimeError('complete odd-model extraction count differs from closed parent bank')
    report=dict(complete=True,parent_complete=True,parent_fibers=len(records),replayed_fibers=replayed,
                literal_odd_models=literal_models,seeds=sorted(seeds.values(),key=lambda s:s['target_energy']),
                seed_classes=len(seeds),literal_energy_histogram=dict(histogram),
                minimum_seed_energy=min(histogram) if histogram else None,
                independently_verified_new_classes=0,elapsed=perf_counter()-tick,
                parent_ledger_sha256=hashlib.sha256((args.parent/'completion_fibers.jsonl').read_bytes()).hexdigest(),
                source_manifest_sha256=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest(),
                scope='ALL odd-integer survivors that passed parent C-profile PSD and anchored binary norm/trace. '
                      'Complete parent ledger audited; all selected full C fibers replayed, stats unchanged. '
                      'Seeds are non-solutions, no full-discovery-speed claim.')
    atomic_write_json(args.output/'report.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='seeds'}),flush=True)


if __name__=='__main__':main()
