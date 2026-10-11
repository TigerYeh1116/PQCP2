"""Complete frozen-anchor audit of norm/support exhaustion-proof equivalence.

Counts duplicate proof domains, never duplicate/new full PQCP classes.
No SAT or UNKNOWN result may close an equivalence class.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.complete_l86_full_rows import anchor_candidates,audited_coarse_centers
from experiments.benchmark_l86_discovery import freeze_sources
from solver.checkpoint import atomic_write_json
from solver.structured_energy import integer_periodic_autocorrelation_profile as pacf
from solver.cyclic_support_native import dihedral_support_key


def norm_support_key(c,*,common_units):
    d=len(c);norm=tuple(pacf(c));mask=sum((v!=0)<<j for j,v in enumerate(c))
    if not common_units:return norm,dihedral_support_key(mask,d)
    # q and -q give the same norm and dihedral support key. Thus all common
    # units are represented by q=1..(d-1)/2 at these supported prime lengths.
    transformed=[tuple(norm[q*u%d] for u in range(d)) for q in range(1,d//2+1)]
    minimum=min(transformed)
    masks=[sum(((mask>>(q*j%d))&1)<<j for j in range(d))
           for q,values in enumerate(transformed,1) if values==minimum]
    return minimum,min(dihedral_support_key(m,d) for m in masks)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=ROOT/'results/L86_fixed_coarse_rows_radius2_top8_20261008/input_centers.json')
    p.add_argument('--coarse-input',type=Path,default=ROOT/'results/L86_independent_norm_long_20261007/coarse_witnesses.json')
    p.add_argument('--output',type=Path,default=ROOT/'results'/('L86_coarse_anchor_equivalence_'+datetime.now().strftime('%Y%m%d_%H%M%S')))
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    inputs=(('root',json.loads(args.input.read_text())),('coarse',audited_coarse_centers(args.coarse_input,0)))
    freeze_sources(args.output/'source_snapshot');rows=[]
    for label,data in inputs:
        atomic_write_json(args.output/(label+'_inputs.json'),data)
        for radius in (1,2):
            seen=set();without_units=set();with_units=set()
            for item in data['seeds']:
                for _,_,_,c,key in anchor_candidates(item,'coarse-row',radius):
                    if key in seen:continue
                    seen.add(key);without_units.add(norm_support_key(c,common_units=False));with_units.add(norm_support_key(c,common_units=True))
            row=dict(source=label,radius=radius,literal_distinct_anchors=len(seen),
                norm_support_dihedral_classes=len(without_units),norm_support_common_unit_classes=len(with_units),
                common_unit_duplicate_anchors=len(seen)-len(with_units),common_unit_duplicate_fraction=1-len(with_units)/len(seen))
            rows.append(row);print(json.dumps(row),flush=True)
    atomic_write_json(args.output/'report.json',dict(complete=True,rows=rows,
        input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),coarse_input_sha256=hashlib.sha256(args.coarse_input.read_bytes()).hexdigest(),
        scope='All literal anchors of the stated radius balls. Equivalence only after ALL original k/eta are completely UNSAT. '
              'Common units map the full target set bijectively; norm plus support determines every remaining equation. '
              'SAT and UNKNOWN cannot exclude other homometric signed rows/new full classes.',
        claim='Combinatorial proof-domain redundancy, not a measured new-discovery speed improvement.'))
    script=args.output/'audit_source.py';shutil.copy2(__file__,script)


if __name__=='__main__':main()
