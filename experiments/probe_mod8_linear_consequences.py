"""Audit degree-three linear consequences of the existing exact mod8 system.

Boolean products use z_i**2=z_i. Multiply each quadratic equation by 1
and each variable, then eliminate higher-degree monomials first. Every
reported consequence carries an independently replayed row-combination
certificate. This diagnostic does not enable a new production exclusion.
"""
import argparse
import ctypes
import hashlib
from itertools import combinations
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from solver.checkpoint import atomic_write_json
from solver.compressed_repair import _validated_reference
from solver.odd_lift_native import difference_sign_problem
from solver.target_profiles import TargetContentProfile
from solver.ternary_support_search import support_mask
from solver.ternary_system_native import _library,prepare_native_system,run_native_system


def degree_three_consequences(plan):
    q=int(plan['meta'][2]);equations=plan['d']//2+2
    if not plan['consistent'] or q>24:raise ValueError('consistent raw affine space through dimension 24 required')
    u=ctypes.c_uint32;constant=u();linear=(u*q)();quadratic=(u*(q*q))()
    if _library().pqcp_quadratic_plan(plan['mask_array'],plan['base'],plan['toggles'],q,plan['d'],
            plan['wanted'],plan['negative'],ctypes.byref(constant),linear,quadratic):
        raise RuntimeError('native quadratic coefficient construction failed')
    originals=[]
    for equation in range(equations):
        terms=set()
        if constant.value>>equation&1:terms.add(0)
        for i in range(q):
            if int(linear[i])>>equation&1:terms.add(1<<i)
            for j in range(i+1,q):
                if int(quadratic[i*q+j])>>equation&1:terms.add((1<<i)|(1<<j))
        originals.append(terms)
    monomials=[0]
    for degree in (1,2,3):monomials.extend(sum(1<<i for i in group) for group in combinations(range(q),degree))
    positions={term:i for i,term in enumerate(monomials)}
    basis={};row_number=0
    for original in originals:
        for multiplier in [0]+[1<<j for j in range(q)]:
            row=0
            for term in original:row^=1<<positions[term|multiplier]
            provenance=1<<row_number;row_number+=1
            while row:
                pivot=row.bit_length()-1
                if pivot not in basis:basis[pivot]=(row,provenance);break
                other,proof=basis[pivot];row^=other;provenance^=proof
    results=[]
    for pivot,(row,proof) in sorted(basis.items()):
        if pivot>q:continue
        expected={monomials[j] for j in range(q+1) if row>>j&1}
        # Independent symbolic-set replay, rather than reuse of bitset rows.
        replay=set();remaining=proof
        while remaining:
            bit=remaining&-remaining;number=bit.bit_length()-1;remaining^=bit
            equation,multiplier=divmod(number,q+1)
            factor=0 if not multiplier else 1<<(multiplier-1)
            product=set()
            for term in originals[equation]:
                value=term|factor
                if value in product:product.remove(value)
                else:product.add(value)
            replay.symmetric_difference_update(product)
        if replay!=expected:raise RuntimeError('derived polynomial failed independent certificate replay')
        results.append(dict(constant=bool(row&1),coefficients=(row>>1)&((1<<q)-1),
                            original_multiplier_rows_xor=hex(proof)))
    return dict(free_variables=q,input_equations=equations,multiplied_rows=row_number,
                monomials=len(monomials),linear_rank=sum(r['coefficients']!=0 for r in results),
                constant_contradiction=any(not r['coefficients'] and r['constant'] for r in results),
                consequences=results,all_certificates_independently_replayed=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=ROOT/'results/L86_independent_norm_phase_million_20261008/coarse_witnesses.json')
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():p.error('output already exists; preserve earlier evidence')
    inputs=json.loads(args.input.read_text())['rows'];rows=[]
    for source in inputs:
        target=TargetContentProfile(**source['target']);c=tuple(tuple(w) for w in source['result']['attempts']['coarse']['words'])
        a,b,d,wanted_c=_validated_reference(*c,target);masks=tuple(support_mask(w) for w in c)
        masks_g,wanted_g,negative_g=difference_sign_problem(masks,target)
        for kind,supports,wanted,negative in (('C',masks,wanted_c,(a.count(-1),b.count(-1))),('G',masks_g,wanted_g,negative_g)):
            plan=prepare_native_system(supports,d,wanted,negative,conditioning=False)
            if not plan['consistent'] or plan['meta'][2]>24:
                rows.append(dict(trial=source['trial'],kind=kind,consistent=plan['consistent'],unsupported_dimension=plan['meta'][2]>24));continue
            tick=perf_counter();probe=degree_three_consequences(plan);elapsed=perf_counter()-tick
            exact=run_native_system(plan,2**23)
            if probe['constant_contradiction'] and (exact['status']!='UNSAT_FIXED_SUPPORT' or exact.get('quadratic_matches',0)):
                raise RuntimeError('constant contradiction conflicts with complete native results')
            row=dict(trial=source['trial'],kind=kind,masks=supports,wanted=wanted,negative=negative,
                     probe=probe,probe_seconds=elapsed,native=exact)
            rows.append(row)
            print(json.dumps(dict(trial=source['trial'],kind=kind,linear_rank=probe['linear_rank'],
                contradiction=probe['constant_contradiction'],seconds=elapsed)),flush=True)
    source_paths=[Path(__file__),ROOT/'csrc/ternary_quadratic_signs.c',ROOT/'csrc/ternary_system.c']
    atomic_write_json(args.output,dict(rows=rows,complete=len(rows)==2*len(inputs),
        input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
        scope='Audited degree-three necessary consequences on selected literal support domains; diagnostic only, not global exclusion or discovery-speed evidence.'))


if __name__=='__main__':main()
