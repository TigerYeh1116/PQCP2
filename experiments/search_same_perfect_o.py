"""Complete existence check of the identical-perfect-O even-negative L86 family.

The certified primitive odd kernel reduces all integer trit differences to
21 signed basis choices. Solve the remaining single C word exactly; only an
independently verified full length-86 pair counts as a new discovery.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.benchmark_l86_discovery import discovery_key
from experiments.search_perfect43_basis import perfect43_bases
from solver.checkpoint import atomic_write_json
from solver.perfect_odd_kernel import interleave_from_complementary_trits,perfect43_kernel_certificate
from solver.search_runner import append_verified_solution_if_new
from solver.structured_energy import integer_periodic_autocorrelation_profile as pacf,target_profile
from solver.ternary_system_native import prepare_native_system,run_native_system
from solver.verifier import verify_pqcp


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    tick=perf_counter()
    source=perfect43_bases()
    rows,keys,first_hit=[],set(),None
    for ingredient in source['rows']:
        o=tuple(ingredient['signs'])
        certificate=perfect43_kernel_certificate(o)
        for t,g in enumerate(certificate['basis'],1):
            k=min(2*t,43-2*t)
            mask=sum(int(v==0)<<i for i,v in enumerate(g))
            wanted=tuple(21 if u==0 else 10 if u in (k,43-k) else 1 for u in range(43))
            if tuple(pacf(g))!=tuple(22 if u==0 else -11 if u in (k,43-k) else 0 for u in range(43)):
                raise RuntimeError('basis difference PACF formula failed')
            plan=prepare_native_system((mask,0),43,wanted,(6,0))
            out=run_native_system(plan,2**23)
            record=dict(o_mask=ingredient['mask'],t=t,k=k,c_support=mask,
                primitive_certificate={a:b for a,b in certificate.items() if a!='basis'},result=out)
            if out['words'] is not None:
                c=out['words'][0]
                if tuple(pacf(c))!=wanted or sum(c)!=9:
                    raise RuntimeError('single-word C witness failed independent integer audit')
                words=interleave_from_complementary_trits(c,g,o)
                check=verify_pqcp(*words)
                if not check.is_valid or tuple(check.profile)!=target_profile(86,2*k,-1):
                    raise RuntimeError('primitive-kernel construction failed full target audit')
                if first_hit is None:first_hit=perf_counter()-tick
                keys.add(discovery_key(*words))
                record['full_words']=words
                append_verified_solution_if_new(86,*words,args.output)
            rows.append(record)
    report=dict(rows=rows,statuses=dict(Counter(row['result']['status'] for row in rows)),
        complete=all(row['result']['status']!='UNKNOWN' for row in rows),
        independently_verified_new_classes=len(keys),first_verified_elapsed=first_hit,
        elapsed=perf_counter()-tick,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='ALL E_A/E_B signs and all identical perfect O translation classes, even negative target; existence tested, not enumeration of every sign witness or all L86 PQCPs.',
        theorem='Primitive integral odd kernel: ranks mod 2/11 and Gram=22I force each trit difference to one basis vector or zero. Zero cannot satisfy normal even sidelobes.')
    atomic_write_json(args.output/'report.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='rows'}),flush=True)


if __name__=='__main__':main()
