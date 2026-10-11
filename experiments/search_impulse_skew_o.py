"""Complete impulse/skew-O restricted family, with all finite-field preimages.

All 122880 necessary C rows are retained; O need not be perfect. Odd target
is forced mathematically, then all even lags and full86 verifier decide hits.
Frobenius skew norm has at most eight literal preimages per C. No rotation
quotient, heuristic cutoff or proof about general PQCP86 is used.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys
from time import perf_counter

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.benchmark_l86_discovery import discovery_key,freeze_sources,verified_inventory
from solver.certified_profile_psd import profile_psd_obstruction
from solver.checkpoint import atomic_write_json
from solver.impulse_skew_construction import (impulse_pair_certificate,iter_impulse_c_masks,
                                              required_skew_intersections)
from solver.perfect_odd_kernel import interleave_from_complementary_trits
from solver.sds_norm_completion import intersection_profile
from solver.skew_norm_completion import skew_norm_preimages
from solver.search_runner import append_verified_solution_if_new
from solver.structured_energy import target_profile
from solver.verifier import verify_pqcp


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'results'/(
        'L86_impulse_skew_O_'+datetime.now().strftime('%Y%m%d_%H%M%S')))
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False);tick=perf_counter()
    snapshot=args.output/'source_snapshot';freeze_sources(snapshot);(snapshot/'experiments').mkdir(exist_ok=True)
    for path in (ROOT/'experiments').glob('*.py'):shutil.copy2(path,snapshot/'experiments'/path.name)
    manifest={str(p.relative_to(snapshot)):hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(snapshot.rglob('*')) if p.is_file() and p.name!='source_manifest.json'}
    atomic_write_json(snapshot/'source_manifest.json',manifest)
    certificate=impulse_pair_certificate();atomic_write_json(args.output/'pair_xor_certificate.json',certificate)
    raw=(ROOT/'86.txt').read_bytes() if (ROOT/'86.txt').exists() else b''
    (args.output/'initial_inventory.txt').write_bytes(raw);known=verified_inventory(raw)
    keys=set();first_hit=None;count=psd_excluded=range_excluded=norm_empty=norm_preimages=integer_checks=0
    full_hits=[];profile_cache={}
    for negative in iter_impulse_c_masks(certificate):
        count+=1;wanted=required_skew_intersections(negative,certificate)
        if any(not 0<=v<=20 for v in wanted[1:]):range_excluded+=1;continue
        if wanted in profile_cache:out=profile_cache[wanted]
        elif profile_psd_obstruction(wanted) is not None:
            out=None
            if len(profile_cache)<4096:profile_cache[wanted]=out
        else:
            desired=sum((v%2)<<i for i,v in enumerate(wanted));out=skew_norm_preimages(desired)
            if len(profile_cache)<4096:profile_cache[wanted]=out
        if out is None:psd_excluded+=1;continue
        if not out['masks']:norm_empty+=1;continue
        norm_preimages+=len(out['masks'])
        for o_mask in out['masks']:
            integer_checks+=1
            if intersection_profile(o_mask,43)!=wanted:continue
            c=tuple(0 if certificate['holes']>>i&1 else 1-2*((negative>>i)&1) for i in range(43))
            g=tuple(int(certificate['holes']>>i&1) for i in range(43))
            o=tuple(1-2*((o_mask>>i)&1) for i in range(43))
            words=interleave_from_complementary_trits(c,g,o);check=verify_pqcp(*words)
            if not check.is_valid or tuple(check.profile)!=target_profile(86,41,1):
                raise RuntimeError('impulse/skew candidate failed independent full L86 target audit')
            full_hits.append(dict(c_negative=negative,o_mask=o_mask,words=words))
            key=discovery_key(*words)
            if key not in known:
                if key not in keys and first_hit is None:first_hit=perf_counter()-tick
                keys.add(key);append_verified_solution_if_new(86,*words,args.output)
        if count%10000==0:print(json.dumps(dict(c_rows=count,full_new_classes=len(keys),elapsed=perf_counter()-tick)),flush=True)
    if count!=122880:raise RuntimeError('C cardinality/kernel enumeration incomplete')
    report=dict(complete=True,c_rows=count,range_excluded=range_excluded,psd_excluded=psd_excluded,
                skew_norm_empty=norm_empty,skew_norm_preimages=norm_preimages,integer_profile_checks=integer_checks,
                full_hits=full_hits,independently_verified_new_classes=len(keys),first_verified_elapsed=first_hit,
                elapsed=perf_counter()-tick,source_manifest_sha256=hashlib.sha256(
                    json.dumps(manifest,sort_keys=True).encode()).hexdigest(),
                scope='ALL C signs of sum nine complementary to g=delta_1+delta_-1, ALL sum-one skew O. '
                      'No perfect-O requirement, no rotation quotient. Restricted family only, no global exclusion.')
    atomic_write_json(args.output/'report.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='full_hits'}),flush=True)


if __name__=='__main__':main()
