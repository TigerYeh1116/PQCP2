"""Frozen CPU bit-tail whole-model audit and integer residual diagnostic.

Input cases are literal L43 sign problems from the existing frozen GPU audit.
Subproblem timings do not measure discovery of complete new L86 classes.
"""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
from time import perf_counter

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.benchmark_l86_discovery import freeze_sources
from solver.checkpoint import atomic_write_json
from solver.mps_mod8_prefix import quadratic_coefficients,complete_mod8_models
from solver.mod8_model_reference import all_mod8_indices
from solver.structured_energy import integer_periodic_autocorrelation_profile as pacf
from solver.ternary_system_native import _library,prepare_native_system,run_native_system


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--input',type=Path,default=ROOT/'results/L86_mod8_bit_sliced_final_frontier_audit_20261008/input_cases.json')
    p.add_argument('--_frozen',action='store_true',help=argparse.SUPPRESS)
    args=p.parse_args();args.output=args.output.resolve()
    if not args._frozen:
        args.output.mkdir(parents=True,exist_ok=False);snapshot=args.output/'source_snapshot';freeze_sources(snapshot)
        for name in ('verify_native_bit_tail.py','benchmark_l86_discovery.py'):
            dst=snapshot/'experiments'/name;dst.parent.mkdir(exist_ok=True);shutil.copy2(ROOT/'experiments'/name,dst)
        shutil.copy2(args.input,args.output/'input_cases.json')
        manifest={str(f.relative_to(snapshot)):hashlib.sha256(f.read_bytes()).hexdigest()
                  for f in sorted(snapshot.rglob('*')) if f.is_file() and f.name!='source_manifest.json'}
        atomic_write_json(snapshot/'source_manifest.json',manifest)
        code=subprocess.run([sys.executable,str(snapshot/'experiments/verify_native_bit_tail.py'),
                             '--_frozen','--output',str(args.output)],cwd=snapshot).returncode
        if code:raise SystemExit(code)
        return
    if ROOT!=(args.output/'source_snapshot'):raise RuntimeError('actual frozen execution required')
    manifest=json.loads((ROOT/'source_manifest.json').read_text())
    if any(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha for name,sha in manifest.items()):
        raise RuntimeError('frozen source changed')
    inputs=json.loads((args.output/'input_cases.json').read_text());cases=inputs['fixtures']+inputs['frontiers']
    plans=[prepare_native_system(tuple(r['masks']),43,tuple(r['wanted']),tuple(r['negative']),conditioning=False) for r in cases]
    lib=_library();model=lib.pqcp_quadratic_bit_tail_models
    model.argtypes=lib.pqcp_quadratic_hybrid_signs.argtypes[:-2]+[
        ctypes.POINTER(ctypes.c_uint32),ctypes.POINTER(ctypes.c_uint64)];model.restype=ctypes.c_int
    bit=lib.pqcp_quadratic_bit_tail_signs;bit.argtypes=lib.pqcp_quadratic_hybrid_signs.argtypes;bit.restype=ctypes.c_int
    # Diagnostic build audits EVERY retained projected-tail syndrome against
    # literal signed integer cuts. Tail 8 forces the original hybrid core.
    audit_path=args.output/('residual_audit'+('.dylib' if sys.platform=='darwin' else '.so'))
    sources=[ROOT/'csrc'/name for name in ('ternary_system.c','ternary_signs.c','ternary_weight_signs.c','ternary_quadratic_signs.c')]
    command=['cc','-O3','-std=c11','-Wall','-Wextra','-Werror','-DPQCP_HYBRID_AUDIT_RESIDUAL',
             '-fPIC','-dynamiclib' if sys.platform=='darwin' else '-shared',*[str(s) for s in sources],'-o',str(audit_path)]
    subprocess.run(command,check=True,capture_output=True)
    audit=ctypes.CDLL(str(audit_path));normal=lib.pqcp_quadratic_hybrid_signs
    checked=audit.pqcp_quadratic_hybrid_signs;checked.argtypes=normal.argtypes;checked.restype=ctypes.c_int
    counts=[];projected=0
    def solve(fn,plan,tail):
        words=(ctypes.c_uint64*2)();stats=(ctypes.c_uint64*7)()
        code=fn(plan['mask_array'],plan['base'],plan['toggles'],plan['meta'][2],43,
                plan['wanted'],plan['negative'],tail,2**23,words,stats)
        if code not in (0,1):raise RuntimeError('complete native fibre was not decided')
        if code==1:
            rows=tuple(tuple((-1 if words[s]>>i&1 else 1) if plan['masks'][s]>>i&1 else 0 for i in range(43)) for s in range(2))
            if (tuple(w.count(-1) for w in rows)!=tuple(plan['negative']) or
                tuple(x+y for x,y in zip(pacf(rows[0]),pacf(rows[1])))!=tuple(plan['wanted'])):
                raise RuntimeError('literal integer witness failed')
        return code,tuple(words) if code else None,tuple(stats)
    for plan in plans:
        reference=all_mod8_indices(quadratic_coefficients(plan),capacity=4096)
        if not reference['complete']:raise RuntimeError('independent reference overflow')
        output=(ctypes.c_uint32*4096)();count=ctypes.c_uint64()
        arguments=(plan['mask_array'],plan['base'],plan['toggles'],plan['meta'][2],43,plan['wanted'],plan['negative'])
        if model(*arguments,6,4096,output,ctypes.byref(count)) or count.value>4096:
            raise RuntimeError('complete model diagnostic failed')
        if tuple(sorted(output[:count.value]))!=reference['indices']:
            raise RuntimeError('bit-tail omitted or added an original modular model')
        counts.append(count.value)
        # Cap one: count remains exact; never interpret a truncated list as UNSAT.
        tiny=(ctypes.c_uint32*1)();small=ctypes.c_uint64()
        if model(*arguments,6,1,tiny,ctypes.byref(small)) or small.value!=count.value:
            raise RuntimeError('overflow model count changed')
        decision=complete_mod8_models(plan,reference)['status']
        if ('UNSAT_FIXED_SUPPORT','SAT_COMPRESSED')[solve(bit,plan,6)[0]]!=decision:
            raise RuntimeError('exact integer decision differs from independent full models')
        old=solve(normal,plan,8);diagnostic=solve(checked,plan,8)
        if old!=diagnostic:raise RuntimeError('projected original-residual diagnostic mismatch')
        projected+=diagnostic[2][5]
    report=dict(complete=True,actual_frozen_execution=True,whole_original_mod8_sets_audited=len(plans),
        all_original_mod8_model_counts=counts,exact_integer_decisions_and_witnesses_audited=True,
        overflow_counts_exact=True,residual_diagnostic_projected_candidates=projected,
        diagnostic_compile_command=command,input_sha256=hashlib.sha256((args.output/'input_cases.json').read_bytes()).hexdigest(),
        source_manifest_sha256=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest(),
        claim='Whole modular model sets and complete sign decisions only; no new full PQCP speed claim.')
    atomic_write_json(args.output/'report.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
