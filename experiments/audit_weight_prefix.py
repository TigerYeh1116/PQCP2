"""Frozen full-zero-set audit of incremental fixed-weight prefix pruning."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.benchmark_l86_discovery import freeze_sources
from experiments import benchmark_weight_group_pipeline as pipeline
from solver.checkpoint import atomic_write_json
from solver.compressed_repair import _validated_reference
from solver.mod8_model_reference import all_mod8_indices
from solver.projection_group_targets import group_projection_targets
from solver.structured_energy import integer_periodic_autocorrelation_profile as pacf
from solver.ternary_support_search import support_mask
from solver.ternary_system_native import prepare_native_system,run_native_system,_library
from solver.weight_group_targets import group_weight_targets
from solver.weight_prefix_native import prepare_weight_prefix,native_arguments
from solver import weight_prefix_native as prefix


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--domains',type=int,default=96)
    parser.add_argument('--carriers',type=Path)
    parser.add_argument('--count-elimination',action='store_true')
    parser.add_argument('--affine-pruning',action='store_true')
    parser.add_argument('--field-rebasing',action='store_true')
    parser.add_argument('--multi-pivot',action='store_true')
    parser.add_argument('--_frozen-runner',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args();output=args.output.resolve();snapshot=output/'source_snapshot'
    if args.affine_pruning and not args.count_elimination:parser.error('affine pruning requires count elimination')
    if args.field_rebasing and (not args.count_elimination or args.affine_pruning):parser.error('select one affine backend with count elimination')
    if args.multi_pivot and (not args.count_elimination or args.affine_pruning or args.field_rebasing):parser.error('select one affine backend with count elimination')
    if not 1<=args.domains<=512:parser.error('1..512 complete domains required')
    if not args._frozen_runner:
        output.mkdir(parents=True,exist_ok=False);freeze_sources(snapshot)
        for name in ('benchmark_l86_discovery.py','benchmark_weight_group_pipeline.py','search_norm_supports.py',Path(__file__).name):
            path=snapshot/'experiments'/name;path.parent.mkdir(exist_ok=True);shutil.copy2(ROOT/'experiments'/name,path)
        if args.carriers is not None:shutil.copy2(args.carriers.resolve(),output/'input_carriers.json')
        manifest={str(p.relative_to(snapshot)):hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in sorted(snapshot.rglob('*')) if p.is_file() and p.name!='source_manifest.json'}
        atomic_write_json(snapshot/'source_manifest.json',manifest)
        command=[sys.executable,str(snapshot/'experiments'/Path(__file__).name),'--_frozen-runner',
                 '--output',str(output),'--domains',str(args.domains)]
        if args.carriers is not None:command.extend(('--carriers',str(output/'input_carriers.json')))
        if args.count_elimination:command.append('--count-elimination')
        if args.affine_pruning:command.append('--affine-pruning')
        if args.field_rebasing:command.append('--field-rebasing')
        if args.multi_pivot:command.append('--multi-pivot')
        atomic_write_json(output/'execution.json',dict(actual_frozen_execution=True,command=command,cwd=str(snapshot),
            source_manifest_sha256=hashlib.sha256(pipeline.encode(manifest).encode()).hexdigest()))
        raise SystemExit(subprocess.run(command,cwd=snapshot).returncode)
    if ROOT!=snapshot:raise RuntimeError('actual archived execution required')
    manifest=json.loads((snapshot/'source_manifest.json').read_text())
    if any(hashlib.sha256((snapshot/n).read_bytes()).hexdigest()!=h for n,h in manifest.items()):
        raise RuntimeError('frozen source changed')
    import experiments.search_norm_supports as search
    search.initialize();groups=group_weight_targets(group_projection_targets(search.CHOICES))
    atomic_write_json(output/'target_domain.json',json.loads(pipeline.encode(groups)))
    domain=set()
    for group in groups:
        for entry in group['c_groups']:
            refs=tuple(tuple([-1]*n+[0]*z+[1]*p) for n,z,p in entry['contents'])
            wanted=_validated_reference(*refs,entry['targets'][0])[3]
            domain.add((wanted,entry['contents']))
            domain.add((wanted,entry['contents'][::-1]))
    sources=[ROOT/'csrc'/name for name in ('ternary_system.c','ternary_signs.c','ternary_weight_signs.c','ternary_quadratic_signs.c')]
    binary=output/('audit_prefix.dylib' if sys.platform=='darwin' else 'audit_prefix.so')
    compile_command=['cc','-O3','-std=c11','-Wall','-Wextra','-Werror','-fPIC',
        '-DPQCP_WEIGHT_PREFIX_AUDIT_SYNDROME','-dynamiclib' if sys.platform=='darwin' else '-shared',*map(str,sources),'-o',str(binary)]
    subprocess.run(compile_command,check=True,capture_output=True,timeout=30)
    diagnostic=ctypes.CDLL(str(binary));normal=_library()
    diagnostic.pqcp_affine_sign_plan.argtypes=normal.pqcp_affine_sign_plan.argtypes
    diagnostic.pqcp_affine_sign_plan.restype=ctypes.c_int
    diagnostic.pqcp_weight_signs.argtypes=normal.pqcp_weight_signs.argtypes
    diagnostic.pqcp_weight_signs.restype=ctypes.c_int
    fn=(diagnostic.pqcp_weight_prefix_multi_models if args.multi_pivot else
        diagnostic.pqcp_weight_prefix_rebased_models if args.field_rebasing else
        diagnostic.pqcp_weight_prefix_affine_models if args.affine_pruning else
        diagnostic.pqcp_weight_prefix_count_models if args.count_elimination else diagnostic.pqcp_weight_prefix_models)
    qp=ctypes.POINTER(ctypes.c_uint32);up=ctypes.POINTER(ctypes.c_uint64)
    fn.argtypes=normal.pqcp_weight_signs.argtypes[:-3]+[ctypes.c_uint64,qp,up]+([up] if args.count_elimination else []);fn.restype=ctypes.c_int
    diagnostic.pqcp_weight_prefix_signs.argtypes=normal.pqcp_weight_signs.argtypes
    diagnostic.pqcp_weight_prefix_signs.restype=ctypes.c_int
    observations=[];candidate_checks=full_parameters=eliminated_patterns=contradictory_patterns=affine_contradictions=0
    multi_patterns=multi_bits=0
    source_rows=None if args.carriers is None else json.loads(args.carriers.read_text())['rows']
    for trial in range(args.domains*128 if source_rows is None else len(source_rows)):
        source_word=None
        if source_rows is None:
            group=groups[trial%len(groups)];entry=group['c_groups'][trial%len(group['c_groups'])]
            u,sampled=pipeline.sample(group,trial,202610096)
            if sampled['mask'] is None:continue
            masks=(u,sampled['mask']);refs=tuple(tuple([-1]*n+[0]*z+[1]*p) for n,z,p in entry['contents'])
            a,b,d,wanted=_validated_reference(*refs,entry['targets'][0]);negative=(a.count(-1),b.count(-1))
            if trial%2:masks=masks[::-1];negative=negative[::-1]
        else:
            item=source_rows[trial];source_word=tuple(tuple(w) for w in item['result']['attempts']['coarse']['words'])
            wanted=tuple(x+y for x,y in zip(pacf(source_word[0]),pacf(source_word[1])))
            contents=tuple((w.count(-1),w.count(0),w.count(1)) for w in source_word)
            if (wanted,contents) not in domain:raise RuntimeError('source carrier is outside fresh proven content/norm cases')
            masks=tuple(support_mask(w) for w in source_word);negative=tuple(w.count(-1) for w in source_word);d=43
        plan=prepare_native_system(masks,d,wanted,negative)
        prepared=prepare_weight_prefix(plan,2**23)
        if prepared is None:continue
        p=int(prepared['meta'][5]);f=int(prepared['meta'][4]);q=p+f;capacity=1<<q
        u64,u32=ctypes.c_uint64,ctypes.c_uint32
        toggles=(u64*(2*q))(*prepared['patterns'][:2*p],*prepared['free'][:2*f])
        constant=u32();linear=(u32*q)();quadratic=(u32*(q*q))()
        code=normal.pqcp_quadratic_plan(plan['mask_array'],prepared['base'],toggles,q,d,plan['wanted'],plan['negative'],
                                      ctypes.byref(constant),linear,quadratic)
        if code:raise RuntimeError('independent full parameter coefficients failed')
        reference=all_mod8_indices((constant.value,tuple(linear),tuple(quadratic)),capacity=capacity)
        expected=[]
        for index in reference['indices']:
            pattern=index&((1<<p)-1)
            if (bin(pattern).count('1')==negative[prepared['selected']] and
                all(bin(pattern&prepared['checks'][j]).count('1')%2==prepared['rhs'][j] for j in range(prepared['meta'][6]))):
                expected.append(index)
        output_models=(u32*capacity)();count=u64()
        extra=(u64*5)();native_args=(*native_arguments(prepared),capacity,output_models,ctypes.byref(count))
        code=fn(*native_args,extra) if args.count_elimination else fn(*native_args)
        if code or not reference['complete'] or count.value>capacity or tuple(sorted(output_models[:count.value]))!=tuple(expected):
            raise RuntimeError('a complete original modular model was lost or added')
        old_library=prefix._library;prefix._library=lambda:diagnostic
        try:result=prefix.run_weight_prefix(plan,2**23,count_elimination=args.count_elimination,affine_pruning=args.affine_pruning,field_rebasing=args.field_rebasing,multi_pivot=args.multi_pivot)
        finally:prefix._library=old_library
        baseline=run_native_system(plan,2**23,weight_prefix=False)
        if result['status']!=baseline['status']:raise RuntimeError('integer decisions differ')
        if source_word is not None and result['status']!='SAT_COMPRESSED':raise RuntimeError('actual positive carrier was lost')
        observations.append(dict(trial=trial,masks=masks,negative=negative,wanted=wanted,p=p,f=f,selected=prepared['selected'],
            modular_models=count.value,full_parameter_assignments=capacity,status=result['status'],words=result['words']))
        candidate_checks+=count.value;full_parameters+=capacity
        eliminated_patterns+=extra[0];contradictory_patterns+=extra[1]
        affine_contradictions+=extra[2]
        multi_patterns+=extra[3];multi_bits+=extra[4]
        if len(observations)==args.domains:break
    if len(observations)!=args.domains:raise RuntimeError('insufficient eligible domains; no complete audit claimed')
    report=dict(complete=True,actual_frozen_execution=True,domains=len(observations),
        full_parameter_assignments=full_parameters,directly_audited_survivor_syndromes=candidate_checks,
        original_modular_zero_sets_match=True,all_integer_decisions_match=True,
        source_positive_carriers=source_rows is not None,observations=observations,
        count_elimination=args.count_elimination,count_eliminated_patterns=eliminated_patterns,
        count_contradictory_patterns=contradictory_patterns,
        affine_pruning=args.affine_pruning,affine_contradictory_patterns=affine_contradictions,
        field_rebasing=args.field_rebasing,
        multi_pivot=args.multi_pivot,multi_pivot_patterns=multi_patterns,multi_eliminated_bits=multi_bits,
        diagnostic_compile_command=compile_command,diagnostic_binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
        claim='Complete modular sets after exact selected-row weight and projected checks; independent old integer decisions. All positive outputs separately recomputed. No full L86 discoveries or global exclusion.')
    atomic_write_json(output/'report.json',report)
    print(pipeline.encode({k:report[k] for k in ('complete','domains','full_parameter_assignments','directly_audited_survivor_syndromes')}),flush=True)


if __name__=='__main__':main()
