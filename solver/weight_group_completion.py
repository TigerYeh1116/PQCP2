"""Exact OR of full-target edges with memoized, identical C/G sign nodes.

An edge is C_SAT AND G_SAT. Only a complete UNSAT endpoint closes an
edge. The group is UNSAT only when every requested edge is closed.
Cache keys include literal supports, ALL integer norm values and counts.
"""
from dataclasses import asdict
from time import perf_counter

from .compressed_repair import _validated_reference
from .odd_lift_native import difference_sign_problem,reconstruct_odd_pair
from .projection_group_completion import complete_projection_group
from .structured_energy import integer_periodic_autocorrelation_profile as pacf
from .ternary_support_search import support_mask
from .ternary_system_native import prepare_native_system,run_native_system


def complete_weight_group(group,masks,*,max_assignments=2**23,shared=True,even_sign_orbit=False,paired_c=False,weight_prefix=False):
    started=perf_counter();masks=tuple(masks)
    if any(type(v) is not bool for v in (shared,even_sign_orbit,paired_c,weight_prefix)):
        raise ValueError('boolean shared, orbit, paired-C and weight-prefix flags required')
    if type(max_assignments) is not int or not 1<=max_assignments<=2**31:
        raise ValueError('bounded native assignment budget required')
    entries=tuple(group['c_groups'])
    if not entries:raise ValueError('nonempty weight group required')
    prepared=[]
    for entry in entries:
        references=tuple(tuple([-1]*n+[0]*z+[1]*p) for n,z,p in entry['contents'])
        targets=tuple(entry['targets']);a,b,d,wanted=_validated_reference(*references,targets[0])
        if (d>47 or len(masks)!=2 or any(type(m) is not int or not 0<=m<1<<d for m in masks)
                or tuple(bin(m).count('1') for m in masks)!=tuple(sum(v!=0 for v in w) for w in references)):
            raise ValueError('literal supports must match every requested content case')
        for target in targets[1:]:
            if _validated_reference(a,b,target)[3]!=wanted:raise ValueError('one C group must have exactly one C norm')
        prepared.append((references,targets,d,wanted,(a.count(-1),b.count(-1))))
    records=[];nodes={};plans={};prefix_results={};unresolved=False

    def prefix_result(key,plan):
        if not weight_prefix or even_sign_orbit:return None
        if key not in prefix_results:
            from .weight_prefix_native import run_weight_prefix
            prefix_results[key]=run_weight_prefix(plan,max_assignments)
        return prefix_results[key]

    def finish(status,words=None,target=None):
        return dict(status=status,verified=words is not None,words=words,
            actual_target=None if target is None else asdict(target),
            c_groups=records,sign_nodes=list(nodes.values()),elapsed=perf_counter()-started,
            scope='Exactly the requested full target/content edges on these literal supports; no unvisited edge exclusions.')

    def solve(supports,d,wanted,negative):
        key=(tuple(supports),tuple(wanted),tuple(negative))
        if key not in nodes:
            plan=plans.get(key)
            if plan is None:plan=prepare_native_system(key[0],d,key[1],key[2])
            result=prefix_result(key,plan)
            if result is None:result=run_native_system(plan,max_assignments,even_sign_orbit=even_sign_orbit)
            words=result['words']
            if words is not None and (tuple(support_mask(w) for w in words)!=key[0]
                    or tuple(w.count(-1) for w in words)!=key[2]
                    or tuple(x+y for x,y in zip(pacf(words[0]),pacf(words[1])))!=key[1]):
                raise RuntimeError('memoized sign node failed independent integer audit')
            nodes[key]=dict(supports=key[0],wanted=key[1],negative=key[2],result=result)
        return nodes[key]['result']

    # This optional path changes only evaluation sharing. Both raw affine
    # cosets retain their own integer target and complete per-label budget.
    # Ineligible bases, conditioning and partial budgets use the old solver.
    if shared and (paired_c or weight_prefix) and not even_sign_orbit and len(prepared)==2:
        keys=[];pair=[]
        for _,_,d,wanted,negative in prepared:
            key=(masks,tuple(wanted),tuple(negative));keys.append(key)
            if key not in plans:plans[key]=prepare_native_system(masks,d,wanted,negative)
            pair.append(plans[key])
            result=prefix_result(key,plans[key])
            if result is not None:nodes[key]=dict(supports=key[0],wanted=key[1],negative=key[2],result=result)
        if paired_c and keys[0]!=keys[1] and not any(key in nodes for key in keys):
            from .paired_signs_native import run_native_pair
            results=run_native_pair(pair,max_assignments)
            if results is not None:
                for key,result in zip(keys,results):
                    nodes[key]=dict(supports=key[0],wanted=key[1],negative=key[2],result=result)

    for references,targets,d,wanted,negative in prepared:
        if not shared:
            result=complete_projection_group(*references,targets,masks,max_assignments=max_assignments,even_sign_orbit=even_sign_orbit)
            records.append(dict(targets=[asdict(t) for t in targets],result=result))
            if result['verified']:return finish('SAT_PQCP',result['words'],targets[next(i for i,t in enumerate(targets) if asdict(t)==result['actual_target'])])
            if result['status']=='UNKNOWN':unresolved=True
            continue
        c=solve(masks,d,wanted,negative)
        record=dict(targets=[asdict(t) for t in targets],coarse=c,lifts=[]);records.append(record)
        if c['status']=='UNSAT_FIXED_SUPPORT':continue
        # Ordering does not change the Boolean proof. Retain every target
        # whose G node is unresolved; complete G failures are key-specific.
        for target in targets:
            gm,gw,gn=difference_sign_problem(masks,target);g=solve(gm,d,gw,gn)
            record['lifts'].append(dict(target=asdict(target),result=g))
            if g['status']=='UNSAT_FIXED_SUPPORT':continue
            if c['status']=='SAT_COMPRESSED' and g['status']=='SAT_COMPRESSED':
                return finish('SAT_PQCP',reconstruct_odd_pair(c['words'],g['words'],target),target)
            unresolved=True
    return finish('UNKNOWN' if unresolved else 'UNSAT_BINARY_WEIGHT_GROUP')
