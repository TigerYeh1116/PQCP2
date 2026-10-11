"""One exact C problem shared by full targets with the SAME folded profile.

The logical condition is C_SAT AND (G_1_SAT OR ... OR G_t_SAT).
Only complete C-UNSAT or complete G-UNSAT for every target closes a group.
Unresolved or unvisited branches never constitute UNSAT certificates.
"""
from dataclasses import asdict
from time import perf_counter

from .compressed_repair import _validated_reference
from .odd_lift_native import difference_sign_problem,reconstruct_odd_pair
from .structured_energy import integer_periodic_autocorrelation_profile as pacf
from .ternary_support_search import _audit_sign_words,support_mask
from .ternary_system_native import prepare_native_system,run_native_system


def complete_projection_group(a,b,targets,masks,*,max_assignments=2**23,even_sign_orbit=False,difference_first=False,lazy_c_gate=True,weight_prefix=False):
    started=perf_counter();targets=tuple(targets);masks=tuple(masks)
    if not targets or len(set(targets))!=len(targets):raise ValueError('distinct compatible targets required')
    a,b,d,wanted=_validated_reference(a,b,targets[0])
    if (d>47 or len(masks)!=2 or any(type(m) is not int or not 0<=m<1<<d for m in masks)
            or tuple(bin(m).count('1') for m in masks)!=tuple(sum(v!=0 for v in w) for w in (a,b))
            or type(max_assignments) is not int or not 1<=max_assignments<=2**31
            or type(even_sign_orbit) is not bool or type(difference_first) is not bool or type(lazy_c_gate) is not bool
            or type(weight_prefix) is not bool):
        raise ValueError('literal support/content domain and valid native budget required')
    for target in targets[1:]:
        if _validated_reference(a,b,target)[3]!=wanted:
            raise ValueError('all targets must share exactly the same C sign problem')
    c_plan=prepare_native_system(masks,d,wanted,(a.count(-1),b.count(-1)))
    g_data=[difference_sign_problem(masks,t) for t in targets]
    c_result=None;g_results={}

    def finish(status,index=None):
        words=None
        if index is not None:
            words=reconstruct_odd_pair(c_result['words'],g_results[index]['words'],targets[index])
        return dict(status=status,words=words,verified=words is not None,
                    actual_target=asdict(targets[index]) if index is not None else None,
                    attempts=dict(coarse=c_result,lifts=[dict(target=asdict(t),result=g_results.get(i)) for i,t in enumerate(targets)]),
                    elapsed=perf_counter()-started,
                    scope='Exactly these folded-compatible full targets on this literal support/content pair; first verified model only.')

    def solve_c():
        nonlocal c_result
        if c_result is None:
            c_result=run_native_system(c_plan,max_assignments,even_sign_orbit=even_sign_orbit,weight_prefix=weight_prefix)
            if c_result['words'] is not None:_audit_sign_words(c_result['words'],a,b,targets[0],wanted,masks)
        return c_result['status']

    def solve_g(index):
        out=run_native_system(g_plans[index],max_assignments,even_sign_orbit=even_sign_orbit,weight_prefix=weight_prefix)
        g_results[index]=out
        if out['words'] is not None:
            gm,gw,gn=g_data[index];rows=out['words']
            if (tuple(support_mask(w) for w in rows)!=gm or tuple(w.count(-1) for w in rows)!=gn
                    or tuple(x+y for x,y in zip(pacf(rows[0]),pacf(rows[1])))!=gw):
                raise RuntimeError('group G witness failed independent integer audit')
        return out['status']

    # C is a NECESSARY factor for every full target in the group. A complete
    # C-UNSAT closes it before constructing any G affine/cardinality plan.
    # This is not a transfer of G failures or a heuristic cost assumption.
    # UNKNOWN never closes the group: all required G alternatives still run.
    if lazy_c_gate and not difference_first and solve_c()=='UNSAT_FIXED_SUPPORT':
        return finish('UNSAT_BINARY_SUPPORT_GROUP')
    g_plans=[prepare_native_system(m,d,w,n) for m,w,n in g_data]
    if not c_plan['consistent'] and not difference_first:
        solve_c();return finish('UNSAT_BINARY_SUPPORT_GROUP')
    order=sorted(range(len(targets)),key=lambda i:(g_plans[i]['estimated_assignments'],i))
    # Exact affine work counts affect ordering only; the logical proof below
    # remains valid for every ordering and every finite assignment budget.
    c_first=not difference_first and c_plan['estimated_assignments']<=sum(p['estimated_assignments'] for p in g_plans)
    if c_first and solve_c()=='UNSAT_FIXED_SUPPORT':return finish('UNSAT_BINARY_SUPPORT_GROUP')
    for index in order:
        status=solve_g(index)
        if status=='SAT_COMPRESSED':
            coarse_status=solve_c()
            if coarse_status=='UNSAT_FIXED_SUPPORT':return finish('UNSAT_BINARY_SUPPORT_GROUP')
            if coarse_status=='SAT_COMPRESSED':return finish('SAT_PQCP',index)
            return finish('UNKNOWN')
    if all(g_results[i]['status']=='UNSAT_FIXED_SUPPORT' for i in order):
        return finish('UNSAT_BINARY_SUPPORT_GROUP')
    if solve_c()=='UNSAT_FIXED_SUPPORT':return finish('UNSAT_BINARY_SUPPORT_GROUP')
    return finish('UNKNOWN')
