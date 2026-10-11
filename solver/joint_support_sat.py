"""Exact simultaneous support/sign completion by Boolean cut cardinalities.

Every coordinate is one of four states: C=+/-1,G=0 or C=0,G=+/-1.
Unlike C-first sampling, support variables remain shared between C and G.
No spectral rounding, distance cap, reflection assumption or SAT timeout is
an exclusion. This experimental backend is not a production default.
"""
from time import perf_counter
import ctypes

from .compressed_repair import _validated_reference
from .odd_lift_native import difference_sign_problem,reconstruct_odd_pair
from .structured_energy import integer_periodic_autocorrelation_profile as pacf
from .ternary_support_search import support_mask


def build_joint_support_model(a,b,target,*,fixed_supports=None,fixed_c_a=None,
                              known_coarse_words=None,known_difference_words=None,rotation_gauge=True,seed=123,
                              affine_projection=False,mod8_projection=False):
    import z3
    a,b,d,wanted_c=_validated_reference(a,b,target)
    weights=tuple(sum(v!=0 for v in w) for w in (a,b))
    reference_masks=tuple(support_mask(w) for w in (a,b))
    _,wanted_g,negative_g=difference_sign_problem(reference_masks,target)
    if fixed_supports is not None:
        fixed_supports=tuple(fixed_supports)
        if (len(fixed_supports)!=2 or any(type(m) is not int or not 0<=m<1<<d for m in fixed_supports)
                or tuple(bin(m).count('1') for m in fixed_supports)!=weights):
            raise ValueError('literal fixed supports must preserve the reference contents')
    known={}
    for kind,rows,wanted,complementary in (('C',known_coarse_words,wanted_c,False),('G',known_difference_words,wanted_g,True)):
        if rows is None:continue
        if fixed_supports is None:raise ValueError('witness reuse requires fixed literal supports')
        rows=tuple(tuple(w) for w in rows)
        expected_masks=tuple(((1<<d)-1)^m if complementary else m for m in fixed_supports)
        if (len(rows)!=2 or any(len(w)!=d or any(type(v) is not int or v not in (-1,0,1) for v in w) for w in rows)
                or tuple(support_mask(w) for w in rows)!=expected_masks
                or tuple(w.count(-1) for w in rows)!=(negative_g if complementary else (a.count(-1),b.count(-1)))
                or tuple(x+y for x,y in zip(pacf(rows[0]),pacf(rows[1])))!=wanted):
            raise ValueError('known sign witness failed independent exact support/count/norm audit')
        known[kind]=rows
    if fixed_c_a is not None:
        fixed_c_a=tuple(fixed_c_a)
        if (len(fixed_c_a)!=d or any(type(v) is not int or v not in (-1,0,1) for v in fixed_c_a)
                or sorted(fixed_c_a)!=sorted(a)):
            raise ValueError('fixed C_A must preserve all ternary contents')
        if fixed_supports is not None and support_mask(fixed_c_a)!=fixed_supports[0]:
            raise ValueError('fixed C_A and support disagree')
        if 'C' in known and known['C'][0]!=fixed_c_a:
            raise ValueError('known C witness would change the fixed C_A search domain')
    if type(rotation_gauge) is not bool or type(seed) is not int or not 0<=seed<2**31:
        raise ValueError('boolean rotation gauge and bounded integer seed required')
    if (type(affine_projection) is not bool or type(mod8_projection) is not bool
            or (mod8_projection and not affine_projection)
            or (affine_projection and (fixed_supports is None or d>47))):
        raise ValueError('exact affine/mod8 projection requires fixed supports, d<=47 and ordered boolean flags')
    started=perf_counter()
    solver=z3.SolverFor('QF_FD');solver.set(random_seed=seed)
    supports=[[z3.Bool('S_'+str(s)+'_'+str(i)) for i in range(d)] for s in range(2)]
    signs=[[z3.Bool('Z_'+str(s)+'_'+str(i)) for i in range(d)] for s in range(2)]
    def weight(bits,count):
        if not bits:return z3.BoolVal(count==0)
        return z3.PbEq([(bit,1) for bit in bits],count)
    for s,word in enumerate((a,b)):
        solver.add(weight(supports[s],weights[s]))
        if fixed_supports is None:
            solver.add(weight([z3.And(supports[s][i],signs[s][i]) for i in range(d)],word.count(-1)))
            solver.add(weight([z3.And(z3.Not(supports[s][i]),signs[s][i]) for i in range(d)],negative_g[s]))
        else:
            solver.add(weight([signs[s][i] for i in range(d) if fixed_supports[s]>>i&1],word.count(-1)))
            solver.add(weight([signs[s][i] for i in range(d) if not(fixed_supports[s]>>i&1)],negative_g[s]))
        if fixed_supports is not None:
            solver.add(*[bit==bool(fixed_supports[s]>>i&1) for i,bit in enumerate(supports[s])])
        elif not(s==0 and fixed_c_a is not None) and rotation_gauge and weights[s]:
            # Independent cyclic row rotations preserve BOTH row norms and
            # contents. Their physical binary realization is an even shift.
            # Some representative has an active C at coordinate zero.
            solver.add(supports[s][0])
    if fixed_c_a is not None:
        for i,v in enumerate(fixed_c_a):
            solver.add(supports[0][i]==bool(v))
            if v:solver.add(signs[0][i]==(v==-1))
    for rows in known.values():
        for s,row in enumerate(rows):
            for i,v in enumerate(row):
                if v:solver.add(signs[s][i]==(v==-1))
    projections={}
    if affine_projection:
        from .ternary_system_native import _library,prepare_native_system
        for kind,complementary,wanted,counts in (('C',False,wanted_c,(a.count(-1),b.count(-1))),('G',True,wanted_g,negative_g)):
            if kind in known:continue
            masks=tuple(((1<<d)-1)^m if complementary else m for m in fixed_supports)
            plan=prepare_native_system(masks,d,wanted,counts,conditioning=False)
            q=int(plan['meta'][2]);info=dict(consistent=plan['consistent'],free_variables=q,rank=int(plan['meta'][1]),mod8=False)
            projections[kind]=info
            if not plan['consistent']:solver.add(z3.BoolVal(False));continue
            variables=[z3.Bool('Q_'+kind+'_'+str(j)) for j in range(q)]
            for s,mask in enumerate(masks):
                if int(plan['base'][s])&~mask or any(int(plan['toggles'][2*j+s])&~mask for j in range(q)):
                    raise RuntimeError('affine SAT plan escaped its literal supports')
                for i in range(d):
                    if not(mask>>i&1):continue
                    value=z3.BoolVal(bool(int(plan['base'][s])>>i&1))
                    for j in range(q):
                        if int(plan['toggles'][2*j+s])>>i&1:value=z3.Xor(value,variables[j])
                    solver.add(signs[s][i]==value)
            if mod8_projection and q<=24:
                uint=ctypes.c_uint32;constant=uint();linear=(uint*q)();quadratic=(uint*(q*q))()
                status=_library().pqcp_quadratic_plan(plan['mask_array'],plan['base'],plan['toggles'],q,
                    d,plan['wanted'],plan['negative'],ctypes.byref(constant),linear,quadratic)
                if status:raise RuntimeError('exact quadratic SAT projection construction failed')
                for equation in range(d//2+2):
                    value=z3.BoolVal(bool(constant.value>>equation&1))
                    for i in range(q):
                        if int(linear[i])>>equation&1:value=z3.Xor(value,variables[i])
                        for j in range(i+1,q):
                            if int(quadratic[i*q+j])>>equation&1:
                                value=z3.Xor(value,z3.And(variables[i],variables[j]))
                    solver.add(z3.Not(value))
                info['mod8']=True
    for complementary,wanted in ((False,wanted_c),(True,wanted_g)):
        if ('G' if complementary else 'C') in known:continue
        for u in range(1,d//2+1):
            if fixed_supports is not None:
                cuts=[]
                for s in range(2):
                    mask=((1<<d)-1)^fixed_supports[s] if complementary else fixed_supports[s]
                    for i in range(d):
                        j=(i+u)%d
                        if mask>>i&1 and mask>>j&1:cuts.append(z3.Xor(signs[s][i],signs[s][j]))
                # N_C+N_D = number_of_active_edges - 2*number_of_cuts.
                difference=len(cuts)-wanted[u]
                solver.add(z3.BoolVal(False) if difference%2 or difference<0 else weight(cuts,difference//2))
                continue
            terms=[]
            for s in range(2):
                for i in range(d):
                    j=(i+u)%d
                    active=[z3.Not(supports[s][at]) if complementary else supports[s][at] for at in (i,j)]
                    different=z3.Xor(signs[s][i],signs[s][j])
                    # Each active edge contributes exactly +1 or -1.
                    terms.extend([(z3.And(*active,z3.Not(different)),1),(z3.And(*active,different),-1)])
            solver.add(z3.PbEq(terms,wanted[u]))
    return dict(solver=solver,supports=supports,signs=signs,d=d,target=target,
                references=(a,b),wanted_c=wanted_c,wanted_g=wanted_g,negative_g=negative_g,
                fixed_supports=fixed_supports,fixed_c_a=fixed_c_a,rotation_gauge=rotation_gauge,
                independently_audited_known_sign_problems=tuple(known),
                exact_affine_projections=projections,
                construction_seconds=perf_counter()-started)


def check_joint_support_model(problem,*,resource_limit=None):
    """Check the exact model; a bounded result can be UNKNOWN, never UNSAT."""
    import z3
    if resource_limit is not None and (type(resource_limit) is not int or not 1<=resource_limit<2**32):
        raise ValueError('resource_limit must be None or a positive 32-bit count')
    solver=problem['solver'];solver.set(rlimit=0 if resource_limit is None else resource_limit)
    started=perf_counter();status=solver.check();elapsed=perf_counter()-started
    common=dict(words=None,coarse_words=None,difference_words=None,verified=False,
                construction_seconds=problem['construction_seconds'],solve_seconds=elapsed,
                independently_audited_known_sign_problems=problem['independently_audited_known_sign_problems'],
                exact_affine_projections=problem['exact_affine_projections'],
                resource_limit=resource_limit,statistics=str(solver.statistics()))
    if status!=z3.sat:
        return dict(common,status='UNSAT_CONTENT_DOMAIN' if status==z3.unsat else 'UNKNOWN',
                    reason=solver.reason_unknown() if status==z3.unknown else None)
    model=solver.model();d=problem['d']
    active=[[z3.is_true(model.eval(bit,model_completion=True)) for bit in row] for row in problem['supports']]
    negative=[[z3.is_true(model.eval(bit,model_completion=True)) for bit in row] for row in problem['signs']]
    c=tuple(tuple((-1 if negative[s][i] else 1) if active[s][i] else 0 for i in range(d)) for s in range(2))
    g=tuple(tuple(0 if active[s][i] else (-1 if negative[s][i] else 1) for i in range(d)) for s in range(2))
    for rows,wanted in ((c,problem['wanted_c']),(g,problem['wanted_g'])):
        if tuple(x+y for x,y in zip(pacf(rows[0]),pacf(rows[1])))!=wanted:
            raise RuntimeError('joint SAT witness failed independent integer norm audit')
    if (any(sorted(w)!=sorted(r) for w,r in zip(c,problem['references']))
            or tuple(w.count(-1) for w in g)!=problem['negative_g']
            or (problem['fixed_supports'] is not None and tuple(support_mask(w) for w in c)!=problem['fixed_supports'])
            or (problem['fixed_c_a'] is not None and c[0]!=problem['fixed_c_a'])):
        raise RuntimeError('joint SAT witness failed contents/fixed-side audit')
    words=reconstruct_odd_pair(c,g,problem['target'])
    return dict(common,status='SAT_PQCP',words=words,coarse_words=c,difference_words=g,verified=True,reason=None)
