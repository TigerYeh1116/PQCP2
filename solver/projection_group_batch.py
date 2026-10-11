"""Batch exact C/G frontiers without changing any group existence problem.

The invariant is C AND (G_1 OR ... OR G_t). MPS visits complete eligible raw
affine spaces only. Truncated, conditioned, unsupported and overflowed cases
use the original native solver. No unvisited/UNKNOWN branch closes a group.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from time import perf_counter

from .compressed_repair import _validated_reference
from .odd_lift_native import difference_sign_problem,reconstruct_odd_pair
from .structured_energy import integer_periodic_autocorrelation_profile as pacf
from .ternary_support_search import _audit_sign_words,support_mask
from .ternary_system_native import prepare_native_system,run_native_system


class _Group:
    def __init__(self,a,b,targets,masks,budget):
        self.started=perf_counter();self.targets=tuple(targets);self.masks=tuple(masks);self.budget=budget
        if not self.targets or len(set(self.targets))!=len(self.targets):raise ValueError('distinct compatible targets required')
        self.a,self.b,self.d,self.wanted=_validated_reference(a,b,self.targets[0])
        if (self.d>47 or len(self.masks)!=2 or any(type(m) is not int or not 0<=m<1<<self.d for m in self.masks)
                or tuple(bin(m).count('1') for m in self.masks)!=tuple(sum(v!=0 for v in w) for w in (self.a,self.b))):
            raise ValueError('literal supported group required')
        for target in self.targets[1:]:
            if _validated_reference(self.a,self.b,target)[3]!=self.wanted:
                raise ValueError('all targets must share exactly the same C sign problem')
        self.c_plan=prepare_native_system(self.masks,self.d,self.wanted,(self.a.count(-1),self.b.count(-1)))
        self.g_data=[difference_sign_problem(self.masks,t) for t in self.targets]
        self.g_plans=[prepare_native_system(m,self.d,w,n) for m,w,n in self.g_data]
        self.order=sorted(range(len(self.targets)),key=lambda i:(self.g_plans[i]['estimated_assignments'],i))
        self.c=None;self.gs={};self.position=0;self.out=None;self.successful_g=None
        self.stage='C_FIRST' if (not self.c_plan['consistent'] or
            self.c_plan['estimated_assignments']<=sum(p['estimated_assignments'] for p in self.g_plans)) else 'G'

    def frontier(self):
        if self.out is not None:return None
        return self.c_plan if self.stage.startswith('C_') else self.g_plans[self.order[self.position]]

    def finish(self,status,index=None):
        words=None if index is None else reconstruct_odd_pair(self.c['words'],self.gs[index]['words'],self.targets[index])
        self.out=dict(status=status,words=words,verified=words is not None,
            actual_target=asdict(self.targets[index]) if index is not None else None,
            attempts=dict(coarse=self.c,lifts=[dict(target=asdict(t),result=self.gs.get(i)) for i,t in enumerate(self.targets)]),
            elapsed=perf_counter()-self.started,
            scope='Exactly these folded-compatible full targets on this literal support/content pair; first verified model only.')

    def accept(self,result):
        if result['status'] not in ('SAT_COMPRESSED','UNSAT_FIXED_SUPPORT','UNKNOWN'):
            raise RuntimeError('invalid exact sign result status')
        if self.stage.startswith('C_'):
            self.c=result
            if result['words'] is not None:
                _audit_sign_words(result['words'],self.a,self.b,self.targets[0],self.wanted,self.masks)
            if result['status']=='UNSAT_FIXED_SUPPORT':
                self.finish('UNSAT_BINARY_SUPPORT_GROUP');return
            if self.stage=='C_AFTER_G_SAT':
                self.finish('SAT_PQCP',self.successful_g) if result['status']=='SAT_COMPRESSED' else self.finish('UNKNOWN')
                return
            if self.stage=='C_AFTER_G_UNKNOWN':self.finish('UNKNOWN');return
            self.stage='G';return
        index=self.order[self.position];self.gs[index]=result
        if result['words'] is not None:
            gm,gw,gn=self.g_data[index];rows=result['words']
            if (tuple(support_mask(w) for w in rows)!=gm or tuple(w.count(-1) for w in rows)!=gn
                    or tuple(x+y for x,y in zip(pacf(rows[0]),pacf(rows[1])))!=gw):
                raise RuntimeError('batch G witness failed independent all-shift integer audit')
        if result['status']=='SAT_COMPRESSED':
            self.successful_g=index
            if self.c is None:self.stage='C_AFTER_G_SAT'
            elif self.c['status']=='SAT_COMPRESSED':self.finish('SAT_PQCP',index)
            else:self.finish('UNKNOWN')
            return
        self.position+=1
        if self.position<len(self.order):return
        if all(self.gs[i]['status']=='UNSAT_FIXED_SUPPORT' for i in self.order):
            self.finish('UNSAT_BINARY_SUPPORT_GROUP')
        elif self.c is None:self.stage='C_AFTER_G_UNKNOWN'
        else:self.finish('UNKNOWN')


def complete_projection_group_batch(jobs,*,max_assignments=2**23,use_mps=True,native_workers=8,minimum_gpu_cases=8):
    """Jobs are (reference C_A, reference C_B, compatible targets, supports)."""
    jobs=tuple(jobs)
    if (not jobs or len(jobs)>1024 or type(max_assignments) is not int or not 1<=max_assignments<=2**31
            or type(use_mps) is not bool or type(native_workers) is not int or not 1<=native_workers<=16
            or type(minimum_gpu_cases) is not int or not 1<=minimum_gpu_cases<=1024):
        raise ValueError('bounded batch, assignment budget and worker configuration required')
    contexts=[_Group(*job,max_assignments) for job in jobs]
    gpu_available=False
    if use_mps:
        import torch
        gpu_available=torch.backends.mps.is_available()
    with ThreadPoolExecutor(max_workers=native_workers) as pool:
        while any(c.out is None for c in contexts):
            frontiers=[(c,c.frontier()) for c in contexts if c.out is None]
            eligible=[];native=[]
            for ctx,plan in frontiers:
                q=int(plan['meta'][2])
                if (gpu_available and plan['consistent'] and plan['selected']<0 and plan['d']==43
                        and 16<=q<=24 and (1<<q)<=max_assignments):eligible.append((ctx,plan))
                else:native.append((ctx,plan))
            if len(eligible)<minimum_gpu_cases:native+=eligible;eligible=[]
            # CPU and GPU work are independent. Join all original CPU
            # results before advancing logical frontiers to a new round.
            pending=[(ctx,pool.submit(run_native_system,plan,max_assignments)) for ctx,plan in native]
            if eligible:
                from .mps_mod8_prefix import enumerate_mod8_candidates,complete_mod8_models
                tick=perf_counter();plans=[plan for _,plan in eligible]
                candidates=enumerate_mod8_candidates(plans,tail=6,capacity=256,bit_sliced=True)
                solved=[complete_mod8_models(plan,model) for plan,model in zip(plans,candidates)]
                elapsed=perf_counter()-tick
                for (ctx,plan),models,result in zip(eligible,candidates,solved):
                    if not models['complete']:
                        pending.append((ctx,pool.submit(run_native_system,plan,max_assignments)));continue
                    result.update(rank=int(plan['meta'][1]),free_variables=int(plan['meta'][2]),
                        assignments=1<<int(plan['meta'][2]),construction_seconds=plan['construction_seconds'],
                        solve_seconds=elapsed/len(eligible),gpu_batch_seconds=elapsed,gpu_batch_size=len(eligible),
                        engine='MPS_EXACT_BIT_SLICED_TAIL',quadratic_matches=models['candidate_count'])
                    ctx.accept(result)
            for ctx,future in pending:ctx.accept(future.result())
    return tuple(c.out for c in contexts)
