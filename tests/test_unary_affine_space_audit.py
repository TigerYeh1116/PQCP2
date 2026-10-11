from itertools import product
import ctypes

from solver.coarse_row_completion import complete_fixed_coarse_row_target
from solver.structured_energy import integer_periodic_autocorrelation_profile as pacf
from solver.unary_norm_batch import complete_unary_norm_batch
from solver.unary_norm_batch import _library
from solver.ternary_system_native import prepare_native_system


def test_every_small_literal_norm_count_and_support_agrees_with_original_full_auditor():
    for d in (3,5):
        cases={(tuple(pacf(row)),row.count(-1)) for row in product((-1,0,1),repeat=d)}
        for profile,negative in cases:
            masks=tuple(m for m in range(1<<d) if bin(m).count('1')==profile[0])
            old=complete_unary_norm_batch(masks,d,profile,negative,2**20)
            for method in ('affine_audit','linear_certificate'):
                new=complete_unary_norm_batch(masks,d,profile,negative,2**20,**{method:True})
                for k in ('candidates','unsat_supports','sat_supports','deferred_supports','raw_candidates_inspected'):
                    assert old[k]==new[k]
                assert new['auditor_candidates_inspected']<=old['auditor_candidates_inspected']


def test_affine_audit_preserves_small_full_pqcp_decisions_and_models():
    for row in product((-1,0,1),repeat=3):
        for k,eta in product((1,2),(-1,1)):
            args=dict(k=k,eta=eta,unary_batch_free=12,dihedral_support=True)
            a=complete_fixed_coarse_row_target(row,**args)
            for method in ('unary_affine_audit','unary_linear_certificate'):
                b=complete_fixed_coarse_row_target(row,prebatch_dihedral=True,**{method:True},**args)
                assert (a['status'],a['complete'],a['words'])==(b['status'],b['complete'],b['words'])


def test_linear_certificate_accepts_independent_dense_plans_and_rejects_bad_bases():
    fn=_library().pqcp_verify_unary_affine_certificate
    u,i=ctypes.c_uint64,ctypes.c_int
    profile=tuple(pacf((1,-1,0,0,0)))
    plan=prepare_native_system((3,0),5,profile,(1,0))
    q=plan['meta'][2];base=int(plan['base'][0]);directions=tuple(int(plan['toggles'][2*j]) for j in range(q))
    assert q==1
    def verify(b,ts,seen):
        return fn(3,5,(i*5)(*profile),1,b,(u*max(1,len(ts)))(*ts),len(ts),seen)
    assert verify(base,directions,1<<q)==1
    assert verify(base^1,directions,1<<q)==0
    assert verify(base,(0,),1<<q)==0
    assert verify(base,(),1)==0
    assert verify(base,directions,1)==0
    assert verify(base,(directions[0]|(1<<5),),1<<q)==0
    # Every consistent dense plan from actual small signed rows supplies
    # the same necessary affine space, with independently computed rank.
    for row in product((-1,0,1),repeat=5):
        mask=sum((v!=0)<<j for j,v in enumerate(row));rho=tuple(pacf(row));negative=row.count(-1)
        p=prepare_native_system((mask,0),5,rho,(negative,0));free=p['meta'][2]
        assert p['consistent']
        ts=tuple(int(p['toggles'][2*j]) for j in range(free))
        assert fn(mask,5,(i*5)(*rho),negative,p['base'][0],(u*max(1,free))(*ts),free,1<<free)==1
