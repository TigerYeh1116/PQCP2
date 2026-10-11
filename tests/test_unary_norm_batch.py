import base64
from itertools import product
import struct

from solver.unary_norm_batch import complete_unary_norm_batch
from solver.structured_energy import integer_periodic_autocorrelation_profile as pacf


def test_entire_small_literal_sign_spaces_match_batch_partitions():
    for d in (3,5):
        rows=tuple(product((-1,0,1),repeat=d));models={}
        for row in rows:
            profile=tuple(pacf(row));negative=row.count(-1);mask=sum((v!=0)<<j for j,v in enumerate(row))
            models.setdefault((profile,negative),{}).setdefault(mask,set()).add(row)
        for (profile,negative),literal in models.items():
            masks=tuple(mask for mask in range(1<<d) if bin(mask).count('1')==profile[0])
            result=complete_unary_norm_batch(masks,d,profile,negative,2**20)
            assert result['complete_input_partition'] and result['deferred_supports']==0
            found={mask:word for mask,word,_ in result['candidates']}
            assert set(found)==set(literal)
            assert all(word in literal[mask] for mask,word in found.items())
            assert result['unsat_supports']+len(found)==len(masks)
            blob=base64.b64decode(result['certificate_base64']);prefix='<' if result['certificate_byteorder']=='little' else '>'
            decoded=list(struct.iter_unpack(prefix+'5Q',blob))
            assert [r[0] for r in decoded]==list(masks)
            assert sum((r[1]&255)==0 for r in decoded)==result['unsat_supports']


def test_budget_and_free_dimension_only_defer_never_falsely_exclude():
    d=11;row=tuple(-1 if j%3==0 else 1 for j in range(d));profile=tuple(pacf(row));mask=(1<<d)-1
    result=complete_unary_norm_batch((mask,),d,profile,row.count(-1),1)
    assert result['deferred_supports']==1 and result['candidates']==((mask,None,0),)
    exact=complete_unary_norm_batch((mask,),d,profile,row.count(-1),2**20)
    assert exact['sat_supports']==1 and exact['candidates'][0][1] is not None


def test_large_prime_true_norms_keep_every_true_support():
    for mask in ((1<<43)-1,0x5F1C37B5A21,0x32494D32295):
        mask&=(1<<43)-1
        row=tuple((-1 if j%3 else 1) if mask>>j&1 else 0 for j in range(43))
        out=complete_unary_norm_batch((mask,),43,pacf(row),row.count(-1),2**31)
        assert out['unsat_supports']==0 and len(out['candidates'])==1
        word=out['candidates'][0][1]
        if word is not None:assert tuple(pacf(word))==tuple(pacf(row))


def test_complete_coarse_pipeline_with_batch_keeps_all_small_full_models():
    from solver.coarse_row_completion import complete_fixed_coarse_row_target
    from solver.unary_norm_carrier_cache import UnaryNormCarrierCache
    from solver.verifier import verify_pqcp
    for d in (3,5):
        for c in product((-1,0,1),repeat=d):
            cache=UnaryNormCarrierCache()
            for k in range(1,d):
                for eta in (-1,1):
                    options=dict(k=k,eta=eta,support_bounds=True,dihedral_support=True,two_adic=True)
                    original=complete_fixed_coarse_row_target(c,**options)
                    batch=complete_fixed_coarse_row_target(c,unary_batch_free=12,norm_carrier_cache=cache,**options)
                    assert (batch['status'],batch['complete'])==(original['status'],original['complete'])
                    assert bool(batch['words'])==bool(original['words'])
                    if batch['words']:assert verify_pqcp(*batch['words']).is_valid
