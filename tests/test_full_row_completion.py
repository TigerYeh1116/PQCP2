from itertools import product

from solver.cyclic_support import cyclic_norm_family,iter_family,rotation_representatives
from solver.cyclic_support_native import all_weighted_norm_supports
from solver.full_row_completion import (crt_trit_rows,reconstruct_crt_binary,complete_fixed_binary_row_target,
                                       signed_norm_psd_obstruction)
from solver.structured_energy import integer_periodic_autocorrelation_profile as pacf
from solver.verifier import verify_pqcp


def test_crt_binary_trit_bijection_and_every_original_pacf_coordinate():
    for d in (3,5):
        for word in product((0,1),repeat=2*d):
            c,g=crt_trit_rows(word);assert reconstruct_crt_binary(c,g)==word
            nc,ng=pacf(c),pacf(g);original=pacf(tuple(1-2*v for v in word))
            assert all(original[2*u]==2*(nc[u]+ng[u]) and original[(2*u+d)%(2*d)]==2*(nc[u]-ng[u]) for u in range(d))


def test_all_generic_weighted_norm_preimages_match_python_entire_crt_product():
    for d in (3,5):
        for independent in product((0,1),repeat=d//2+1):
            desired=independent[0]+sum(independent[u]*((1<<u)|(1<<(d-u))) for u in range(1,d//2+1))
            for quotient in (False,True):
                family=cyclic_norm_family(desired,d)
                if quotient:family=rotation_representatives(family)
                for weight in range(d+1):
                    expected=tuple(m for m in iter_family(family) if bin(m).count('1')==weight)
                    out=all_weighted_norm_supports(family,weight)
                    assert out['complete'] and out['visited']==family.size and out['masks']==expected


def test_full_row_first_witness_or_exhaustion_matches_entire_literal_l6_space():
    rows=tuple(product((0,1),repeat=6));cases=positive=negative=0
    for a in rows:
        expected={}
        for b in rows:
            check=verify_pqcp(a,b)
            if check.is_valid:expected.setdefault((check.nonzero_shifts[0],check.profile[check.nonzero_shifts[0]]//4),set()).add(b)
        for k,eta in product((1,2),(-1,1)):
            for pruning in (False,True):
                out=complete_fixed_binary_row_target(a,k=k,eta=eta,certified_psd=pruning,prime_norm=pruning)
                bounded=complete_fixed_binary_row_target(a,k=k,eta=eta,certified_psd=pruning,prime_norm=pruning,support_bounds=True)
                assert (out['status'],out['complete'],out['words'])==(bounded['status'],bounded['complete'],bounded['words'])
                lifted=complete_fixed_binary_row_target(a,k=k,eta=eta,support_bounds=True,two_adic=True)
                assert bool(lifted['words'])==bool(expected.get((k,eta),()))
                if lifted['words']:assert lifted['words'][1] in expected[k,eta]
                else:assert lifted['complete']
                assert bool(out['words'])==bool(expected.get((k,eta),()))
                if out['words']:
                    assert out['words'][0]==a and out['words'][1] in expected[k,eta]
                    assert verify_pqcp(*out['words']).is_valid and not out['complete'];positive+=1
                else:assert out['complete'] and out['status']=='UNSAT_FIXED_FULL_A';negative+=1
                cases+=1
    assert cases==512 and positive and negative


def test_true_signed_trit_norms_keep_zero_boundary_and_negative_coefficients():
    negative=False
    for d in (3,5):
        for row in product((-1,0,1),repeat=d):
            profile=tuple(pacf(row));negative|=min(profile)<0
            assert signed_norm_psd_obstruction(profile) is None
    assert negative


def test_signed_support_bounds_keep_every_literal_binary_trit_positive_control():
    from solver.cyclic_support_native import signed_complement_supports
    for d in (3,5):
        for word in product((0,1),repeat=2*d):
            c,g=crt_trit_rows(word);profiles=tuple(pacf(c)),tuple(pacf(g))
            desired=sum((v%2)<<i for i,v in enumerate(profiles[0]))
            family=cyclic_norm_family(desired,d)
            out=signed_complement_supports(family,profiles,(c.count(-1),g.count(-1)))
            support=sum((v!=0)<<i for i,v in enumerate(c))
            assert out['complete'] and support in out['masks']


def test_support_dihedral_key_is_exact_and_does_not_complement_or_decimate():
    from solver.cyclic_support_native import dihedral_support_key
    for d in (3,5,11):
        all_bits=(1<<d)-1
        for mask in range(1<<d):
            reflected=sum(((mask>>i)&1)<<(-i%d) for i in range(d))
            expected=min(((v<<u)|(v>>(d-u)))&all_bits if u else v
                         for v in (mask,reflected) for u in range(d))
            assert dihedral_support_key(mask,d)==expected
        assert dihedral_support_key(all_bits,d)==all_bits
