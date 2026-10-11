from solver.impulse_skew_construction import (impulse_pair_certificate,iter_impulse_c_masks,
                                              required_skew_intersections)
from solver.interleaving import periodic_sign_profile
from solver.perfect_odd_kernel import interleave_from_complementary_trits
from solver.correlation import full_correlation_profile
from solver.structured_energy import integer_periodic_autocorrelation_profile as pacf,target_profile


def test_complete_c_cardinality_and_original_linear_pair_xor_equations():
    certificate=impulse_pair_certificate();masks=tuple(iter_impulse_c_masks(certificate))
    assert certificate['rank']==20 and len(masks)==len(set(masks))==122880
    opposite=set(certificate['opposite_pairs'])
    for mask in masks:
        assert bin(mask).count('1')==16 and mask&certificate['holes']==0 and mask&1==0
        assert all((((mask>>p)^(mask>>(43-p)))&1)==int(p in opposite) for p in range(2,22))


def test_arbitrary_skew_o_forces_all_odd86_targets_and_even_completion_equations():
    certificate=impulse_pair_certificate();masks=tuple(iter_impulse_c_masks(certificate))
    o=tuple(1-2*int(1<=i<=21) for i in range(43))
    assert sum(o)==1 and all(o[i]+o[-i]==0 for i in range(1,43))
    for negative in (masks[0],masks[1234],masks[-1]):
        c=tuple(0 if certificate['holes']>>i&1 else 1-2*((negative>>i)&1) for i in range(43))
        g=tuple(int(certificate['holes']>>i&1) for i in range(43))
        assert sum(c)==9
        words=interleave_from_complementary_trits(c,g,o)
        profile=tuple(full_correlation_profile(*words));desired=target_profile(86,41,1)
        assert all(profile[u]==desired[u] for u in range(1,86,2))
        wanted=required_skew_intersections(negative,certificate)
        expected_o=(43,)+tuple(-41+4*v for v in wanted[1:])
        pc,pg=pacf(c),pacf(g)
        assert all(pc[u]+pg[u]+expected_o[u]==0 for u in range(1,43))
        assert sum(wanted)==441
