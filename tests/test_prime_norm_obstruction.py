from itertools import product
from random import Random

from solver.prime_norm_obstruction import (real_roots_mod_p,lifted_tables,prime_norm_obstruction,
                                           verify_prime_norm_certificate)
from solver.structured_energy import integer_periodic_autocorrelation_profile as pacf


def test_complete_real_root_census_against_every_mod_p_value():
    from solver.prime_norm_obstruction import _psi_and_derivative
    for d,p in ((3,5),(5,19),(11,43),(43,257)):
        expected=tuple(t for t in range(p) if not _psi_and_derivative(t,d//2,p)[0])
        assert real_roots_mod_p(d,p)==expected and len(expected)==d//2


def test_every_small_binary_and_ternary_integer_norm_survives():
    for d in (3,5):
        for row in product((-1,0,1),repeat=d):
            profile=pacf(row)
            assert prime_norm_obstruction(profile,limit=100,native=False) is None
            assert prime_norm_obstruction(profile,limit=100) is None
    for row in product((0,1),repeat=11):assert prime_norm_obstruction(pacf(row),limit=300) is None


def test_native_and_reference_certificates_agree_on_all_small_signed_symmetric_profiles():
    for d in (3,5):
        for half in product(range(-2,3),repeat=d//2+1):
            profile=tuple(half)+tuple(reversed(half[1:]))
            candidate=prime_norm_obstruction(profile,limit=100)
            reference=prime_norm_obstruction(profile,limit=100,native=False)
            assert candidate==reference
            if candidate:assert verify_prime_norm_certificate(candidate)


def test_l43_prime_tables_retain_integer_norms_and_certify_impossible_profiles():
    rng=Random(43);tables=lifted_tables(43,50000);assert len({p for p,t,c in tables})>50
    for _ in range(50):
        row=tuple(rng.choice((-1,0,1)) for i in range(43))
        assert prime_norm_obstruction(pacf(row)) is None
    found=[]
    for _ in range(100):
        half=(21,)+tuple(rng.randrange(7,13) for i in range(21));profile=half+tuple(reversed(half[1:]))
        certificate=prime_norm_obstruction(profile)
        assert certificate==prime_norm_obstruction(profile,native=False)
        if certificate:found.append(certificate)
    assert found and all(verify_prime_norm_certificate(c) for c in found)


def test_a_single_norm_obstruction_must_not_be_applied_to_two_unknown_rows():
    a=pacf((1,1,0));b=pacf((1,1,-1));combined=tuple(x+y for x,y in zip(a,b))
    assert combined==(5,0,0)
    assert prime_norm_obstruction(a,limit=5) is None and prime_norm_obstruction(b,limit=5) is None
    certificate=prime_norm_obstruction(combined,limit=5)
    assert certificate is not None and verify_prime_norm_certificate(certificate)


def test_l86_profile_passes_range_sum_psd_but_has_certified_single_norm_obstruction():
    from solver.certified_profile_psd import profile_psd_obstruction
    from solver.sds_norm_completion import complete_binary_intersection_profile
    half=(20,10,7,9,10,9,8,8,9,10,9,9,10,7,9,9,9,9,10,9,9,11)
    wanted=half+tuple(reversed(half[1:]))
    assert sum(wanted)==400 and profile_psd_obstruction(wanted) is None
    original=complete_binary_intersection_profile(wanted)
    candidate=complete_binary_intersection_profile(wanted,prime_norm=True)
    assert original['complete'] and candidate['complete'] and not original['masks'] and not candidate['masks']
    assert original['visited']>0 and candidate['visited']==0
    assert candidate['prime_norm_certificate']['prime']==859
    assert verify_prime_norm_certificate(candidate['prime_norm_certificate'])


def test_generic_completion_prime_filter_keeps_all_small_literal_models():
    from solver.sds_norm_completion import complete_binary_intersection_profile,intersection_profile
    for d in (3,5):
        for mask in range(1<<d):
            wanted=intersection_profile(mask,d)
            before=complete_binary_intersection_profile(wanted,rotation_quotient=False)
            after=complete_binary_intersection_profile(wanted,rotation_quotient=False,prime_norm=True,prime_limit=100)
            assert before['complete'] and after['complete'] and set(before['masks'])==set(after['masks'])
