from fractions import Fraction
from itertools import product
from math import cos,pi

from solver.certified_profile_psd import (SCALE,cosine_upper_grid,profile_psd_obstruction,
                                        rational_pi_bounds)
from solver.sds_norm_completion import complete_binary_intersection_profile,intersection_profile


def test_rational_machin_pi_and_outward_cosine_grid_against_high_precision_reference():
    from sympy import N,pi as symbolic_pi,cos as symbolic_cos,Rational
    lower,upper=rational_pi_bounds()
    value=N(symbolic_pi,90)
    assert N(Rational(lower.numerator,lower.denominator),90)<value<N(Rational(upper.numerator,upper.denominator),90)
    for d in (3,5,11,43):
        for phase,bound in enumerate(cosine_upper_grid(d)):
            actual=N(symbolic_cos(2*symbolic_pi*phase/d),90)
            assert N(Rational(bound,SCALE),90)>=actual
            assert N(Rational(bound,SCALE),90)-actual<N(Rational(1,10**8),90)


def test_every_literal_small_binary_profile_survives_exact_psd_certification():
    for d in (3,5,11):
        for mask in range(1<<d):
            assert profile_psd_obstruction(intersection_profile(mask,d)) is None
    # Include constant and empty supports with exact zero Fourier coefficients.
    for d in (3,5,11,43):
        for mask in (0,(1<<d)-1):assert profile_psd_obstruction(intersection_profile(mask,d)) is None


def test_negative_upper_bound_is_a_literal_integer_certificate_not_a_float_decision():
    wanted=(3,2,0,1,0,0,0,0,1,0,2)
    certificate=profile_psd_obstruction(wanted)
    assert certificate is not None
    numerator=3*SCALE+2*sum(wanted[u]*certificate['cosine_upper_numerators'][u-1] for u in range(1,6))
    assert numerator==certificate['upper_bound_numerator']<0
    fast=complete_binary_intersection_profile(wanted)
    rollback=complete_binary_intersection_profile(wanted,certified_psd=False)
    assert fast['complete'] and rollback['complete'] and fast['masks']==rollback['masks']==()
    assert fast['visited']==0 and rollback['visited']>0
    assert fast['psd_certificate']==certificate


def test_native_and_python_certificates_match_all_small_nonnegative_symmetric_profiles():
    for d in (3,5):
        for half in product(range(d+1),repeat=d//2+1):
            wanted=tuple(half)+tuple(reversed(half[1:]))
            assert profile_psd_obstruction(wanted)==profile_psd_obstruction(wanted,native=False)
    # General public inputs outside binary intersection ranges retain the
    # arbitrary-precision Python path, with the same mathematical certificate.
    wanted=(10**40,0,10**40,10**40,0)
    assert profile_psd_obstruction(wanted)==profile_psd_obstruction(wanted,native=False)


def test_affine_batch_matches_every_individual_small_certificate_and_signed_base():
    from solver.certified_profile_psd import sparse_profile_psd_obstructions
    for d in (3,5):
        for half in product(range(-1,d+2),repeat=d//2+1):
            base=tuple(half)+tuple(reversed(half[1:]));offsets=[];profiles=[]
            for tau in range(1,d//2+1):
                for eta in (-1,1):
                    wanted=list(base);wanted[tau]+=eta;wanted[-tau]+=eta
                    if all(0<=v<=d for v in wanted):offsets.append((tau,eta));profiles.append(tuple(wanted))
            if not offsets:continue
            actual=sparse_profile_psd_obstructions(base,offsets)
            assert actual==tuple(profile_psd_obstruction(p,native=False) for p in profiles)
    # Multiple native chunks and zero Fourier boundary: no strict negative
    # upper bound may reject a true norm, including empty/constant supports.
    for d in (11,43):
        for mask in (0,(1<<d)-1,1234&((1<<d)-1)):
            wanted=intersection_profile(mask,d);base=list(wanted);base[1]-=1;base[-1]-=1
            assert sparse_profile_psd_obstructions(base,[(1,1)]*130)==(None,)*130
