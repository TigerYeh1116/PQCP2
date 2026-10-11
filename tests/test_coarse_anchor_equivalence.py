from itertools import product

from experiments.audit_coarse_anchor_equivalence import norm_support_key
from solver.coarse_row_completion import complete_fixed_coarse_row_target
from solver.structured_energy import integer_periodic_autocorrelation_profile as pacf


def test_norm_support_key_preserves_every_declared_row_gauge():
    for d in (3,5,11,43):
        c=tuple((-1,0,1)[(j*j+3*j+1)%3] for j in range(d));key=norm_support_key(c,common_units=True)
        for q in range(1,d):
            transformed=tuple(c[q*j%d] for j in range(d))
            for shift in (0,1,d//2):
                row=transformed[shift:]+transformed[:shift]
                assert norm_support_key(row,common_units=True)==key
                assert norm_support_key(tuple(-v for v in row),common_units=True)==key


def test_homometric_sign_rows_are_proof_equivalent_but_not_a_sat_class_dedup():
    # H Q versus H Q*, with disjoint 5-point blocks. Unsigned H is symmetric
    # up to shift, so supports are dihedral while signed H is asymmetric.
    h=(1,1,-1,1,-1);q=(0,9,18,29);rows=[]
    for sign in (1,-1):
        c=[0]*43
        for offset in q:
            for j,value in enumerate(h):c[(j+sign*offset)%43]=value
        rows.append(tuple(c))
    a,b=rows
    assert sum(a)==sum(b)==4 and sum(v!=0 for v in a)==20
    assert tuple(pacf(a))==tuple(pacf(b))
    assert norm_support_key(a,common_units=False)==norm_support_key(b,common_units=False)
    assert norm_support_key(a,common_units=True)==norm_support_key(b,common_units=True)
    literal_images=set()
    for reverse in (a,tuple(a[-j%43] for j in range(43))):
        for shift in range(43):
            row=reverse[shift:]+reverse[:shift];literal_images.add(row);literal_images.add(tuple(-v for v in row))
    assert b not in literal_images


def test_key_classes_have_identical_complete_existence_for_all_small_targets():
    for d in (3,5):
        classes={}
        for c in product((-1,0,1),repeat=d):
            outcomes=[]
            for eta in (-1,1):
                found=False
                for k in range(1,d):
                    result=complete_fixed_coarse_row_target(c,k=k,eta=eta,support_bounds=True)
                    assert not result['status'].startswith('UNKNOWN')
                    found|=bool(result['words'])
                outcomes.append(found)
            key=norm_support_key(c,common_units=True)
            if key in classes:assert classes[key]==outcomes
            else:classes[key]=outcomes
