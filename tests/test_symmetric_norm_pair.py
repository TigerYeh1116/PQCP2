from itertools import product

from solver.symmetric_norm_pair import symmetric_norm_support_pairs
from solver.structured_energy import integer_periodic_autocorrelation_profile as pacf


def test_signed_reflection_norm_mod4_depends_only_on_support():
    for d in (3,5,11):
        for half in product((-1,0,1),repeat=(d+1)//2):
            row=half+tuple(reversed(half[1:]));support=tuple(int(v!=0) for v in row)
            assert tuple(v%4 for v in pacf(row))==tuple(v%4 for v in pacf(support))


def test_all_small_symmetric_support_pairs_match_entire_literal_domain():
    for d in (3,5,11):
        rows=tuple(h+tuple(reversed(h[1:])) for h in product((0,1),repeat=(d+1)//2))
        for eta in (-1,1):
            target=(d,eta)+(0,)*(d-3)+(eta,);expected=set()
            for a,b in product(rows,repeat=2):
                if sum(a)+sum(b)==d and all((x+y-z)%4==0 for x,y,z in zip(pacf(a),pacf(b),target)):
                    expected.add(tuple(sum(v<<j for j,v in enumerate(r)) for r in (a,b)))
            result=symmetric_norm_support_pairs(d,eta)
            assert result['complete'] and set(result['support_pairs'])==expected
            assert result['excludes_entire_symmetric_model']==(not expected)


def test_l43_both_target_signs_have_complete_empty_weight_slice():
    for eta in (-1,1):
        result=symmetric_norm_support_pairs(43,eta)
        assert result['complete'] and result['excludes_entire_symmetric_model'] and not result['support_pairs']
        assert result['required_common_weight']==20
        for row in result['rows']:
            if not row['affine']['consistent']:
                check=row['affine']['contradiction_mask']
                assert all(bin(check&col).count('1')%2==0 for col in result['linear_columns'])
                assert bin(check&row['rhs']).count('1')%2==1
        if eta==-1:assert all(not r['affine']['consistent'] for r in result['rows'])
        else:
            assert [r['affine']['rank'] for r in result['rows']]==[17]*4
            candidates=[c['common_weight'] for r in result['rows'] for c in r['candidates']]
            assert sorted(candidates)==sorted([0,14,22,24,16,18,26,40]*2)
