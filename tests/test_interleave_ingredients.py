import random

from experiments.search_perfect43_basis import perfect43_bases
from experiments.search_perfect43_interleave import odd_vectors,packed_profile,sign_profile,signs
from solver.interleaving import odd_cross_vectors,periodic_sign_profile


def test_perfect43_generated_ingredients_have_all_integer_sidelobes_minus_one():
    basis = perfect43_bases()
    assert basis['complete'] and len(basis['rows'])==8
    for row in basis['rows']:
        assert sum(row['signs'])==1
        assert periodic_sign_profile(row['signs'])==(43,)+(-1,)*42


def test_packed_profiles_and_odd_offsets_match_literal_sign_oracles():
    rng = random.Random(4332)
    masks = [rng.getrandbits(43) for _ in range(24)]
    for mask in masks:
        literal = periodic_sign_profile(signs(mask))[1:22]
        assert sign_profile(mask)==literal
        assert packed_profile(mask)==sum(((p+1)//4)*256**i for i,p in enumerate(literal))
    for a,b in zip(masks[::2],masks[1::2]):
        assert odd_vectors(a,b)==odd_cross_vectors(signs(a),signs(b))
    # Balanced base-256 extraction uniquely recovers each four-row residual
    # coefficient, including signed coefficients and carries between bytes.
    for a,b,c,d in zip(masks[::4],masks[1::4],masks[2::4],masks[3::4]):
        residual = [((x+1)//4+(y+1)//4-(z+1)//4-(w+1)//4)
                    for x,y,z,w in zip(*(sign_profile(m) for m in (a,b,c,d)))]
        encoded = packed_profile(a)+packed_profile(b)-packed_profile(c)-packed_profile(d)
        for expected in residual:
            digit = (encoded+128)%256-128
            assert digit==expected
            encoded=(encoded-digit)//256
        assert encoded==0
