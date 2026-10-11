from itertools import product
import random

import pytest

from experiments.search_periodic82_boundary import periodic82_parents,rotation_masks
from solver.boundary_tail_search import search_boundary_tails,search_boundary_tail_masks
from solver.correlation import full_correlation_profile
from solver.verifier import verify_pqcp


@pytest.mark.parametrize('tail',(1,2))
def test_every_l6_prefix_pair_and_tail_against_the_full_literal_binary_oracle(tail):
    length=6
    rows=tuple(product((0,1),repeat=length-tail))
    assignments=tuple(product((0,1),repeat=2*tail))
    saw_sat=saw_nullspace=False
    for a in rows:
        for b in rows:
            expected=set()
            for bits in assignments:
                words=(a+bits[:tail],b+bits[tail:])
                if verify_pqcp(*words).is_valid:expected.add(words)
            result=search_boundary_tails((a,b),length)
            assert result['complete'] and result['covered_tail_assignments']==2**(2*tail)
            actual={hit['words'] for hit in result['witnesses']}
            assert actual==expected
            saw_sat |= bool(actual)
            saw_nullspace |= result['nullity']>0
    assert saw_sat and saw_nullspace


@pytest.mark.parametrize('length',(66,86,94))
def test_native_projection_crossing_bit64_against_all_256_literal_tails(length):
    rng=random.Random(222+length)
    prefixes=tuple(tuple(rng.randrange(2) for _ in range(length-4)) for side in range(2))
    expected=set()
    eligible=0
    for bits in product((0,1),repeat=8):
        words=(prefixes[0]+bits[:4],prefixes[1]+bits[4:])
        profile=tuple(full_correlation_profile(*words))
        outer=tuple((profile[u]//4)%2 for u in range(4,length//2+1))
        # This independent test uses actual FULL length PACFs to classify
        # projected tail assignments. The midpoint must remain zero mod 8.
        eligible += int(not outer[-1] and sum(outer)<=1)
        if verify_pqcp(*words).is_valid:expected.add(words)
    out=search_boundary_tails(prefixes,length)
    assert out['surviving_tail_assignments']==eligible
    assert {hit['words'] for hit in out['witnesses']}==expected


def test_printed_periodic82_sources_and_bit_rotations_are_literal_and_verified():
    for row in periodic82_parents():
        assert row['profile']==(164,)+(0,)*81
        for word,mask in zip(row['words'],row['masks']):
            for shift,changed in enumerate(rotation_masks(mask)):
                literal=tuple((changed >> i)&1 for i in range(82))
                assert literal==tuple(word[(i-shift)%82] for i in range(82))


def test_prefix_reversal_is_a_full_word_symmetry_when_all_tails_are_retained():
    rng=random.Random(987)
    for parent in periodic82_parents():
        original=tuple(tuple(w)+tuple(rng.randrange(2) for _ in range(4)) for w in parent['words'])
        transformed=tuple(w[:82][::-1]+w[82:][::-1] for w in original)
        assert transformed==tuple(w[::-1][4:]+w[::-1][:4] for w in original)
        assert tuple(full_correlation_profile(*original))==tuple(full_correlation_profile(*transformed))


def test_invalid_packed_prefixes_cannot_be_silently_truncated():
    with pytest.raises(ValueError):search_boundary_tail_masks((1 << 82,0),86,4)
    with pytest.raises(ValueError):search_boundary_tail_masks((0,0),86,5)


def test_cyclic_window_repair_matches_the_full_unrestricted_literal_neighborhood():
    from solver.boundary_tail_repair import exact_cyclic_window_repair
    from solver.target_profiles import pair_content
    center=((0,0,0,0,1,1),(0,0,0,1,1,1))
    assert not verify_pqcp(*center).is_valid
    expected=set()
    for sa in range(6):
        for sb in range(6):
            for assignment in product((0,1),repeat=4):
                words=[list(w) for w in center]
                for side,s in enumerate((sa,sb)):
                    for j in range(2):words[side][(s+4+j)%6]=assignment[2*side+j]
                pair=tuple(map(tuple,words))
                if verify_pqcp(*pair).is_valid:expected.add(pair)
    result=exact_cyclic_window_repair(*center,block_size=2)
    assert result['complete'] and result['queries']==36 and result['covered_assignments']==36*16
    assert {hit['words'] for hit in result['witnesses']}==expected
    assert any(pair_content(*words)!=pair_content(*center) for words in expected)


def test_center_quotient_preserves_windows_and_does_not_use_unit_decimation():
    from solver.boundary_tail_repair import cyclic_window_center_key
    from solver.solution_equivalence import pqcp_equivalence_key
    rng=random.Random(98334)
    original=tuple(tuple(rng.randrange(2) for _ in range(14)) for side in range(2))
    changed=(original[0][3:]+original[0][:3],tuple(v^1 for v in original[1][::-1]))
    assert cyclic_window_center_key(*original)==cyclic_window_center_key(*changed[::-1])
    decimated=tuple(tuple(w[3*i%14] for i in range(14)) for w in original)
    assert pqcp_equivalence_key(*original)==pqcp_equivalence_key(*decimated)
    assert cyclic_window_center_key(*original)!=cyclic_window_center_key(*decimated)


def test_l86_nonempty_projection_across_bit64_has_exact_literal_candidate_count():
    # A non-solution whose FULL PACF is congruent to the k=1,+ target mod 8.
    # Constructed from an exact C word and an affine G base, never from an
    # archived full PQCP. This forces a nonempty large-length projection;
    # random prefixes alone would almost always have empty outer equations.
    center=tuple(tuple(map(int,w)) for w in (
        '01000101010001010101000101010001001001110001110101001100110101010101110000001100101010',
        '01101000111010011111110011001101100101001000000000111001010011101010000011001110110000'))
    assert not verify_pqcp(*center).is_valid
    expected=set();eligible=0
    prefixes=tuple(w[:82] for w in center)
    for bits in product((0,1),repeat=8):
        words=(prefixes[0]+bits[:4],prefixes[1]+bits[4:])
        profile=tuple(full_correlation_profile(*words))
        outer=tuple((profile[u]//4)%2 for u in range(4,44))
        eligible+=int(not outer[-1] and sum(outer)<=1)
        if verify_pqcp(*words).is_valid:expected.add(words)
    result=search_boundary_tails(prefixes,86)
    assert result['surviving_tail_assignments']==eligible and eligible>=1
    assert {hit['words'] for hit in result['witnesses']}==expected
