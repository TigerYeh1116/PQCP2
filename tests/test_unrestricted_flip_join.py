from itertools import combinations,product
import json
import random

import pytest

from solver.correlation import periodic_autocorrelation
from solver.target_profiles import pair_content
from solver.unrestricted_flip_join import exact_unrestricted_flip_repair
from solver.verifier import verify_pqcp


def test_all_l6_non_solution_centers_match_literal_radius_one_oracle():
    rows=tuple(product((0,1),repeat=6))
    valid=tuple((a,b) for a,b in product(rows,repeat=2) if verify_pqcp(a,b).is_valid)
    hits=misses=changed_contents=0
    for center in product(rows,repeat=2):
        if verify_pqcp(*center).is_valid:continue
        expected=any(all(sum(x!=y for x,y in zip(old,new))<=1 for old,new in zip(center,words))
                     for words in valid)
        result=exact_unrestricted_flip_repair(*center,radius=1)
        assert bool(result['witnesses'])==expected
        assert result['existence_decided']
        if expected:
            hits+=1
            changed_contents+=pair_content(*result['witnesses'][0]['words'])!=pair_content(*center)
        else:
            misses+=1
            assert result['complete'] and result['enumerated_per_row']==(7,7)
    assert hits and misses and changed_contents


def test_quadratic_flip_signature_matches_original_integer_pacf_including_midpoint():
    rng=random.Random(7887)
    for n in (6,14,68,86,94):
        row=tuple(rng.randrange(2) for _ in range(n))
        for r in range(5):
            subsets=tuple(combinations(range(n),r)) if n==6 else tuple(
                tuple(sorted(rng.sample(range(n),r))) for _ in range(10))
            for flipped in subsets:
                changed=tuple(v^int(i in flipped) for i,v in enumerate(row))
                for u in range(1,n//2+1):
                    predicted=sum(1^row[(i+u)%n]^row[(i-u)%n] for i in flipped)
                    predicted+=sum(int((j-i)%n in (u,n-u))*(2 if u==n//2 else 1)
                                   for i,j in combinations(flipped,2))
                    delta=periodic_autocorrelation(changed,u)-periodic_autocorrelation(row,u)
                    assert delta%4==0 and (delta//4)%2==predicted%2


def test_native_nonempty_large_word_repair_across_bit64():
    # Existing independently checked L68 pair serves only as a kernel control,
    # never as a new L86 discovery or search seed in performance experiments.
    a=tuple(map(int,'01100101100111010000000010010101000000011000101001101101111100001100'))
    b=tuple(map(int,'11110000010101000110101001011010111110011000010111000100110011010011'))
    assert verify_pqcp(a,b).is_valid
    center=(tuple(v^int(i==65) for i,v in enumerate(a)),tuple(v^int(i==67) for i,v in enumerate(b)))
    assert not verify_pqcp(*center).is_valid
    result=exact_unrestricted_flip_repair(*center,radius=1)
    assert result['witnesses'] and result['mod8_candidate_pairs']>0
    assert result['full_coordinate_checks']>0


def test_radius_four_complete_small_product_ball_and_l86_pruning():
    center=((0,)*6,(0,)*6)
    result=exact_unrestricted_flip_repair(*center,radius=4)
    assert result['witnesses'] and result['expected_configurations_per_row']==57
    result=exact_unrestricted_flip_repair(*center,radius=5)
    assert result['witnesses'] and result['expected_configurations_per_row']==63
    result=exact_unrestricted_flip_repair((0,)*86,(0,)*86,radius=1)
    assert result['complete'] and result['enumerated_per_row']==(87,87)
    assert result['indexed_configurations']==0


def test_archived_solution_cannot_be_used_as_repair_center():
    center=((0,0,0,0,0,1),(0,0,0,0,1,1))
    assert verify_pqcp(*center).is_valid
    with pytest.raises(ValueError):exact_unrestricted_flip_repair(*center,radius=1)


def test_log_selection_audits_literal_profile_and_recomputes_energy(tmp_path):
    from experiments.repair_unrestricted_flips import audited_log_centers
    from solver.correlation import full_correlation_profile
    from solver.structured_energy import target_profile
    rng=random.Random(6687)
    words=tuple(tuple(rng.randrange(2) for _ in range(86)) for _ in range(2))
    profile=tuple(full_correlation_profile(*words))
    row=dict(L=86,A=words[0],B=words[1],k=1,sign=-1,profile=profile,target_energy=0,verified=True)
    path=tmp_path/'torch.jsonl';path.write_text(json.dumps(row)+'\n'+json.dumps(row)+'\n')
    out=audited_log_centers(path,16)
    assert len(out['seeds'])==1 and out['audited_non_solution_classes']==1
    wanted=target_profile(86,1,-1)
    assert out['seeds'][0]['independent_full_energy']==sum(((x-y)//4)**2 for x,y in zip(profile[1:44],wanted[1:44]))
    assert out['seeds'][0]['independent_full_energy']>0
    row['profile']=list(profile);row['profile'][1]+=4
    path.write_text(json.dumps(row)+'\n')
    with pytest.raises(ValueError,match='independent integer audit'):audited_log_centers(path,16)


def test_l86_content_square_classification_is_complete_for_signed_row_sums():
    # Enumerate possible signed even DC/ALT sums directly; the code table is
    # justified by these full integer identities, not by fitted observations.
    for total,expected in ((164,{8,10}),(180,{6,12})):
        actual={abs(a) for a in range(-86,87,2) for b in range(-86,87,2) if a*a+b*b==total}
        assert actual==expected
        for a,b in product(range(-12,13,2),repeat=2):
            if a*a+b*b==total:
                assert {abs(a),abs(b)}==expected


def test_grouped_l86_join_keeps_all_checks_and_halves_target_lookups():
    # Literal non-solution center, independently generated from GS ingredients.
    # A previous full radius-4 run had nonempty mod-8 matches, so this is not
    # merely comparing two immediate empty-family returns.
    center=tuple(tuple(map(int,w)) for w in (
        '00010011100100011100010011111011100001010000010000010110111010110001001001000110110001',
        '00001110110000110110100110100000001001010101000000011101010111100111100011010011101101'))
    assert tuple(map(len,center))==(86,86)
    assert not verify_pqcp(*center).is_valid
    plain=exact_unrestricted_flip_repair(*center,radius=4,content_grouping=False)
    grouped=exact_unrestricted_flip_repair(*center,radius=4,content_grouping=True)
    assert plain['complete'] and grouped['complete']
    assert plain['mod8_candidate_pairs']>0
    assert grouped['enumerated_per_row']==plain['enumerated_per_row']
    assert grouped['indexed_configurations']==plain['indexed_configurations']
    assert 2*grouped['exact_signature_lookups']==plain['exact_signature_lookups']
    assert grouped['full_coordinate_checks']==plain['full_coordinate_checks']
    assert grouped['mod8_candidate_pairs']<=plain['mod8_candidate_pairs']
