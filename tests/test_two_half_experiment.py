from itertools import product

from experiments.complete_two_l86_halves import packed_profile,select_json_centers
from solver.correlation import full_correlation_profile
from solver.interleaving import interleave_bits
from solver.structured_energy import full_error_energy
from solver.two_half_row_completion import complete_two_parity_halves
from solver.verifier import verify_pqcp


def test_packed_profile_matches_original_periodic_equations_for_entire_l6_space():
    rows=tuple(product((0,1),repeat=6))
    for words in product(rows,repeat=2):
        profile=tuple(sum(1 if row[i]==row[(i+u)%6] else -1 for row in words for i in range(6)) for u in range(6))
        assert packed_profile(words)==profile


def test_json_center_selection_recomputes_energy_and_ignores_stale_source_scores():
    masks=(1687788056856,3946068457116,3493597975730,1089223681038)
    def signs(mask):return tuple(1-2*((mask>>i)&1) for i in range(43))
    words=(interleave_bits(signs(masks[0]),signs(masks[2])),interleave_bits(signs(masks[1]),signs(masks[3]),odd_sign=-1))
    inputs=dict(seeds=[dict(words=((0,)*86,(0,)*86),target_energy=-100),dict(words=words,target_energy=999999)])
    selected=select_json_centers(inputs,1)
    assert len(selected['seeds'])==1 and selected['seeds'][0]['words']==words
    row=selected['seeds'][0];profile=tuple(full_correlation_profile(*words))
    best=min(full_error_energy(profile,k,eta)//16 for k in range(1,43) for eta in (-1,1))
    assert row['profile']==profile and row['independent_full_energy']==best and not verify_pqcp(*words).is_valid


def test_all_ordered_roles_keep_full_fiber_union_under_parity_changing_equivalence():
    center=((0,0,0,0,0,1),(0,0,0,0,0,0));assert not verify_pqcp(*center).is_valid
    out=complete_two_parity_halves(*center,radius=1)
    expected={hit['words'] for hit in out['witnesses']};assert expected
    transformations=(lambda a,b:(a[1:]+a[:1],b),lambda a,b:(b,a),
                     lambda a,b:(tuple(a[-i%6] for i in range(6)),tuple(b[-i%6] for i in range(6))))
    for transform in transformations:
        changed=transform(*center);candidate=complete_two_parity_halves(*changed,radius=1,joint_gram=True)
        assert {hit['words'] for hit in candidate['witnesses']}=={transform(*words) for words in expected}
