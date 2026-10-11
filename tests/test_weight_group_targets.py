from dataclasses import replace

from solver.compressed_repair import _validated_reference
from solver.projection_group_targets import group_projection_targets
from solver.target_profiles import canonical_target_content_profiles
from solver.torch_compressed_search import paired_contents,search_paired_contents
from solver.weight_group_targets import group_weight_targets


def test_all_53_l86_cases_survive_exact_row_exchange_and_parity_grouping():
    choices=tuple((t,c) for t in canonical_target_content_profiles(86) for c in search_paired_contents(t))
    projected=group_projection_targets(choices);groups=group_weight_targets(projected)
    assert len(projected)==27 and len(groups)==14
    seen=set();targets=0
    for g in groups:
        assert g['weights'][0]<g['weights'][1] and sum(g['weights'])==43
        for entry in g['c_groups']:
            assert entry['projection_index'] not in seen;seen.add(entry['projection_index'])
            original=projected[entry['projection_index']]
            refs=tuple(tuple([-1]*n+[0]*z+[1]*p) for n,z,p in entry['contents'])
            for old,new in zip(original['targets'],entry['targets']):
                expected=replace(old,a_even_ones=old.b_even_ones,a_odd_ones=old.b_odd_ones,
                    b_even_ones=old.a_even_ones,b_odd_ones=old.a_odd_ones) if entry['row_swapped'] else old
                assert new==expected
                _validated_reference(*refs,new);targets+=1
    assert seen==set(range(27)) and targets==53


def test_every_small_target_and_content_keeps_inverse_provenance():
    for length in (6,10):
        choices=tuple((t,c) for t in canonical_target_content_profiles(length) for c in paired_contents(t))
        projected=group_projection_targets(choices);groups=group_weight_targets(projected)
        assert sum(len(e['targets']) for g in groups for e in g['c_groups'])==sum(len(g['targets']) for g in projected)
        assert {e['projection_index'] for g in groups for e in g['c_groups']}==set(range(len(projected)))
