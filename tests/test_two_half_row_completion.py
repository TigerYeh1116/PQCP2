from collections import defaultdict
from itertools import product

import pytest

from solver.half_row_completion import complete_three_parity_rows
from solver.two_half_row_completion import complete_two_parity_halves,two_half_row_fibers
from solver.verifier import verify_pqcp


def halves(words):return (words[0][::2],words[0][1::2],words[1][::2],words[1][1::2])


def all_l6_pairs():
    rows=tuple(product((0,1),repeat=6))
    return tuple(product(rows,repeat=2))


def test_every_l6_literal_fixed_triple_against_entire_original_binary_space():
    valid=tuple(pair for pair in all_l6_pairs() if verify_pqcp(*pair).is_valid)
    small=tuple(product((0,1),repeat=3));empty_unknown_hits=0
    for free in range(4):
        expected=defaultdict(set)
        for pair in valid:expected[tuple(row for i,row in enumerate(halves(pair)) if i!=free)].add(pair)
        for fixed in product(small,repeat=3):
            trial=list(fixed);trial.insert(free,None)
            for trace in (False,True):
                out=complete_three_parity_rows(trial,free_component=free,trace_backend=trace)
                assert out['complete'] and out['input_kind']=='THREE_LITERAL_PARITY_ROWS'
                assert {hit['words'] for hit in out['witnesses']}==expected.get(fixed,set())
                empty_unknown_hits+=sum(halves(hit['words'])[free]==(0,0,0) for hit in out['witnesses'])
                gram=complete_three_parity_rows(trial,free_component=free,trace_backend=trace,joint_gram=True)
                assert gram['complete'] and {hit['words'] for hit in gram['witnesses']}==expected.get(fixed,set())
                from solver.joint_gram_pruning import verify_joint_gram_certificate
                for record in gram['profile_records']:
                    if 'joint_gram_result' in record['result']:
                        assert all(verify_joint_gram_certificate(c) for c in record['result']['joint_gram_result']['certificates'])
                finite=complete_three_parity_rows(trial,free_component=free,trace_backend=trace,finite_trace=True,joint_gram=True)
                assert finite['complete'] and {hit['words'] for hit in finite['witnesses']}==expected.get(fixed,set())
                from solver.finite_field_trace_pruning import verify_finite_trace_certificate
                for record in finite['profile_records']:
                    assert all(verify_finite_trace_certificate(c) for c in record['result'].get('finite_trace_result',{}).get('certificates',()))
                batch=complete_three_parity_rows(trial,free_component=free,trace_backend=trace,
                                                finite_trace=True,joint_gram=True,batched_psd=True)
                assert {hit['words'] for hit in batch['witnesses']}==expected.get(fixed,set())
                assert tuple((r['wanted'],r['targets'],r['result'].get('psd_certificate')) for r in batch['profile_records'])==tuple(
                    (r['wanted'],r['targets'],r['result'].get('psd_certificate')) for r in finite['profile_records'])
                affine=complete_three_parity_rows(trial,free_component=free,trace_backend=trace,
                                                  finite_trace=True,joint_gram=True,batched_psd=True,affine_finite=True)
                assert {hit['words'] for hit in affine['witnesses']}==expected.get(fixed,set())
                for record in affine['profile_records']:
                    assert all(verify_finite_trace_certificate(c) for c in record['result'].get('finite_trace_result',{}).get('certificates',()))
    assert empty_unknown_hits>0


def test_two_half_union_matches_every_literal_l6_solution_for_all_radii():
    pairs=all_l6_pairs();valid=tuple(pair for pair in pairs if verify_pqcp(*pair).is_valid)
    centers=(pairs[0],pairs[10],pairs[543],pairs[1445],pairs[2080],pairs[-1])
    nonempty=empty=changed_contents=0
    for center in centers:
        if center in valid:continue
        original=halves(center)
        for radius in range(4):
            expected=set()
            for pair in valid:
                candidate=halves(pair);distances=tuple(sum(x!=y for x,y in zip(a,b)) for a,b in zip(original,candidate))
                if any(distances[varied]<=radius and all(distances[i]==0 for i in range(4) if i not in (varied,free))
                       for varied in range(4) for free in range(4) if varied!=free):expected.add(pair)
            out=complete_two_parity_halves(*center,radius=radius)
            assert out['complete'] and {hit['words'] for hit in out['witnesses']}==expected
            trace=complete_two_parity_halves(*center,radius=radius,trace_backend=True)
            assert {hit['words'] for hit in trace['witnesses']}==expected
            nonempty+=bool(expected);empty+=not expected
            changed_contents+=any(tuple(map(sum,halves(pair)))!=tuple(map(sum,original)) for pair in expected)
    assert nonempty and empty and changed_contents


def test_partial_inputs_and_centers_are_not_silently_accepted():
    with pytest.raises(ValueError):complete_three_parity_rows(((0,)*3,)*4)
    with pytest.raises(ValueError):complete_three_parity_rows(((0,)*3,None,(0,)*3,None))
    with pytest.raises(ValueError):complete_three_parity_rows(((0,)*3,(0,)*5,(0,)*3,None))
    solution=((0,0,0,0,0,1),(0,0,0,0,1,1));assert verify_pqcp(*solution).is_valid
    with pytest.raises(ValueError):list(two_half_row_fibers(*solution))
    with pytest.raises(ValueError):complete_two_parity_halves((0,)*6,(0,)*6,radius=True)
