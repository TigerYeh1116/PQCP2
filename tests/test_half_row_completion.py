from collections import defaultdict
from itertools import product

import pytest

from solver.half_row_completion import complete_parity_half_row,complete_parity_half_rows
from solver.verifier import verify_pqcp


def _halves(words):return (words[0][::2],words[0][1::2],words[1][::2],words[1][1::2])


def test_every_non_solution_l6_center_and_free_half_matches_full_literal_oracle():
    rows=tuple(product((0,1),repeat=6));all_pairs=tuple(product(rows,repeat=2))
    valid={pair for pair in all_pairs if verify_pqcp(*pair).is_valid}
    indices=[defaultdict(set) for _ in range(4)]
    for pair in valid:
        halves=_halves(pair)
        for free in range(4):indices[free][tuple(h for i,h in enumerate(halves) if i!=free)].add(pair)
    hits=misses=content_changes=0
    for center in all_pairs:
        if center in valid:continue
        halves=_halves(center)
        for free in range(4):
            expected=indices[free].get(tuple(h for i,h in enumerate(halves) if i!=free),set())
            out=complete_parity_half_row(*center,free_component=free)
            assert out['complete'] and out['raw_half_assignments']==8
            actual={hit['words'] for hit in out['witnesses']}
            assert actual==expected
            trace=complete_parity_half_row(*center,free_component=free,trace_backend=True)
            assert trace['complete'] and {hit['words'] for hit in trace['witnesses']}==expected
            hits+=bool(actual);misses+=not actual
            content_changes+=any(sum(_halves(words)[free])!=sum(halves[free]) for words in actual)
    assert hits and misses and content_changes


def test_l86_physical_negative_odd_row_is_kept_and_support_phases_are_restored():
    # Original published PCS4 ingredients of sums 11/7/1/1; the B-odd row
    # is physically NEGATED. This exercises weight 22, not coded weight 21.
    from solver.interleaving import interleave_bits
    masks=(1687788056856,3946068457116,3493597975730,1089223681038)
    def row(m):return tuple(1-2*((m>>i)&1) for i in range(43))
    words=(interleave_bits(row(masks[0]),row(masks[2])),interleave_bits(row(masks[1]),row(masks[3]),odd_sign=-1))
    assert not verify_pqcp(*words).is_valid
    out=complete_parity_half_row(*words,free_component=3)
    assert out['complete'] and out['norm_visited']>0
    assert out['completed_support_orbits']>0 and out['restored_phase_candidates']>=43
    assert any(record['wanted'][0]==22 for record in out['profile_records'])
    for hit in out['witnesses']:assert verify_pqcp(*hit['words']).is_valid
    trace=complete_parity_half_row(*words,free_component=3,trace_backend=True)
    assert trace['complete']
    assert {hit['words'] for hit in trace['witnesses']}=={hit['words'] for hit in out['witnesses']}


def test_unsupported_lengths_and_saved_solutions_do_not_become_unsat_certificates():
    with pytest.raises(ValueError):complete_parity_half_row((0,)*68,(0,)*68)
    solution=((0,0,0,0,0,1),(0,0,0,0,1,1))
    assert verify_pqcp(*solution).is_valid
    with pytest.raises(ValueError):complete_parity_half_row(*solution)
    with pytest.raises(ValueError):complete_parity_half_rows(*solution)


def test_batch_preparation_keeps_each_literal_fiber_and_checks_options():
    center=((0,0,0,0,0,0),(0,0,0,0,1,1))
    assert not verify_pqcp(*center).is_valid
    for trace in (False,True):
        batch=complete_parity_half_rows(*center,trace_backend=trace)
        for free,out in enumerate(batch):
            single=complete_parity_half_row(*center,free_component=free,trace_backend=trace)
            assert {hit['words'] for hit in out['witnesses']}=={hit['words'] for hit in single['witnesses']}
            def profiles(result):
                return tuple((r['wanted'],r['targets'],
                              {k:v for k,v in r['result'].items() if k!='elapsed'})
                             for r in result['profile_records'])
            assert profiles(out)==profiles(single)
    for components in ((),(0,0),(4,),(True,)):
        with pytest.raises(ValueError):complete_parity_half_rows(*center,free_components=components)


def test_all_scalar_l86_cases_force_the_reported_parity_row_sum_shapes():
    # Independent finite integer proof over every possible signed parity sum.
    values=tuple(range(-43,44,2));pairs=defaultdict(list)
    for e,o in product(values,repeat=2):pairs[((e+o)**2,(e-o)**2)].append((e,o))
    found=set()
    for parity,eta in product((0,1),(-1,1)):
        dc=172+8*eta;alt=172+8*eta*(-1 if parity else 1)
        shapes=set()
        for e,o in product(values,repeat=2):
            for f,p in pairs.get((dc-(e+o)**2,alt-(e-o)**2),()):
                shapes.add(tuple(sorted((tuple(sorted((abs(e),abs(o)))),tuple(sorted((abs(f),abs(p))))))))
        expected=(((1,7),(1,11)),) if parity else ((((1,9),(1,9)),) if eta==-1 else (((3,9),(3,9)),))
        assert shapes==set(expected)
        found.add((parity,eta))
    assert len(found)==4
