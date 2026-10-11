from itertools import product

import z3

from solver.global_pqcp_sat import content_cases,normalized_cases,build_global_solver,canonicalize_pair
from solver.verifier import verify_pqcp


def test_l86_has_exactly_the_three_derived_canonical_content_cases():
    cases=normalized_cases(43)
    assert len(cases)==3
    assert {tuple(c['half_sums']) for c in cases}=={(7,-1,11,1),(9,-3,9,3),(9,-1,9,1)}


def test_every_l6_solution_normalizes_into_an_encoded_case_without_loss():
    rows=tuple(product((0,1),repeat=6));models={}
    for a,b in product(rows,repeat=2):
        if verify_pqcp(a,b).is_valid:
            case,words=canonicalize_pair((a,b));key=case['k'],case['eta'],case['negative_half_counts']
            models.setdefault(key,set()).add(words)
    assert models
    for case in normalized_cases(3):
        solver,variables=build_global_solver(case);key=case['k'],case['eta'],case['negative_half_counts']
        for words in models.get(key,()):
            assumptions=[v if bit else z3.Not(v) for vs,row in zip(variables,words) for v,bit in zip(vs,row)]
            assert solver.check(*assumptions)==z3.sat


def test_all_encoded_small_sat_models_are_verified_and_cover_the_exact_canonical_set():
    rows=tuple(product((0,1),repeat=6));expected={}
    for a,b in product(rows,repeat=2):
        if verify_pqcp(a,b).is_valid:
            case,words=canonicalize_pair((a,b));expected.setdefault((case['k'],case['eta']),set()).add(words)
    for case in normalized_cases(3):
        s,variables=build_global_solver(case);found=set()
        while s.check()==z3.sat:
            m=s.model();words=tuple(tuple(int(z3.is_true(m.eval(v,model_completion=True))) for v in row) for row in variables)
            assert verify_pqcp(*words).is_valid
            found.add(words)
            s.add(z3.Or([v!=bool(bit) for vs,row in zip(variables,words) for v,bit in zip(vs,row)]))
        # Multiple rotations can satisfy the first two bits; diagnostic
        # normalization chooses the first such rotation deterministically.
        assert {canonicalize_pair(w)[1] for w in found}==expected[case['k'],case['eta']]


def test_timeout_does_not_enter_the_formula_as_an_exclusion():
    case=normalized_cases(3)[0];s,variables=build_global_solver(case,character_pruning=False)
    s.set(timeout=1);status=s.check()
    assert status in (z3.sat,z3.unknown)
    s.set(timeout=1000)
    assert s.check()==z3.sat and verify_pqcp(*tuple(tuple(int(z3.is_true(s.model().eval(v,model_completion=True))) for v in row) for row in variables)).is_valid


def test_intersection_and_xor_encodings_have_identical_complete_small_models():
    for case in normalized_cases(3):
        results=[]
        for encoding in ('xor','intersection'):
            s,variables=build_global_solver(case,encoding=encoding);models=set()
            while s.check()==z3.sat:
                m=s.model();words=tuple(tuple(int(z3.is_true(m.eval(v,model_completion=True))) for v in row) for row in variables)
                models.add(words);s.add(z3.Or([v!=bool(bit) for vs,row in zip(variables,words) for v,bit in zip(vs,row)]))
            results.append(models)
        assert results[0] and results[0]==results[1]
