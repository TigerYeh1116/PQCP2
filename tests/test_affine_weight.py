"""Literal full-space oracles for symbolic cardinality conditioning."""
from random import Random

import pytest

from solver.affine_weight import condition_sign_row
from solver.orientation_gf2 import OrientationSystem
from solver.ternary_weight_signs_native import enumerate_weight_signs


def parity(x):
    return bin(x).count('1') & 1


def system_from_rows(variables, equations):
    basis = {}
    for row, rhs in equations:
        while row:
            pivot = row.bit_length()-1
            if pivot not in basis:
                basis[pivot] = row, rhs
                break
            prior, value = basis[pivot]
            row ^= prior
            rhs ^= value
        assert row or not rhs
    return OrientationSystem(tuple(variables), (),
                             tuple((p, *basis[p]) for p in sorted(basis)), True)


def test_all_conditioned_spaces_equal_literal_full_affine_space():
    rng = Random(943)
    for n in range(1, 9):
        variables = tuple((i % 2, i//2) for i in range(n))
        for case in range(20):
            witness = rng.randrange(1 << n)
            rows = [(rng.randrange(1 << n), 0) for _ in range(case % (n+1))]
            rows = [(row, parity(row & witness)) for row, _ in rows]
            system = system_from_rows(variables, rows)
            literal = {v for v in range(1 << n) if all(parity(v & row) == rhs for row, rhs in rows)}
            for side in (0, 1):
                plan = condition_sign_row(system, side)
                projected_columns = [i for i, (s, _) in enumerate(variables) if s == side]
                produced = set()
                for pattern in range(1 << len(projected_columns)):
                    expected = {v for v in literal if sum(((v >> col) & 1) << j
                                for j, col in enumerate(projected_columns)) == pattern}
                    if not all(parity(pattern & mask) == rhs for mask, rhs in plan.checks):
                        assert not expected
                        continue
                    values = [sum((parity(g & gm) ^ parity(pattern & tm) ^ c) << i
                              for i, (gm, tm, c) in enumerate(plan.expressions))
                              for g in range(1 << plan.free_count)]
                    assert len(values) == len(set(values)) == 1 << plan.free_count
                    assert set(values) == expected
                    assert not produced & expected
                    produced |= expected
                assert produced == literal


@pytest.mark.parametrize('side', [0, 1])
@pytest.mark.parametrize('negative', [(0, 0), (0, 1), (1, 1), (1, 2), (2, 3), (3, 3)])
def test_conditioned_native_matches_every_literal_sign_pair(side, negative):
    # Deliberately no necessary equations: the native kernel independently
    # tests the original integer correlations on all cardinality-feasible signs.
    variables = tuple((s, i) for s in range(2) for i in range(3))
    system = OrientationSystem(variables, (), (), True)
    plan = condition_sign_row(system, side)
    all_words = [tuple(-1 if v >> i & 1 else 1 for i in range(3)) for v in range(8)]
    for target in (-6, -2, 0, 2, 6):
        expected = {(a, b) for a in all_words for b in all_words
            if (a.count(-1), b.count(-1)) == negative
            and sum(a[i]*a[(i+1) % 3]+b[i]*b[(i+1) % 3] for i in range(3)) == target}
        out = enumerate_weight_signs(system, plan, (7, 7), 3, (6, target, target), negative, 64)
        assert out['status'] == ('SAT_COMPRESSED' if expected else 'UNSAT_FIXED_SUPPORT')
        if expected:
            assert out['words'] in expected


def test_conditioned_native_cap_and_empty_weight_projection():
    variables = tuple((s, i) for s in range(2) for i in range(3))
    system = OrientationSystem(variables, (), (), True)
    plan = condition_sign_row(system, 0)
    out = enumerate_weight_signs(system, plan, (7, 7), 3, (6, 0, 0), (1, 1), 1)
    assert out['status'] == 'UNKNOWN' and out['assignments'] == 1
    # A0=A1=A2. Weight one has no affine preimage, so it can be ruled out
    # without testing any assignment, even with an assignment budget of one.
    constrained = system_from_rows(variables, [(3, 0), (6, 0)])
    plan = condition_sign_row(constrained, 0)
    out = enumerate_weight_signs(constrained, plan, (7, 7), 3, (6, 0, 0), (1, 1), 1)
    assert out['status'] == 'UNSAT_FIXED_SUPPORT'
    assert out['assignments'] == out['compatible_patterns'] == 0


def test_native_conditioned_affine_dependencies_match_literal_integer_oracle():
    rng = Random(7631)
    variables = tuple((s, i) for s in range(2) for i in range(3))
    for case in range(12):
        witness = rng.randrange(64)
        masks = [rng.randrange(64) for _ in range(case % 7)]
        rows = [(mask, parity(mask & witness)) for mask in masks]
        system = system_from_rows(variables, rows)
        literal = [v for v in range(64) if all(parity(v & mask) == rhs for mask, rhs in rows)]
        for side in (0, 1):
            plan = condition_sign_row(system, side)
            for na in range(4):
                for nb in range(4):
                    for target in (-6, -2, 0, 2, 6):
                        expected = set()
                        for v in literal:
                            words = tuple(tuple(-1 if v >> (3*s+i) & 1 else 1 for i in range(3))
                                          for s in range(2))
                            if ((words[0].count(-1), words[1].count(-1)) == (na, nb)
                                    and sum(w[i]*w[(i+1) % 3] for w in words for i in range(3)) == target):
                                expected.add(words)
                        out = enumerate_weight_signs(system, plan, (7, 7), 3,
                                                     (6, target, target), (na, nb), 64)
                        assert out['status'] == ('SAT_COMPRESSED' if expected else 'UNSAT_FIXED_SUPPORT')
                        if expected:
                            assert out['words'] in expected
