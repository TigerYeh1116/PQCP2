"""Literal necessary affine spaces and integer targets validate native construction."""
from random import Random

import pytest

from solver.structured_energy import integer_periodic_autocorrelation_profile as pacf
from solver.ternary_system_native import prepare_native_system, run_native_system


def parity(value):
    return bin(value).count('1') & 1


def plan_words(plan, free, pattern=0):
    signs = list(plan['base'])
    for bit in range(plan['meta'][4]):
        if free >> bit & 1:
            for side in (0, 1):
                signs[side] ^= plan['toggles'][2*bit+side]
    for bit in range(plan['meta'][5]):
        if pattern >> bit & 1:
            for side in (0, 1):
                signs[side] ^= plan['patterns'][2*bit+side]
    return tuple(tuple(0 if not mask >> i & 1 else -1 if signs[s] >> i & 1 else 1
                       for i in range(plan['d'])) for s, mask in enumerate(plan['masks']))


@pytest.mark.parametrize('d', [3, 5, 7])
def test_native_raw_and_symbolic_fibres_equal_literal_mod4_words(d):
    rng = Random(734+d)
    for case in range(30):
        positions = rng.sample(range(2*d), d)
        masks = tuple(sum(1 << i for i in range(d) if s*d+i in positions) for s in (0, 1))
        variables = [(s, i) for s in (0, 1) for i in range(d) if masks[s] >> i & 1]
        witness = [0, 0]
        for s, i in variables:
            witness[s] |= rng.randrange(2) << i
        words = tuple(tuple(0 if not mask >> i & 1 else -1 if witness[s] >> i & 1 else 1
                            for i in range(d)) for s, mask in enumerate(masks))
        wanted = tuple(x+y for x, y in zip(pacf(words[0]), pacf(words[1])))
        negative = tuple(w.count(-1) for w in words)
        literal = set()
        actual = set()
        for bits in range(1 << d):
            candidate = [[0]*d for _ in range(2)]
            for col, (s, i) in enumerate(variables):
                candidate[s][i] = -1 if bits >> col & 1 else 1
            candidate = tuple(tuple(w) for w in candidate)
            profile = tuple(x+y for x, y in zip(pacf(candidate[0]), pacf(candidate[1])))
            if (all((v-w) % 4 == 0 for v, w in zip(profile, wanted))
                    and all((w.count(-1)-n) % 2 == 0 for w, n in zip(candidate, negative))):
                literal.add(candidate)
            if profile == wanted and tuple(w.count(-1) for w in candidate) == negative:
                actual.add(candidate)
        raw = prepare_native_system(masks, d, wanted, negative, conditioning=False)
        assert raw['consistent']
        represented = {plan_words(raw, f) for f in range(1 << raw['meta'][4])}
        assert represented == literal
        for side in (0, 1):
            plan = prepare_native_system(masks, d, wanted, negative, force_side=side)
            represented = set()
            for pattern in range(1 << plan['meta'][5]):
                expected = {w for w in literal if sum((w[side][i] == -1) << j for j, i in enumerate(
                    i for i in range(d) if masks[side] >> i & 1)) == pattern}
                if not all(parity(pattern & plan['checks'][j]) == plan['rhs'][j] for j in range(plan['meta'][6])):
                    assert not expected
                    continue
                produced = {plan_words(plan, f, pattern) for f in range(1 << plan['meta'][4])}
                assert produced == expected
                assert not represented & produced
                represented |= produced
            assert represented == literal
            out = run_native_system(plan, 1 << d)
            assert out['status'] == 'SAT_COMPRESSED' and out['words'] in actual
            projected = run_native_system(plan, 1 << d, conditioned_hybrid=True)
            assert projected['status'] == 'SAT_COMPRESSED' and projected['words'] in actual
            # A genuinely partial cap must keep old assignment order/status.
            if plan['estimated_assignments'] > 1:
                old = run_native_system(plan, 1)
                short = run_native_system(plan, 1, conditioned_hybrid=True)
                assert (short['status'], short['words'], short['assignments']) == (
                    old['status'], old['words'], old['assignments'])


def test_native_range_and_linear_contradictions_are_not_unknown():
    plan = prepare_native_system((7, 0), 3, (3, 0, 0), (1, 0))
    # A triangle cut count is even; target zero would require 1.5 cuts.
    assert not plan['consistent']
    assert run_native_system(plan, 1)['status'] == 'UNSAT_FIXED_SUPPORT'
    linear = prepare_native_system((7, 0), 3, (3, 1, 1), (1, 0))
    assert not linear['consistent']
    assert run_native_system(linear, 1)['status'] == 'UNSAT_FIXED_SUPPORT'


def test_projected_cardinality_closes_the_actual_l86_large_affine_exception():
    from solver.affine_weight import condition_sign_row
    from solver.ternary_signs_native import solve_sign_system
    from solver.ternary_support_search import support_sign_system
    masks = (67375149, 8795518140394)
    wanted = (43, 0, -1)+(0,)*38+(-1, 0)
    system = support_sign_system(masks, 43, wanted, (1, 16))
    assert system.consistent and len(system.free_columns) == 24
    projected = condition_sign_row(system, 0)
    assert projected.projection_rank == 2
    compatible = [bits for bits in range(128) if all(parity(bits & m) == r for m, r in projected.checks)]
    assert len(compatible) == 4 and {bin(bits).count('1') for bits in compatible} == {3, 5}
    new = prepare_native_system(masks, 43, wanted, (1, 16))
    assert not new['consistent'] and new['compatible_pattern_count'] == 0
    assert run_native_system(new, 1)['status'] == 'UNSAT_FIXED_SUPPORT'
    old = prepare_native_system(masks, 43, wanted, (1, 16), cardinality_pruning=False)
    assert old['consistent'] and run_native_system(old, 2**23)['status'] == 'UNKNOWN'
    python = solve_sign_system(system, masks, 43, wanted, (1, 16), 2**23)
    assert python['status'] == 'UNSAT_FIXED_SUPPORT' and python['assignments'] == 0


@pytest.mark.parametrize('small_side', [0, 1])
def test_l43_high_bits_match_python_raw_and_conditioned_expressions(small_side):
    from solver.affine_weight import condition_sign_row
    from solver.ternary_support_search import support_sign_system
    rng = Random(1786+small_side)
    occupied = rng.sample(range(86), 43)
    # Explicitly put an eight-entry selected row on either side, including
    # variable IDs above 31 when the selected row is side one.
    small = rng.sample(range(43), 8)
    large = rng.sample(range(43), 35)
    masks = [0, 0]
    masks[small_side] = sum(1 << i for i in small)
    masks[1-small_side] = sum(1 << i for i in large)
    negatives = [1, 1]
    negatives[1-small_side] = 16
    negative_masks = [0, 0]
    negative_masks[small_side] = 1 << small[0]
    negative_masks[1-small_side] = sum(1 << i for i in large[:16])
    words = tuple(tuple(0 if not mask >> i & 1 else -1 if negative_masks[s] >> i & 1 else 1
                        for i in range(43)) for s, mask in enumerate(masks))
    wanted = tuple(x+y for x, y in zip(pacf(words[0]), pacf(words[1])))
    python = support_sign_system(masks, 43, wanted, negatives)
    native = prepare_native_system(masks, 43, wanted, negatives, conditioning=False)
    assert native['consistent'] == python.consistent
    assert native['meta'][1] == python.rank
    assert native['meta'][2] == len(python.free_columns)
    for free in (0, 1, (1 << native['meta'][2])-1, rng.randrange(1 << native['meta'][2])):
        expected = [[0]*43 for _ in range(2)]
        for (s, i), (mask, constant) in zip(python.variables, python.expressions()):
            expected[s][i] = -1 if parity(mask & free) ^ constant else 1
        assert plan_words(native, free) == tuple(tuple(w) for w in expected)
    py = condition_sign_row(python, small_side)
    native = prepare_native_system(masks, 43, wanted, negatives, force_side=small_side)
    assert native['meta'][3] == py.projection_rank and native['meta'][4] == py.free_count
    for pattern in range(256):
        valid = all(parity(pattern & mask) == rhs for mask, rhs in py.checks)
        assert valid == all(parity(pattern & native['checks'][j]) == native['rhs'][j]
                            for j in range(native['meta'][6]))
        if not valid:
            continue
        for free in (0, 1, (1 << py.free_count)-1):
            expected = [[0]*43 for _ in range(2)]
            for (s, i), (gm, tm, constant) in zip(python.variables, py.expressions):
                expected[s][i] = -1 if parity(gm & free) ^ parity(tm & pattern) ^ constant else 1
            assert plan_words(native, free, pattern) == tuple(tuple(w) for w in expected)
