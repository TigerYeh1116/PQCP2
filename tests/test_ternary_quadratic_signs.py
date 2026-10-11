"""The mod-8 quadratic filter is checked against direct integer correlations."""
import ctypes
from random import Random

import pytest

from solver.structured_energy import integer_periodic_autocorrelation_profile as pacf
from solver.ternary_system_native import _library, prepare_native_system, run_native_system


def words(plan, free):
    signs = list(plan['base'])
    for j in range(plan['meta'][2]):
        if free >> j & 1:
            signs[0] ^= plan['toggles'][2*j]
            signs[1] ^= plan['toggles'][2*j+1]
    return tuple(tuple(0 if not mask >> i & 1 else -1 if signs[s] >> i & 1 else 1
                       for i in range(plan['d'])) for s, mask in enumerate(plan['masks']))


def direct_syndrome(plan, pair):
    unsigned = tuple(tuple(int(mask >> i & 1) for i in range(plan['d'])) for mask in plan['masks'])
    terms = tuple(x+y for x, y in zip(pacf(unsigned[0]), pacf(unsigned[1])))
    actual = tuple(x+y for x, y in zip(pacf(pair[0]), pacf(pair[1])))
    result = 0
    for u in range(1, plan['d']//2+1):
        cuts = (terms[u]-actual[u])//2
        required = (terms[u]-plan['wanted'][u])//2
        assert (cuts-required) % 2 == 0
        result |= (((cuts >> 1) ^ (required >> 1)) & 1) << (u-1)
    for s in (0, 1):
        count = pair[s].count(-1)
        assert (count-plan['negative'][s]) % 2 == 0
        result |= (((count >> 1) ^ (plan['negative'][s] >> 1)) & 1) << (plan['d']//2+s)
    return result


def coefficients(plan):
    q = plan['meta'][2]
    u = ctypes.c_uint32
    constant, linear, quadratic = u(), (u*q)(), (u*(q*q))()
    status = _library().pqcp_quadratic_plan(plan['mask_array'], plan['base'], plan['toggles'], q,
        plan['d'], plan['wanted'], plan['negative'], ctypes.byref(constant), linear, quadratic)
    assert status == 0
    return constant.value, tuple(linear), tuple(quadratic)


def evaluate(coef, free):
    constant, linear, quadratic = coef
    q = len(linear)
    value = constant
    for i in range(q):
        if free >> i & 1:
            value ^= linear[i]
            for j in range(i+1, q):
                if free >> j & 1:
                    value ^= quadratic[i*q+j]
    return value


@pytest.mark.parametrize('d', [3, 5, 7, 9, 11])
def test_all_small_free_assignments_equal_integer_mod8_and_raw_solver(d):
    rng = Random(881+d)
    for case in range(12):
        occupied = rng.sample(range(2*d), d)
        masks = tuple(sum(1 << i for i in range(d) if s*d+i in occupied) for s in (0, 1))
        witness = tuple(tuple(0 if not mask >> i & 1 else rng.choice((-1, 1)) for i in range(d)) for mask in masks)
        wanted = tuple(x+y for x, y in zip(pacf(witness[0]), pacf(witness[1])))
        if case % 2:
            # Change one exact target by four without changing its GF2
            # space. Include UNSAT cases, not only constructed witnesses.
            unsigned = tuple(tuple(int(mask >> i & 1) for i in range(d)) for mask in masks)
            maximum = pacf(unsigned[0])[1]+pacf(unsigned[1])[1]
            alternatives = [v for v in (wanted[1]-4, wanted[1]+4) if abs(v) <= maximum]
            if alternatives:
                wanted = (wanted[0], alternatives[0])+wanted[2:-1]+(alternatives[0],)
        plan = prepare_native_system(masks, d, wanted, tuple(w.count(-1) for w in witness), conditioning=False)
        coef = coefficients(plan)
        feasible = set()
        for free in range(1 << plan['meta'][2]):
            pair = words(plan, free)
            assert evaluate(coef, free) == direct_syndrome(plan, pair)
            if (tuple(x+y for x, y in zip(pacf(pair[0]), pacf(pair[1]))) == wanted
                    and tuple(w.count(-1) for w in pair) == tuple(plan['negative'])):
                feasible.add(pair)
        for cap in (1, 3, 1 << d):
            old = run_native_system(plan, cap, quadratic=False)
            new = run_native_system(plan, cap, quadratic=True)
            assert (new['status'], new['words'], new['assignments']) == (old['status'], old['words'], old['assignments'])
            if cap == 1 << d:
                assert new['status'] == ('SAT_COMPRESSED' if feasible else 'UNSAT_FIXED_SUPPORT')
                if feasible:
                    assert new['words'] in feasible
            for tail in (1, 3, 5, 8):
                hybrid = run_native_system(plan, cap, quadratic=True, hybrid_tail=tail)
                if cap < 1 << plan['meta'][2]:
                    assert (hybrid['status'], hybrid['words'], hybrid['assignments']) == (
                        old['status'], old['words'], old['assignments'])
                    assert hybrid['prefix_branches'] == 0
                else:
                    assert hybrid['status'] == ('SAT_COMPRESSED' if feasible else 'UNSAT_FIXED_SUPPORT')
                    if feasible:
                        assert hybrid['words'] in feasible
                    else:
                        assert hybrid['assignments'] == 1 << plan['meta'][2]


def test_l43_quadratics_agree_with_independent_integer_evaluation_at_high_bits():
    masks = (6895337172962, 6144029134102)
    wanted = (43, 1)+(0,)*40+(1,)
    plan = prepare_native_system(masks, 43, wanted, (9, 8), conditioning=False)
    assert plan['meta'][2] == 21
    coef = coefficients(plan)
    rng = Random(86)
    for free in [0, (1 << 21)-1]+[1 << j for j in range(21)]+[rng.randrange(1 << 21) for _ in range(100)]:
        assert evaluate(coef, free) == direct_syndrome(plan, words(plan, free))
    old, new = run_native_system(plan, 2**23, quadratic=False), run_native_system(plan, 2**23, quadratic=True)
    assert old['status'] == new['status'] == 'UNSAT_FIXED_SUPPORT'
    assert old['assignments'] == new['assignments'] == 2**21
    assert new['quadratic_matches'] < old['count_matches']
    for tail in (3, 4, 5, 6):
        hybrid = run_native_system(plan, 2**23, quadratic=True, hybrid_tail=tail)
        assert hybrid['status'] == 'UNSAT_FIXED_SUPPORT' and hybrid['assignments'] == 2**21
        assert hybrid['quadratic_matches'] == new['quadratic_matches']
        assert hybrid['prefix_branches'] == 2**(21-tail)
        assert hybrid['projected_candidates'] < 2**21


def test_dependent_tail_columns_enumerate_all_nullspace_members():
    # Global sign reversal stays in this even-weight affine space, and its
    # projected column can be zero. It must still be enumerated.
    plan = prepare_native_system((15, 0), 5, (4, -1, -1, -1, -1), (2, 0), conditioning=False)
    assert plan['consistent']
    old = run_native_system(plan, 256, quadratic=True)
    hybrid = run_native_system(plan, 256, quadratic=True, hybrid_tail=8)
    assert hybrid['maximum_tail_nullity'] > 0
    assert hybrid['status'] == old['status']
    if hybrid['words'] is not None:
        assert tuple(x+y for x, y in zip(pacf(hybrid['words'][0]), pacf(hybrid['words'][1]))) == (4, -1, -1, -1, -1)
