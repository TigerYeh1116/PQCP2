"""Complete modular zero sets, separate integer labels and cap fallback."""
import ctypes
from random import Random

import pytest

from solver.mod8_model_reference import all_mod8_indices
from solver.mps_mod8_prefix import quadratic_coefficients
from solver.paired_signs_native import paired_arguments, run_native_pair
from solver.structured_energy import integer_periodic_autocorrelation_profile as pacf
from solver.ternary_system_native import _library, prepare_native_system, run_native_system


def models(plans, capacity, tail=6):
    q = int(plans[0]['meta'][2])
    args = paired_arguments(plans, 1 << q, tail)
    assert args is not None
    fn = _library().pqcp_quadratic_paired_bit_tail_models
    p = ctypes.POINTER(ctypes.c_uint64)
    ip = ctypes.POINTER(ctypes.c_int)
    qp = ctypes.POINTER(ctypes.c_uint32)
    fn.argtypes = [p, p, p, ctypes.c_int, ctypes.c_int, ip, ip, ctypes.c_int,
                  ctypes.c_uint64, qp, p]
    fn.restype = ctypes.c_int
    output = (ctypes.c_uint32*(2*capacity))()
    count = (ctypes.c_uint64*2)()
    assert fn(*args, capacity, output, count) == 0
    return tuple(tuple(sorted(output[k*capacity:k*capacity+min(count[k], capacity)]))
                 for k in range(2)), tuple(count)


def words_at(plan, index):
    signs = list(plan['base'])
    for j in range(plan['meta'][2]):
        if index >> j & 1:
            for s in range(2):
                signs[s] ^= plan['toggles'][2*j+s]
    return tuple(tuple(0 if not plan['masks'][s] >> i & 1 else -1 if signs[s] >> i & 1 else 1
                       for i in range(plan['d'])) for s in range(2))


@pytest.mark.parametrize('d', (3, 5, 7, 9, 11))
def test_all_modular_models_and_all_literal_integer_solutions_for_both_labels(d):
    rng = Random(860020+d)
    for _ in range(8):
        active = set(rng.sample(range(2*d), d))
        masks = tuple(sum(1 << i for i in range(d) if s*d+i in active) for s in range(2))
        references = [tuple(tuple(rng.choice((-1, 1)) if m >> i & 1 else 0 for i in range(d))
                            for m in masks) for k in range(2)]
        plans = tuple(prepare_native_system(masks, d,
            tuple(x+y for x, y in zip(pacf(ref[0]), pacf(ref[1]))),
            tuple(w.count(-1) for w in ref), conditioning=False) for ref in references)
        q = int(plans[0]['meta'][2])
        expected_models = tuple(all_mod8_indices(quadratic_coefficients(p), capacity=1 << q)['indices']
                                for p in plans)
        for tail in (1, 3, 6):
            actual, counts = models(plans, 1 << q, tail)
            assert actual == expected_models
            assert counts == tuple(map(len, expected_models))
            result = run_native_pair(plans, 1 << q, tail=tail)
            assert result is not None
            for k, plan in enumerate(plans):
                valid = set()
                for index in range(1 << q):
                    rows = words_at(plan, index)
                    if (tuple(w.count(-1) for w in rows) == tuple(plan['negative']) and
                            tuple(x+y for x, y in zip(pacf(rows[0]), pacf(rows[1]))) == tuple(plan['wanted'])):
                        valid.add(rows)
                assert result[k]['status'] == ('SAT_COMPRESSED' if valid else 'UNSAT_FIXED_SUPPORT')
                assert result[k]['words'] in valid if valid else result[k]['words'] is None


def test_a_positive_label_never_closes_its_negative_companion_and_caps_fall_back():
    rng = Random(860022)
    d = 11
    for _ in range(100):
        masks = (sum(1 << i for i in rng.sample(range(d), 7)),
                 sum(1 << i for i in rng.sample(range(d), 4)))
        ref = tuple(tuple(rng.choice((-1, 1)) if m >> i & 1 else 0 for i in range(d)) for m in masks)
        wanted = tuple(x+y for x, y in zip(pacf(ref[0]), pacf(ref[1])))
        negative = tuple(w.count(-1) for w in ref)
        for u in range(1, d//2+1):
            terms = sum(sum(bool(m >> i & 1 and m >> ((i+u) % d) & 1) for i in range(d)) for m in masks)
            delta = 4 if wanted[u]+4 <= terms else -4 if wanted[u]-4 >= -terms else None
            if delta is None:
                continue
            bad = list(wanted)
            bad[u] += delta
            bad[d-u] += delta
            plans = tuple(prepare_native_system(masks, d, w, negative, conditioning=False) for w in (wanted, bad))
            assert all(p['consistent'] for p in plans)
            q = int(plans[0]['meta'][2])
            for pair in (plans, plans[::-1]):
                out = run_native_pair(pair, 1 << q)
                assert out is not None
                for k, p in enumerate(pair):
                    old = run_native_system(p, 1 << q)
                    assert out[k]['status'] == old['status']
                    if tuple(p['wanted']) == wanted:
                        assert out[k]['words'] is not None
                    else:
                        # Sum_u R(u) is fixed by the two negative counts;
                        # changing only a symmetric shift pair violates it.
                        assert out[k]['status'] == 'UNSAT_FIXED_SUPPORT'
                        assert out[k]['assignments'] == 1 << q
                if q:
                    assert run_native_pair(pair, (1 << q)-1) is None
            return
    pytest.fail('no bounded counterexample control found')


def test_zero_dimensions_exact_overflow_and_unsupported_basis():
    for masks, wanted, negative, expected in [((0, 0), (0, 0, 0), (0, 0), 1),
                                            ((15, 0), (4, -1, -1, -1, -1), (2, 0), 2)]:
        p = prepare_native_system(masks, len(wanted), wanted, negative, conditioning=False)
        actual, counts = models((p, p), 1)
        assert counts == (expected, expected)
        assert len(actual[0]) == len(actual[1]) == 1
        out = run_native_pair((p, p), 1 << p['meta'][2])
        assert out is not None and out[0]['status'] == out[1]['status']
        incompatible = dict(p, selected=0)
        assert run_native_pair((p, incompatible), 1 << p['meta'][2]) is None


@pytest.mark.parametrize('seed', (860030, 860031, 860032))
def test_large_arbitrary_norm_positive_controls_and_exact_modular_zero_sets(seed):
    d = 43
    rng = Random(seed)
    masks = tuple(sum(1 << i for i in rng.sample(range(d), w)) for w in (20, 23))
    references = [tuple(tuple(rng.choice((-1, 1)) if m >> i & 1 else 0 for i in range(d))
                        for m in masks) for _ in range(2)]
    plans = tuple(prepare_native_system(masks, d,
        tuple(x+y for x, y in zip(pacf(ref[0]), pacf(ref[1]))),
        tuple(w.count(-1) for w in ref), conditioning=False) for ref in references)
    q = int(plans[0]['meta'][2])
    expected = tuple(all_mod8_indices(quadratic_coefficients(p), capacity=1 << q)['indices'] for p in plans)
    actual, counts = models(plans, 1 << q)
    assert actual == expected and counts == tuple(map(len, expected))
    result = run_native_pair(plans, 1 << q)
    assert result is not None
    for k, p in enumerate(plans):
        old = run_native_system(p, 1 << q)
        assert result[k]['status'] == old['status'] == 'SAT_COMPRESSED'
        assert result[k]['words'] == old['words']
