"""Literal support enumeration proves the joint-slice minimum and tie scope."""
from threading import Event

import pytest

from solver.cyclic_joint_search import joint_nearest_supports, repair_joint_cyclic_supports
from solver.ternary_support_search import support_signature


@pytest.mark.parametrize('d', [3, 5])
def test_joint_minimum_and_literal_ties_match_full_oracle(d):
    for a in range(1 << d):
        for b in range(0, 1 << d, 1 if d == 3 else 5):
            if bin(a).count('1')+bin(b).count('1') != d:
                continue
            for k in range(1, d//2+1):
                feasible = set()
                for u in range(1 << d):
                    if bin(u).count('1') != bin(a).count('1'):
                        continue
                    for v in range(1 << d):
                        if bin(v).count('1') != bin(b).count('1'):
                            continue
                        du, dv = bin(u ^ a).count('1'), bin(v ^ b).count('1')
                        if du > 2 and dv > 2:
                            continue
                        # Literal PACF products, not polynomial arithmetic.
                        sums = [sum((u >> i & 1)*(u >> ((i+r) % d) & 1)
                                    +(v >> i & 1)*(v >> ((i+r) % d) & 1) for i in range(d))
                                for r in range(1, d//2+1)]
                        if all(value % 2 == int(r == k) for r, value in enumerate(sums, 1)):
                            feasible.add((u, v))
                minimum = min((bin(u ^ a).count('1')+bin(v ^ b).count('1')
                              for u, v in feasible), default=None)
                expected = {(u, v) for u, v in feasible
                            if bin(u ^ a).count('1')+bin(v ^ b).count('1') == minimum}
                out = joint_nearest_supports((a, b), d, k, capacity=4096)
                assert out['scan_complete'] and out['complete_nearest']
                assert out['minimum_distance'] == minimum
                assert set(map(tuple, out['masks'])) == expected
                capped = joint_nearest_supports((a, b), d, k, capacity=1)
                assert capped['minimum_distance'] == minimum
                assert set(map(tuple, capped['masks'])) <= expected
                assert len(capped['masks']) == min(1, len(expected))
                if len(expected) > 1:
                    assert not capped['complete_nearest']


def test_joint_cancellation_does_not_certify_scan(monkeypatch):
    import solver.cyclic_joint_search as module
    event = Event()
    event.set()
    monkeypatch.setattr(module, 'parity_support_family', lambda *a: pytest.fail('scan after cancellation'))
    result = joint_nearest_supports((1, 6), 3, 1, stop_event=event)
    assert not result['scan_complete'] and not result['complete_nearest']


def test_joint_l43_outputs_are_independently_parity_valid():
    a = (1 << 6)-1
    b = ((1 << 37)-1) << 6
    out = joint_nearest_supports((a, b), 43, 1, capacity=8)
    assert out['scan_complete']
    assert len(out['slices']) == 2*(1+6*37)
    for u, v in out['masks']:
        assert bin(u).count('1') == 6 and bin(v).count('1') == 37
        assert min(bin(u ^ a).count('1'), bin(v ^ b).count('1')) <= 2
        assert bin(u ^ a).count('1')+bin(v ^ b).count('1') == out['minimum_distance']
        assert support_signature(u, 43) ^ support_signature(v, 43) == 1


@pytest.mark.parametrize('floor', [0, 1, 2, 3, 4, 5])
def test_joint_later_distance_layers_match_literal_support_union(floor):
    a, b, d = 3, 28, 5
    feasible = {(u, v) for u in range(32) for v in range(32)
        if bin(u).count('1') == 2 and bin(v).count('1') == 3
        and min(bin(u ^ a).count('1'), bin(v ^ b).count('1')) <= 2
        and bin(u ^ a).count('1')+bin(v ^ b).count('1') >= floor
        and support_signature(u, d) ^ support_signature(v, d) == 1}
    minimum = min((bin(u ^ a).count('1')+bin(v ^ b).count('1') for u, v in feasible), default=None)
    expected = {(u, v) for u, v in feasible if bin(u ^ a).count('1')+bin(v ^ b).count('1') == minimum}
    out = joint_nearest_supports((a, b), d, 1, capacity=4096, minimum_total_distance=floor)
    assert out['scan_complete'] and out['complete_nearest']
    assert out['minimum_distance'] == minimum
    assert set(map(tuple, out['masks'])) == expected


def test_content_changing_support_union_matches_literal_masks():
    a, b, d = 1, 6, 3
    bounds = ((1, 3), (0, 2))
    feasible = {(u, v) for u in range(8) for v in range(8)
        if bin(u).count('1')+bin(v).count('1') == d
        and bin(u).count('1') % 2 == 1 and bin(v).count('1') % 2 == 0
        and all(low <= bin(m).count('1') <= high for m, (low, high) in zip((u, v), bounds))
        and min(bin(u ^ a).count('1'), bin(v ^ b).count('1')) <= 2
        and bin(u ^ a).count('1')+bin(v ^ b).count('1') >= 3
        and support_signature(u, d) ^ support_signature(v, d) == 1}
    minimum = min(bin(u ^ a).count('1')+bin(v ^ b).count('1') for u, v in feasible)
    expected = {(u, v) for u, v in feasible if bin(u ^ a).count('1')+bin(v ^ b).count('1') == minimum}
    out = joint_nearest_supports((a, b), d, 1, capacity=4096, weight_bounds=bounds, minimum_total_distance=3)
    assert out['minimum_distance'] == minimum and out['complete_nearest']
    assert set(map(tuple, out['masks'])) == expected


def test_content_changing_sign_completion_matches_literal_ternary_words():
    from itertools import product
    from solver.compressed_repair import _validated_reference
    from solver.target_profiles import canonical_target_content_profiles
    from solver.ternary_support_search import support_mask
    from solver.torch_compressed_search import paired_contents
    from solver.structured_energy import integer_periodic_autocorrelation_profile as pacf
    all_words = list(product((-1, 0, 1), repeat=3))
    for target in canonical_target_content_profiles(6):
        for ca, cb in paired_contents(target):
            a, b = (tuple([-1]*n+[0]*z+[1]*p) for n, z, p in (ca, cb))
            _, _, d, wanted = _validated_reference(a, b, target)
            out = repair_joint_cyclic_supports(a, b, target, max_supports=4096,
                                              content_changes=True, minimum_total_distance=3)
            reference_masks = tuple(support_mask(w) for w in (a, b))
            expected = set()
            for aa in all_words:
                for bb in all_words:
                    if tuple(x+y for x, y in zip(pacf(aa), pacf(bb))) != wanted:
                        continue
                    masks = tuple(support_mask(w) for w in (aa, bb))
                    distances = tuple(bin(m ^ r).count('1') for m, r in zip(masks, reference_masks))
                    if min(distances) > 2 or sum(distances) != out['generation']['minimum_distance']:
                        continue
                    if all(2*w.count(-1)+w.count(0) == e+o and 0 <= e-w.count(-1) <= w.count(0)
                           for w, (e, o) in zip((aa, bb), ((target.a_even_ones, target.a_odd_ones),
                                                        (target.b_even_ones, target.b_odd_ones)))):
                        expected.add((aa, bb))
            assert out['status'] == ('SAT_COMPRESSED' if expected else 'BOUNDED_MISS')
            if expected:
                assert out['words'] in expected
                assert not out['layer_exhausted']
            else:
                assert out['layer_exhausted']
