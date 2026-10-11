"""Independent integer autocorrelations and literal small support oracles."""
from collections import defaultdict
from itertools import combinations

import pytest

from solver.automorphism_construction import (assemble_automorphism_pair,
    positive_automorphism_target, projective_order, search_automorphism_target)
from solver.cyclic_support import cyclic_norm_family, iter_family, rotation_representatives


def ac(word):
    return tuple(sum(word[i]*word[(i+s) % len(word)] for i in range(len(word)))
                 for s in range(len(word)))


@pytest.mark.parametrize('d', (3, 5, 11))
def test_generic_norm_all_targets_and_rotation_orbits(d):
    by_norm = defaultdict(set)
    for mask in range(1 << d):
        word = tuple((mask >> i) & 1 for i in range(d))
        norm = sum((v & 1) << i for i, v in enumerate(ac(word)))
        by_norm[norm].add(mask)
    full = (1 << d)-1
    for half in range(1 << (d//2+1)):
        desired = (half & 1) | sum(((half >> u) & 1)*((1 << u) | (1 << (d-u)))
                                   for u in range(1, d//2+1))
        family = cyclic_norm_family(desired, d)
        assert family.norm_target == desired
        masks = list(iter_family(family))
        assert len(masks) == len(set(masks)) == family.size
        assert set(masks) == by_norm[desired]
        representatives = rotation_representatives(family)
        assert representatives.norm_target == desired
        expanded = set()
        for mask in iter_family(representatives):
            orbit = {mask} | {((mask << u) | (mask >> (d-u))) & full for u in range(1, d)}
            assert not expanded & orbit
            expanded |= orbit
        assert expanded == set(masks)


@pytest.mark.parametrize('d,sc,sh', ((3, 2, 1), (11, 3, 2)))
def test_every_small_constructed_parameter_is_verified_in_original_length(d, sc, sh):
    for v in range(1, d//2+1):
        if projective_order(v, d) != (d-1)//2:
            continue
        for k in range(1, d//2+1):
            spec = positive_automorphism_target(d, v, k, sc, sh)
            result = search_automorphism_target(spec)
            literal_exists = False
            for active in combinations(range(d), spec.weight_c):
                for negatives in combinations(active, spec.negative_c):
                    c = tuple(-1 if i in negatives else 1 if i in active else 0 for i in range(d))
                    if ac(c) != spec.wanted_c:
                        continue
                    inactive = tuple(i for i in range(d) if i not in active)
                    for h_negatives in combinations(inactive, spec.negative_h):
                        h = tuple(-1 if i in h_negatives else 1 if i in inactive else 0 for i in range(d))
                        if ac(h) == spec.wanted_h:
                            literal_exists = True
                            break
                    if literal_exists:
                        break
                if literal_exists:
                    break
            assert result['verified'] == literal_exists
            if not literal_exists:
                assert result['complete'] and result['status'] == 'UNSAT_CONSTRUCTOR_TARGET'
                continue
            assert result['status'] == 'SAT_PQCP'
            a, b = (tuple(1-2*z for z in bits) for bits in result['words'])
            assert tuple(x+y for x, y in zip(ac(a), ac(b))) == tuple(
                4*d if i == 0 else 4 if i in (2*k, 2*d-2*k) else 0
                for i in range(2*d))
            row = result['retained_cases'][-1]
            c, h = (row['attempts'][side]['words'][0] for side in ('c', 'h'))
            assert assemble_automorphism_pair(list(c), tuple(h), spec)[0] == result['words']
            with pytest.raises(ValueError):
                assemble_automorphism_pair(c, c, spec)


def test_length_86_profiles_recurrence_and_unknown_is_not_a_proof():
    for v in range(1, 22):
        if projective_order(v, 43) != 21:
            continue
        for k in range(1, 22):
            spec = positive_automorphism_target(43, v, k, 6, 3)
            assert (spec.weight_c, spec.negative_c, spec.negative_h) == (14, 4, 13)
            assert sum(spec.wanted_c) == 36 and sum(spec.wanted_h) == 9
            assert all(spec.epsilon[t]+spec.epsilon[v*t % 43] ==
                       (2 if t in (k, 43-k) else 0) for t in range(1, 43))
    import threading
    stop = threading.Event()
    stop.set()
    out = search_automorphism_target(positive_automorphism_target(43, 3, 1, 6, 3), stop_event=stop)
    assert out['status'] == 'UNKNOWN' and not out['complete'] and not out['verified']


@pytest.mark.parametrize('d,v,k,sc,sh', ((43, 1, 1, 6, 3), (43, 3, 0, 6, 3),
    (43, 3, 1, 3, 6), (11, 2, 1, 2, 3), (43, True, 1, 6, 3)))
def test_invalid_character_profiles_are_rejected(d, v, k, sc, sh):
    with pytest.raises(ValueError):
        positive_automorphism_target(d, v, k, sc, sh)
