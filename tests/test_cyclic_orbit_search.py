"""Original binary rotations and complete small global partner domains."""
from itertools import permutations, product
from threading import Event

import pytest

from solver.cyclic_orbit_search import search_cyclic_orbits
from solver.target_profiles import canonical_target_content_profiles
from solver.torch_compressed_search import paired_contents
from solver.pqcp_support_solver import complete_pqcp_supports
from solver.ternary_support_search import support_mask


def test_all_even_binary_rotations_induce_same_coarse_and_difference_rotation():
    d = 3
    for bits in product((0, 1), repeat=2*d):
        signs = tuple(1-2*b for b in bits)
        c = tuple((signs[i]+signs[i+d])//2 for i in range(d))
        g = tuple((-1)**i*(signs[i]-signs[i+d])//2 for i in range(d))
        for r in range(0, 2*d, 2):
            shifted = signs[r:]+signs[:r]
            cc = tuple((shifted[i]+shifted[i+d])//2 for i in range(d))
            gg = tuple((-1)**i*(shifted[i]-shifted[i+d])//2 for i in range(d))
            assert cc == tuple(c[(i+r) % d] for i in range(d))
            assert gg == tuple(g[(i+r) % d] for i in range(d))


def test_small_orbit_search_equals_literal_fixed_side_union():
    for target in canonical_target_content_profiles(6):
        for ca, cb in paired_contents(target):
            a, b = (tuple([-1]*n+[0]*z+[1]*p) for n, z, p in (ca, cb))
            masks = support_mask(a), support_mask(b)
            expected = False
            for ma in range(8):
                for mb in range(8):
                    if (bin(ma).count('1'), bin(mb).count('1')) != (3-ca[1], 3-cb[1]):
                        continue
                    if ma != masks[0] and mb != masks[1]:
                        continue
                    expected |= complete_pqcp_supports(a, b, target, (ma, mb))['verified']
            out = search_cyclic_orbits(a, b, target, max_orbits=4096)
            assert out['status'] == ('SAT_PQCP' if expected else 'UNSAT_FIXED_SIDE_UNION')
            assert out['verified'] == expected


def test_orbit_cancellation_is_unknown():
    target = canonical_target_content_profiles(6)[0]
    a, b = (tuple([-1]*n+[0]*z+[1]*p) for n, z, p in paired_contents(target)[0])
    event = Event()
    event.set()
    assert search_cyclic_orbits(a, b, target, stop_event=event)['status'] == 'UNKNOWN'
