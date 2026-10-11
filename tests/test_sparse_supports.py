"""All literal masks prove bracelet representatives and distance parity signatures."""
import pytest

from solver.sparse_supports import sparse_bracelets


@pytest.mark.parametrize('d', [3, 5, 7, 9, 11])
def test_every_weight_against_every_literal_binary_mask(d):
    full = (1 << d)-1
    for weight in range(1, d):
        expected = set()
        for mask in range(1 << d):
            if bin(mask).count('1') != weight:
                continue
            reverse = sum((mask >> i & 1) << ((d-i) % d) for i in range(d))
            orbit = {mask, reverse}
            for m in (mask, reverse):
                orbit |= {((m << r) | (m >> (d-r))) & full for r in range(1, d)}
            expected.add(min(orbit))
        out = sparse_bracelets(d, weight)
        assert out['complete']
        assert out['total'] == len(expected)
        assert {m for m, _ in out['records']} == expected
        for mask, signature in out['records']:
            literal = sum((sum((mask >> i & 1)*(mask >> ((i+u) % d) & 1)
                         for i in range(d)) % 2) << (u-1) for u in range(1, d//2+1))
            assert signature == literal
        capped = sparse_bracelets(d, weight, capacity=1)
        assert capped['complete'] == (len(expected) == 1)
        assert capped['total'] == len(expected)


def test_l43_weight_six_bracelet_count_matches_burnside():
    from math import comb
    out = sparse_bracelets(43, 6, capacity=141778)
    necklaces = comb(43, 6)//43
    reflection_fixed_orbits = comb(21, 3)
    assert out['complete']
    assert out['total'] == (necklaces+reflection_fixed_orbits)//2 == 71554
