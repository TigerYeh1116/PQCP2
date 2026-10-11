"""Every small literal parity-compatible support pair satisfies the bound."""
from collections import defaultdict

import pytest

from solver.support_size_bound import minimum_support_size, support_sizes_meet_bound


@pytest.mark.parametrize('d', [3, 5, 7, 9, 11])
def test_counting_bound_never_excludes_literal_parity_supports(d):
    index = defaultdict(list)
    for mask in range(1 << d):
        bits = [(mask >> i) & 1 for i in range(d)]
        signature = sum((sum(bits[i]*bits[(i+u) % d] for i in range(d)) % 2)
                        << (u-1) for u in range(1, d//2+1))
        index[(sum(bits), signature)].append(mask)
    for (weight, signature), masks in index.items():
        for k in range(1, d//2+1):
            if index.get((d-weight, signature ^ (1 << (k-1)))):
                assert support_sizes_meet_bound((weight, d-weight), d)


def test_l43_supports_below_five_are_impossible():
    assert minimum_support_size(43) == 5
    assert not support_sizes_meet_bound((4, 39), 43)
    assert support_sizes_meet_bound((5, 38), 43)
