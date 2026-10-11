from itertools import product

import pytest

from solver.near_difference_profiles import minimum_difference_error_shell


@pytest.mark.parametrize('d', (3, 5, 7, 9))
def test_minimum_shell_equals_every_literal_even_zero_sum_error_vector(d):
    wanted = (d,) + (0,) * (d-1)
    expected = set()
    # At norm 8 each entry is necessarily in {-2,0,2}; enumerating this
    # entire small cube checks shell coverage independently of positions.
    for independent in product((-2, 0, 2), repeat=d//2):
        if sum(independent) != 0 or sum(v*v for v in independent) != 8:
            continue
        expected.add((d,) + independent + independent[::-1])
    actual = {row for plus, minus, row in minimum_difference_error_shell(wanted)}
    assert actual == expected


def test_l43_shell_retains_dc_origin_symmetry_and_all_420_distinct_profiles():
    wanted = (43, -1) + (0,) * 40 + (-1,)
    rows = list(minimum_difference_error_shell(wanted))
    assert len(rows) == len({row for _, _, row in rows}) == 420
    assert all(row[0] == 43 and sum(row) == sum(wanted)
               and all(row[u] == row[-u % 43] for u in range(43))
               and sum((row[u]-wanted[u])**2 for u in range(1, 22)) == 8
               for _, _, row in rows)


def test_literal_l14_near_witness_reconstructs_energy4_in_original_binary_space():
    from solver.correlation import full_correlation_profile
    from solver.odd_lift_native import difference_sign_problem
    from solver.structured_energy import integer_periodic_autocorrelation_profile as pacf, target_profile
    from solver.target_profiles import TargetContentProfile
    from solver.ternary_support_search import support_mask
    from solver.torch_joint_search import joint_binary_words
    from solver.verifier import verify_pqcp

    target = TargetContentProfile(14, 1, -1, 1, 4, 3, 3)
    c = ((1, 0, 1, 0, 0, 0, 0), (-1, 1, -1, 0, 1, 1, 0))
    g = ((0, -1, 0, 1, 1, 1, 1), (0, 0, 0, 1, 0, 0, -1))
    masks = tuple(support_mask(w) for w in c)
    gm, wanted, negatives = difference_sign_problem(masks, target)
    near = next(row for plus, minus, row in minimum_difference_error_shell(wanted)
                if (plus, minus) == (1, 3))
    assert tuple(support_mask(w) for w in g) == gm
    assert tuple(w.count(-1) for w in g) == negatives
    assert tuple(x+y for x, y in zip(pacf(g[0]), pacf(g[1]))) == near
    words = joint_binary_words(c, g, target)
    original = tuple(full_correlation_profile(*words))
    full_target = target_profile(14, 1, -1)
    assert sum((x-y)**2 for x, y in zip(original[1:8], full_target[1:8])) == 4*16
    assert not verify_pqcp(*words).is_valid
