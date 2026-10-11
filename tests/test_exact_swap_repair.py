from itertools import combinations, product

from solver.correlation import full_correlation_profile
from solver.exact_swap_repair import exact_two_row_swap_repair, row_swap_deltas
from solver.structured_energy import target_profile
from solver.target_profiles import TargetContentProfile


def neighbors(row):
    yield row
    for p, q in combinations(range(len(row)), 2):
        if p % 2 == q % 2 and row[p] != row[q]:
            yield tuple(v ^ int(i in (p, q)) for i, v in enumerate(row))


def test_all_l6_row_swap_deltas_against_original_integer_pacf():
    for row in product((0, 1), repeat=6):
        base = full_correlation_profile(row, row)
        retained = set()
        for delta, swaps in row_swap_deltas(row).items():
            for swap in swaps:
                word = tuple(v ^ int(i in swap) for i, v in enumerate(row))
                actual = full_correlation_profile(word, row)
                assert tuple((x-y)//4 for x, y in zip(actual[1:4], base[1:4])) == delta
                retained.add(word)
        assert retained == set(neighbors(row))


def test_meet_in_middle_matches_literal_cartesian_neighborhood():
    # All centers of one valid L6 content, with both compatible target shifts.
    rows_a = [r for r in product((0, 1), repeat=6) if sum(r[::2]) == 0 and sum(r[1::2]) == 1]
    rows_b = [r for r in product((0, 1), repeat=6) if sum(r[::2]) == 1 and sum(r[1::2]) == 1]
    targets = (TargetContentProfile(6, 1, 1, 0, 1, 1, 1),)
    for a, b in product(rows_a, rows_b):
        expected = {(x, y) for x, y in product(neighbors(a), neighbors(b))
                    if tuple(full_correlation_profile(x, y)) == target_profile(6, 1, 1)}
        out = exact_two_row_swap_repair(a, b, targets=targets)
        assert {w['words'] for w in out['witnesses']} == expected
        assert out['complete']
