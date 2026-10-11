from itertools import product
from fractions import Fraction
import random

import pytest

from solver.colour_projection import project_colour_row, certify_assignment


def exact_score(c, g, labels):
    return sum((Fraction(float(c[i])), -Fraction(float(c[i])), Fraction(float(g[i])), -Fraction(float(g[i])))[label]
               for i, label in enumerate(labels))


def test_all_small_count_classes_match_literal_nearest_word_oracle():
    rng = random.Random(43128)
    for n in (3, 5):
        words = tuple(product(range(4), repeat=n))
        groups = {}
        for word in words:
            groups.setdefault(tuple(word.count(v) for v in range(4)), []).append(word)
        c, g = ([rng.randrange(-8, 9)/8 for _ in range(n)] for _ in range(2))
        for counts, feasible in groups.items():
            labels, _ = project_colour_row(c, g, counts)
            assert labels in feasible
            assert exact_score(c, g, labels) == max(exact_score(c, g, word) for word in feasible)


def test_rejected_float_certificate_falls_back_to_exact_integer_solution(monkeypatch):
    monkeypatch.setattr('solver.colour_projection.certify_assignment', lambda *args: False)
    c, g, counts = (2**40, 2**-100, -2**40, 0.), (-1., 3., 4., -2.), (1, 1, 1, 1)
    labels, native = project_colour_row(c, g, counts)
    assert not native
    feasible = [word for word in product(range(4), repeat=4) if tuple(word.count(v) for v in range(4)) == counts]
    assert exact_score(c, g, labels) == max(exact_score(c, g, word) for word in feasible)
    assert not certify_assignment(c, g, (0, 1, 2, 3), (0, 0, 1, 2), (0.,)*4, (0.,)*4)


def test_nonfinite_or_invalid_counts_are_rejected():
    with pytest.raises(ValueError): project_colour_row((float('nan'),), (0.,), (1, 0, 0, 0))
    with pytest.raises(ValueError): project_colour_row((0.,), (0.,), (0, 0, 0, 0))
