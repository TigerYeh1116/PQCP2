"""Exact coarse repair agrees with a literal ternary oracle on small cases."""

from itertools import permutations, product
import pytest

from solver.compressed_repair import repair_compressed_pair
from solver.structured_energy import fold_profile, integer_periodic_autocorrelation_profile, target_profile
from solver.target_profiles import canonical_target_content_profiles
from solver.torch_compressed_search import paired_contents
from solver.ternary_repair_native import repair_two_swaps


def words(content):
    n, z, p = content
    return sorted(set(permutations((-1,)*n+(0,)*z+(1,)*p)))


@pytest.mark.parametrize('length', [6, 14])
def test_repair_vs_complete_fixed_content_oracle(length):
    target = canonical_target_content_profiles(length)[0]
    wanted = tuple(v//4 for v in fold_profile(target_profile(length, target.k, target.eta), 2))
    saw_sat = saw_unsat = False
    for ca, cb in paired_contents(target):
        aa, bb = words(ca), words(cb)
        by_profile = {}
        for b in bb:
            by_profile.setdefault(tuple(integer_periodic_autocorrelation_profile(b)), []).append(b)
        solutions = []
        for a in aa:
            needed = tuple(x-y for x, y in zip(wanted, integer_periodic_autocorrelation_profile(a)))
            solutions.extend((a, b) for b in by_profile.get(needed, ()))
        for a, b in ((aa[0], bb[0]), (aa[-1], bb[-1])):
            for radius in (0, 2, 4, length):
                expected = any(sum(x != y for x, y in zip(a+b, u+v)) <= radius
                               for u, v in solutions)
                result = repair_compressed_pair(a, b, target, radius=radius, timeout_ms=10000)
                assert result.status == ('SAT_COMPRESSED' if expected else 'UNSAT_LOCAL')
                saw_sat |= expected
                saw_unsat |= not expected
                if expected:
                    assert result.words in solutions
                    assert result.distance <= radius
                if radius == 4:
                    native, _ = repair_two_swaps(a, b, target)
                    assert native.status == result.status
                    if expected:
                        assert native.words in solutions
                    bounded, evaluations = repair_two_swaps(a, b, target, max_evaluations=1)
                    assert evaluations <= 1
                    if bounded.status == 'UNKNOWN':
                        assert bounded.words is None
                        assert bounded.reason == 'evaluation budget exhausted'
                    elif bounded.status == 'SAT_COMPRESSED':
                        assert bounded.words in solutions
    assert saw_sat
    if length == 14:
        assert saw_unsat


def test_unknown_is_not_unsat(monkeypatch):
    import z3
    target = canonical_target_content_profiles(6)[0]
    ca, cb = paired_contents(target)[0]
    monkeypatch.setattr(z3.Solver, 'check', lambda self: z3.unknown)
    result = repair_compressed_pair(words(ca)[0], words(cb)[0], target)
    assert result.status == 'UNKNOWN' and result.words is None


@pytest.mark.parametrize('radius', [-1, 7, True])
def test_invalid_radius(radius):
    target = canonical_target_content_profiles(6)[0]
    ca, cb = paired_contents(target)[0]
    with pytest.raises(ValueError, match='radius'):
        repair_compressed_pair(words(ca)[0], words(cb)[0], target, radius=radius)


def test_native_cap_really_returns_unknown():
    target = canonical_target_content_profiles(14)[0]
    for ca, cb in paired_contents(target):
        result, count = repair_two_swaps(words(ca)[0], words(cb)[0], target, max_evaluations=1)
        if result.status == 'UNKNOWN':
            assert count == 1 and result.words is None
            return
    pytest.fail('fixture must exercise a capped unfinished enumeration')


def test_two_swap_ball_equals_literal_radius_four_for_ternary_alphabet():
    def swaps(pair):
        for side, word in enumerate(pair):
            for i in range(len(word)):
                for j in range(i+1, len(word)):
                    changed = list(word)
                    changed[i], changed[j] = changed[j], changed[i]
                    yield tuple(tuple(changed) if s == side else w for s, w in enumerate(pair))
    references = list(product(product((-1, 0, 1), repeat=3), repeat=2))
    references += [(w, (0,)*4) for w in product((-1, 0, 1), repeat=4)]
    for reference in references:
        one = {reference, *swaps(reference)}
        two = one | {pair for intermediate in one for pair in swaps(intermediate)}
        expected = {pair for pair in product(*(set(permutations(w)) for w in reference))
                    if sum(x != y for w, r in zip(pair, reference) for x, y in zip(w, r)) <= 4}
        assert two == expected
