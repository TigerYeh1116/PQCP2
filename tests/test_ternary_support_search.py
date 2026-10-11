"""Independent literal oracles for support matching and ALL-sign completion."""
from itertools import permutations, product
from threading import Event

import pytest

from solver.compressed_repair import _validated_reference
from solver.structured_energy import integer_periodic_autocorrelation_profile as pacf
from solver.target_profiles import canonical_target_content_profiles
from solver.torch_compressed_search import paired_contents
from solver.ternary_support_search import (complete_support_signs, matching_supports,
    relocated_supports, repair_support_signs, support_mask, support_signature, support_sign_system,
    support_neighborhood)


@pytest.mark.parametrize('d', [3, 5, 7])
def test_packed_signature_and_neighborhood_match_literal_oracle(d):
    for bits in product((0, 1), repeat=d):
        mask = support_mask(bits)
        expected = sum((v & 1) << (u-1) for u, v in enumerate(pacf(bits)) if 1 <= u <= d//2)
        assert support_signature(mask, d) == expected
        neighborhood = set(relocated_supports(mask, d))
        literal = {support_mask(w) for w in product((0, 1), repeat=d)
                   if sum(w) == sum(bits) and sum(x != y for x, y in zip(w, bits)) <= 2}
        assert neighborhood == literal
        incremental = dict(support_neighborhood(mask, d))
        assert set(incremental) == literal
        assert all(signature == support_signature(m, d) for m, signature in incremental.items())


@pytest.mark.parametrize('length', [6, 14])
@pytest.mark.parametrize('backend', ['native', 'z3'])
def test_exact_support_solver_and_repair_match_all_fixed_content_words(length, backend):
    target = canonical_target_content_profiles(length)[0]
    saw_sat = saw_unsat = False
    for ca, cb in paired_contents(target):
        lists = [sorted(set(permutations([-1]*n+[0]*z+[1]*p))) for n, z, p in (ca, cb)]
        a, b = lists[0][0], lists[1][0]
        _, _, d, wanted = _validated_reference(a, b, target)
        solutions = {}
        for aa in lists[0]:
            for bb in lists[1]:
                if tuple(x+y for x, y in zip(pacf(aa), pacf(bb))) == wanted:
                    solutions.setdefault((support_mask(aa), support_mask(bb)), set()).add((aa, bb))
        neighborhoods = tuple(set(relocated_supports(support_mask(w), d)) for w in (a, b))
        expected_matches = {(ma, mb) for ma in neighborhoods[0] for mb in neighborhoods[1]
            if all((sum(((ma >> i & 1)*(ma >> ((i+u) % d) & 1)
                         +(mb >> i & 1)*(mb >> ((i+u) % d) & 1)) for i in range(d))-wanted[u]) % 2 == 0
                   for u in range(1, d//2+1))}
        assert set(matching_supports(a, b, wanted)) == expected_matches
        for masks in product(*neighborhoods):
            result = complete_support_signs(a, b, target, masks, timeout_ms=10000, backend=backend)
            expected = masks in solutions
            assert result['status'] == ('SAT_COMPRESSED' if expected else 'UNSAT_FIXED_SUPPORT')
            if expected:
                saw_sat = True
                assert result['words'] in solutions[masks]
                system = support_sign_system(masks, d, wanted, (a.count(-1), b.count(-1)))
                # EVERY literal solution satisfies EVERY independently generated linear row.
                for words in solutions[masks]:
                    values = sum((words[s][i] == -1) << col for col, (s, i) in enumerate(system.variables))
                    assert system.consistent
                    assert all(bin(values & row).count('1') % 2 == rhs for _, row, rhs, _ in system.equations)
            else:
                saw_unsat = True
        out = repair_support_signs(a, b, target, max_supports=1000, timeout_ms=10000, backend=backend)
        expected = bool(set(solutions) & expected_matches)
        assert out['status'] == ('SAT_COMPRESSED' if expected else 'UNSAT_SUPPORT_NEIGHBORHOOD')
    assert saw_sat
    if length == 14:
        assert saw_unsat


def test_cancellation_and_solver_unknown_never_claim_unsat(monkeypatch):
    import z3
    target = canonical_target_content_profiles(6)[0]
    ca, cb = paired_contents(target)[0]
    a, b = (tuple([-1]*n+[0]*z+[1]*p) for n, z, p in (ca, cb))
    cancelled = Event(); cancelled.set()
    result = repair_support_signs(a, b, target, stop_event=cancelled)
    assert result['status'] == 'UNKNOWN'
    monkeypatch.setattr(z3.Solver, 'check', lambda self: z3.unknown)
    assert repair_support_signs(a, b, target, backend='z3')['status'] == 'UNKNOWN'


def test_native_assignment_cap_and_full_exhaustion():
    from solver.ternary_signs_native import enumerate_signs
    from solver.orientation_gf2 import OrientationSystem
    # Deliberately unconstrained affine space: native kernel must test all
    # counts and integer shifts itself. The first all-positive assignment
    # fails the required negative counts, so cap=1 cannot prove emptiness.
    variables = tuple((s, i) for s in range(2) for i in range(3))
    system = OrientationSystem(variables, (), (), True)
    partial = enumerate_signs(system, (7, 7), 3, (6, -2, -2), (1, 1), 1)
    assert partial['status'] == 'UNKNOWN' and partial['assignments'] == 1
    # All length-three rows of weight one have PAF=-1 at both shifts.
    complete = enumerate_signs(system, (7, 7), 3, (6, 0, 0), (1, 1), 64)
    assert complete['status'] == 'UNSAT_FIXED_SUPPORT' and complete['assignments'] == 64
    hit = enumerate_signs(system, (7, 7), 3, (6, -2, -2), (1, 1), 64)
    assert hit['status'] == 'SAT_COMPRESSED'
    assert tuple(x+y for x, y in zip(*(pacf(w) for w in hit['words']))) == (6, -2, -2)


def test_support_limit_is_unknown_and_homometric_masks_are_not_dropped(monkeypatch):
    import solver.ternary_support_search as module
    target = canonical_target_content_profiles(6)[0]
    ca, cb = paired_contents(target)[0]
    a, b = (tuple([-1]*n+[0]*z+[1]*p) for n, z, p in (ca, cb))
    monkeypatch.setattr(module, 'matching_supports', lambda *args: iter([(1, 2), (2, 1)]))
    monkeypatch.setattr(module, 'complete_support_signs', lambda *args, **kwargs:
                        dict(status='UNSAT_FIXED_SUPPORT', words=None))
    assert repair_support_signs(a, b, target, max_supports=1)['status'] == 'UNKNOWN'
    result = repair_support_signs(a, b, target, max_supports=2)
    assert result['status'] == 'UNSAT_SUPPORT_NEIGHBORHOOD'
    assert len(result['attempts']) == 2
