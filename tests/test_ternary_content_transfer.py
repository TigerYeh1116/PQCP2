from itertools import permutations

from solver.target_profiles import canonical_target_content_profiles
from solver.torch_compressed_search import paired_contents
from solver.structured_energy import integer_periodic_autocorrelation_profile as pacf, fold_profile, target_profile
from solver.ternary_content_transfer import repair_content_transfer, _moves, _apply
from solver.ternary_repair_native import repair_two_swaps


def words(c):
    n, z, p = c
    return sorted(set(permutations((-1,)*n+(0,)*z+(1,)*p)))


def test_transfer_deltas_are_exact():
    from itertools import product
    for d in (3, 5):
        for w in product((-1, 0, 1), repeat=d):
            old = pacf(w)
            for create in (False, True):
                for delta, move in _moves(w, create):
                    new = pacf(_apply(w, move))
                    assert delta == tuple(new[u]-old[u] for u in range(1, d//2+1))
                    assert new[0]-old[0] == (2 if create else -2)


def test_union_covers_full_radius_four_across_contents():
    saw_transfer_sat = saw_transfer_unsat = False
    for length in (6, 14):
        for target in canonical_target_content_profiles(length):
            wanted = tuple(v//4 for v in fold_profile(target_profile(length, target.k, target.eta), 2))
            solutions, refs = [], []
            for ca, cb in paired_contents(target):
                aa, bb = words(ca), words(cb)
                by_profile = {}
                for b in bb:
                    by_profile.setdefault(tuple(pacf(b)), []).append(b)
                for a in aa:
                    solutions.extend((a, b) for b in by_profile.get(tuple(x-y for x, y in zip(wanted, pacf(a))), ()))
                refs.extend(((aa[0], bb[0]), (aa[-1], bb[-1])))
            for a, b in refs:
                close = [(u, v) for u, v in solutions if sum(x != y for x, y in zip(a+b, u+v)) <= 4]
                transferred = [(u, v) for u, v in close if sorted(u) != sorted(a)]
                result, _ = repair_content_transfer(a, b, target)
                assert (result.words is not None) == bool(transferred)
                saw_transfer_sat |= bool(transferred)
                saw_transfer_unsat |= not transferred
                if result.words:
                    assert result.words in transferred
                fixed, _ = repair_two_swaps(a, b, target)
                assert bool(close) == (fixed.words is not None or result.words is not None)
    assert saw_transfer_sat and saw_transfer_unsat
