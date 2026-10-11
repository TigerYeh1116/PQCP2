"""Original binary pairs, independent of both ternary sign solvers."""
from collections import defaultdict
from itertools import product

from solver.correlation import full_correlation_profile
from solver.pqcp_support_solver import complete_pqcp_supports
from solver.structured_energy import target_profile, integer_periodic_autocorrelation_profile as pacf
from solver.target_profiles import canonical_target_content_profiles, pair_content, TargetContentProfile
from solver.ternary_support_search import support_mask
from solver.torch_compressed_search import paired_contents


def test_all_l6_fixed_supports_against_all_original_binary_pairs():
    bits = list(product((0, 1), repeat=6))
    saw_sat = saw_unsat = False
    for target in canonical_target_content_profiles(6):
        expected = defaultdict(set)
        for a in bits:
            for b in bits:
                if (pair_content(a, b) != (target.a_even_ones, target.a_odd_ones,
                                          target.b_even_ones, target.b_odd_ones)
                        or tuple(full_correlation_profile(a, b)) != target_profile(6, target.k, target.eta)):
                    continue
                masks = tuple(sum(int(w[i] == w[i+3]) << i for i in range(3)) for w in (a, b))
                expected[masks].add((a, b))
        for ca, cb in paired_contents(target):
            a, b = (tuple([-1]*n+[0]*z+[1]*p) for n, z, p in (ca, cb))
            for ma in range(8):
                for mb in range(8):
                    if (bin(ma).count('1'), bin(mb).count('1')) != (3-ca[1], 3-cb[1]):
                        continue
                    out = complete_pqcp_supports(a, b, target, (ma, mb))
                    wanted = expected[(ma, mb)]
                    assert out['status'] == ('SAT_PQCP' if wanted else 'UNSAT_BINARY_SUPPORT')
                    if wanted:
                        assert out['verified'] and out['words'] in wanted
                        saw_sat = True
                    else:
                        assert not out['verified']
                        assert out['excluded_by'] in ('coarse', 'difference')
                        saw_unsat = True
                    partial = complete_pqcp_supports(a, b, target, (ma, mb), max_assignments=1)
                    if partial['status'] == 'UNSAT_BINARY_SUPPORT':
                        assert not wanted
                    if partial['status'] == 'SAT_PQCP':
                        assert partial['words'] in wanted
    assert saw_sat


def test_two_problem_solver_rejects_known_dead_coarse_support():
    # Literal masks from the experimentally discovered L86 coarse witness.
    # Its native lift has been completely exhausted, not merely timed out.
    target = TargetContentProfile(86, 1, -1, 16, 22, 18, 21)
    a = tuple([-1]*7+[0]*24+[1]*12)
    b = tuple([-1]*10+[0]*19+[1]*14)
    out = complete_pqcp_supports(a, b, target, (1900755849245, 2652063888105))
    assert out['status'] == 'UNSAT_BINARY_SUPPORT' and out['excluded_by'] == 'difference'
    assert out['attempts']['coarse']['status'] == 'SAT_COMPRESSED'
    assert out['attempts']['difference']['status'] == 'UNSAT_FIXED_SUPPORT'


def test_l14_two_problem_completion_against_original_binary_word_index():
    import torch
    from solver.torch_compressed_search import search_compressed
    index = defaultdict(list)
    for bits in product((0, 1), repeat=14):
        key = (sum(bits[::2]), sum(bits[1::2]),
               sum(int(bits[i] == bits[i+7]) << i for i in range(7)))
        index[key].append((bits, pacf(tuple(1-2*v for v in bits))))
    targets = canonical_target_content_profiles(14)
    profiles = tuple(targets[i % len(targets)] for i in range(32))
    result = search_compressed(profiles, device=torch.device('cpu'), seed=791, steps=1)
    saw_sat = saw_unsat = False
    for pair, target in zip(result.words.int().tolist(), profiles):
        a, b = (tuple(w) for w in pair)
        masks = support_mask(a), support_mask(b)
        wanted = target_profile(14, target.k, target.eta)
        expected = {(aa, bb) for aa, pa in index[(target.a_even_ones, target.a_odd_ones, masks[0])]
                    for bb, pb in index[(target.b_even_ones, target.b_odd_ones, masks[1])]
                    if tuple(x+y for x, y in zip(pa, pb)) == wanted}
        out = complete_pqcp_supports(a, b, target, masks)
        assert out['status'] == ('SAT_PQCP' if expected else 'UNSAT_BINARY_SUPPORT')
        if expected:
            assert out['words'] in expected and out['verified']
            saw_sat = True
        else:
            saw_unsat = True
    assert saw_sat and saw_unsat
