from itertools import product

from solver.correlation import full_correlation_profile
from solver.cyclic_support_blocks import support_parity_energy_lower_bound
from solver.structured_energy import target_profile
from solver.target_profiles import canonical_target_content_profiles, pair_content


def test_l6_bound_against_all_literal_binary_pairs_with_valid_contents():
    words = tuple(product((0, 1), repeat=6))
    seen = 0
    for target in canonical_target_content_profiles(6):
        counts = (target.a_even_ones, target.a_odd_ones, target.b_even_ones, target.b_odd_ones)
        for a in words:
            for b in words:
                if pair_content(a, b) != counts:
                    continue
                profile = tuple(full_correlation_profile(a, b))
                if profile[3] != 0:
                    continue
                masks = tuple(sum(int(w[i] == w[i+3]) << i for i in range(3)) for w in (a, b))
                bound = support_parity_energy_lower_bound(masks, 3, min(target.k, 3-target.k))
                wanted = target_profile(6, target.k, target.eta)
                error = sum((profile[u]-wanted[u])**2 for u in range(1, 4))
                assert error >= 16*bound
                seen += 1
    assert seen > 0


def test_all_saved_gs_seeds_attain_the_fixed_support_lower_bound():
    # Literal production specimen, independent of git-ignored result banks.
    rows = [dict(words=tuple(tuple(map(int, word)) for word in (
        '00010011100100011100010011111011100001010000010000010110111010110001001001000110110001',
        '00001110110000110110100110100000001001010101000000011101010111100111100011010011101101')),
        k=9, sign=1)]
    for row in rows:
        a, b = row['words']
        d = len(a)//2
        masks = tuple(sum(int(w[i] == w[i+d]) << i for i in range(d)) for w in (a, b))
        bound = support_parity_energy_lower_bound(masks, d, min(row['k'], d-row['k']))
        profile = tuple(full_correlation_profile(a, b))
        wanted = target_profile(2*d, row['k'], row['sign'])
        energy = sum((profile[u]-wanted[u])**2 for u in range(1, d+1))//16
        assert d == 43 and profile[d] == 0
        assert bound == energy == 8
