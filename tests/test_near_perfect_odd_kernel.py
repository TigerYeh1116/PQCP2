from functools import lru_cache
import random

import pytest

from experiments.search_same_near_perfect_o import near_perfect_ingredients
from solver.correlation import full_correlation_profile
from solver.near_perfect_odd_kernel import near_perfect43_kernel_certificate
from solver.perfect_odd_kernel import interleave_from_complementary_trits
from solver.structured_energy import integer_periodic_autocorrelation_profile as pacf


@lru_cache(maxsize=1)
def ingredients():
    return near_perfect_ingredients()


def test_all_168_literal_ingredients_have_exact_unit_rotation_provenance():
    data = ingredients()
    assert data['complete'] and data['literal_flip_count'] == 168 and len(data['rows']) == 4
    representatives = {(row['parent_mask'], row['flip']): row['o'] for row in data['rows']}
    for record in data['provenance']:
        parent = tuple(-1 if record['parent_mask'] >> i & 1 else 1 for i in range(43))
        o = tuple(1 if i == record['flip'] else v for i, v in enumerate(parent))
        q, r = record['q'], record['r']
        assert tuple(o[(q*i+r) % 43] for i in range(43)) == representatives[
            record['representative_mask'], record['representative_flip']]


def test_certified_kernel_constructions_cancel_every_original_odd_sidelobe():
    rng = random.Random(3456)
    for row in ingredients()['rows']:
        certificate = near_perfect43_kernel_certificate(row['parent'], row['flip'])
        assert certificate['smith_invariants'] == (1,) * 21
        basis, inverse = certificate['basis'], certificate['integral_right_inverse']
        assert all(sum(basis[i][k]*inverse[k][j] for k in range(43)) == int(i == j)
                   for i in range(21) for j in range(21))
        assert certificate['checked_signed_pairs'] == 420
        # No two-basis combination survives both the norm and trit conditions
        # for these FOUR certified representatives, not a general claim.
        assert len(certificate['candidates']) == 21
        for candidate in certificate['candidates']:
            g = candidate['word']
            active = [i for i, v in enumerate(g) if v == 0]
            negative = set(rng.sample(active, (len(active)-9)//2))
            c = tuple(0 if g[i] else -1 if i in negative else 1 for i in range(43))
            assert sum(c) == 9 and sum(g) == 0
            original = tuple(full_correlation_profile(*interleave_from_complementary_trits(c, g, row['o'])))
            assert all(original[u] == 0 for u in range(1, 86, 2))
            pc, pg, po = tuple(pacf(c)), tuple(pacf(g)), certificate['profile']
            assert original[::2] == tuple(2*(pc[u]+pg[u]+po[u]) for u in range(43))


def test_uncertified_parent_cannot_silently_prune_integer_vectors():
    row = ingredients()['rows'][0]
    invalid = list(row['parent'])
    invalid[row['flip']] = 1
    with pytest.raises(ValueError):
        near_perfect43_kernel_certificate(invalid, row['flip'])
