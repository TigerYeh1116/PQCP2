"""Truncated C search must return exactly the brute-force witness orbits for its kept patterns (d=23)."""
import itertools
import random

from solver.compressed_repair import _validated_reference
from solver.gf2_support_family import gf2_support_ring, iter_support_family
from solver.structured_energy import integer_periodic_autocorrelation_profile as pacf
from solver.truncated_search import family_target, top_signs, truncated_scan


def _completes(support_b, d, need, negatives):
    """Brute force: any sign pattern on support_b with this many negatives and PACF == need (u = 1..d//2)."""
    positions = [j for j in range(d) if support_b >> j & 1]
    for neg in itertools.combinations(positions, negatives):
        x = [0]*d
        for j in positions:
            x[j] = -1 if j in neg else 1
        r = pacf(x)
        if all(r[u] == need[u-1] for u in range(1, d//2+1)):
            return True
    return False


def test_truncated_scan_matches_brute_force_on_l46():
    import experiments.search_l94_supports as search
    search._GROUPS = None
    groups = [g for g in search._groups(tuple(range(1, 23)), 46) if g['contents'][0][0]+g['contents'][0][2] == 11]
    search._GROUPS = None
    d, ring, rng, keep = 23, gf2_support_ring(23), random.Random(5), 4
    canon = lambda v: min(((v << s) | (v >> (d-s))) & ((1 << d)-1) for s in range(d))
    checked = 0
    for trial in range(4):
        group = groups[trial % len(groups)]
        (an, az, ap), (bn, bz, bp) = group['contents']
        a, b, _, wanted = _validated_reference((-1,)*an+(0,)*az+(1,)*ap, (-1,)*bn+(0,)*bz+(1,)*bp, group['targets'][0])
        k = min(group['targets'][0].k, d-group['targets'][0].k)
        support_a = sum(1 << j for j in rng.sample(range(d), an+ap))
        if not family_target(d, support_a, k)[0]:
            continue
        patterns, _, _ = top_signs(d, support_a, an, keep)
        hits, stats = truncated_scan(d, support_a, k, d-an-ap, bn, wanted, patterns)
        orbits = {canon(v) for v in iter_support_family(ring, support_a, k) if bin(v).count('1') == d-an-ap}
        assert stats['weight_matches'] == len(orbits)
        expected = {o for o in orbits
                    if any(_completes(o, d, [wanted[u]-p[u-1] for u in range(1, d//2+1)], bn) for p in patterns)}
        assert {canon(sb) for sb, _, _ in hits} == expected
        checked += 1
    assert checked >= 3
