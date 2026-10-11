from itertools import product

from solver.interleaving import interleave_bits, odd_cross_vectors, periodic_sign_profile, sparse_odd_matches


def test_all_small_interleaves_match_direct_full_periodic_correlation():
    for d in (1, 3, 5):
        rows = tuple(product((-1, 1), repeat=d))
        for e in rows:
            for o in rows:
                pe, po = periodic_sign_profile(e), periodic_sign_profile(o)
                seen = set()
                for vector, offsets in odd_cross_vectors(e, o).items():
                    for delta in offsets:
                        seen.add(delta)
                        bits = interleave_bits(e, o, delta)
                        p = periodic_sign_profile(tuple(1-2*v for v in bits))
                        assert p[::2] == tuple(a+b for a, b in zip(pe, po))
                        assert vector == p[1:d+1:2]
                assert seen == set(range(d))


def test_sparse_join_preserves_preimages_and_excludes_half_shift():
    vectors = tuple(product((-4, 0, 4), repeat=3))
    left = {v: [(i, 0), (i, 1)] for i, v in enumerate(vectors)}
    right = {v: [(i, 2), (i, 3)] for i, v in enumerate(vectors)}
    for eta in (-1, 1):
        actual = set(sparse_odd_matches(left, right, eta))
        expected = set()
        for a, pre_a in left.items():
            for b, pre_b in right.items():
                delta = tuple(x-y for x, y in zip(a, b))
                for r in range(2):
                    wanted = tuple(4*eta if i == r else 0 for i in range(3))
                    if delta == wanted:
                        expected.update((2*r+1, pa, pb) for pa in pre_a for pb in pre_b)
        assert actual == expected
