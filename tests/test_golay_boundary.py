"""Literal original-length PACF validates the independently derived expansion."""
from itertools import combinations, product

from experiments.search_golay_boundary import audit_golay, boundary_coefficients, generate_prefixes
from solver.correlation import full_correlation_profile


def compare(pair, tail, bits):
    words = tuple(tuple((1-v)//2 for v in prefix)+tuple(bits[side*tail:(side+1)*tail])
                  for side, prefix in enumerate(pair))
    signs = tuple(1-2*v for v in bits)
    original = full_correlation_profile(*words)
    for u in range(1, len(words[0])):
        linear, edges = boundary_coefficients(pair, tail, u)
        expanded = sum(c*v for c, v in zip(linear, signs))+sum(signs[a]*signs[b] for a, b in edges)
        assert expanded == original[u]


def test_all_tails_of_small_golay_prefix_match_original_binary_pacf():
    pair = ((1, 1, 1, -1), (1, 1, -1, 1))
    audit_golay(pair)
    for tail in (1, 2, 3):
        for bits in product((0, 1), repeat=2*tail):
            compare(pair, tail, bits)


def test_generated_l86_prefixes_and_full_quadratic_interpolation_oracle():
    prefixes, counts = generate_prefixes()
    assert counts == [(20, 8), (40, 56), (80, 384)]
    # Both sides of the identity are multilinear degree <=2 in tail bits.
    # 0, all units, and all pairs therefore determine their exact coefficients.
    patterns = [(0,)*12]
    for size in (1, 2):
        patterns.extend(tuple(int(i in active) for i in range(12))
                        for active in combinations(range(12), size))
    assert len(patterns) == 79
    for index in (0, 1, 63, 127, 255, 383):
        for bits in patterns:
            compare(prefixes[index], 6, bits)
