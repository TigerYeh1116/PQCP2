"""Certified identical-O odd kernels after one negative-to-positive flip.

This is an existence reduction for one explicit L86 construction family.
It does not exclude general PQCPs. Smith invariants are computed over ZZ;
failure to certify saturation raises rather than pruning unknown vectors.
"""
from itertools import combinations

from .interleaving import periodic_sign_profile


def near_perfect43_kernel_certificate(parent, flipped_position):
    from sympy import Matrix, ZZ
    from sympy.polys.matrices import DomainMatrix
    from sympy.polys.matrices.normalforms import smith_normal_decomp

    parent = tuple(parent)
    if (len(parent) != 43 or sum(parent) != 1
            or periodic_sign_profile(parent) != (43,) + (-1,) * 42
            or type(flipped_position) is not int or not 0 <= flipped_position < 43
            or parent[flipped_position] != -1):
        raise ValueError('perfect sum-one parent and one negative position required')
    o = tuple(1 if i == flipped_position else v for i, v in enumerate(parent))
    profile = periodic_sign_profile(o)
    if sum(o) != 3 or not set(profile[1:]) <= {-5, -1, 3}:
        raise RuntimeError('one-flip profile identity failed')
    basis = tuple(tuple((o[(i-t) % 43]-o[(i+t) % 43])//2
                        for i in range(43)) for t in range(1, 22))
    gram = tuple(tuple(sum(x*y for x, y in zip(a, b)) for b in basis) for a in basis)
    for t, g in enumerate(basis, 1):
        if sum(g) != 0:
            raise RuntimeError('kernel basis sum identity failed')
        cross = tuple(sum(g[(i+u) % 43]*o[i] for i in range(43)) for u in range(43))
        expected = tuple((profile[(u-t) % 43]-profile[(u+t) % 43])//2 for u in range(43))
        if cross != expected or any(cross[u]+cross[-u % 43] for u in range(43)):
            raise RuntimeError('odd-kernel convolution identity failed')
    if (any(not 20 <= gram[i][i] <= 24 for i in range(21))
            or any(abs(gram[i][j]) > 4 for i in range(21) for j in range(21) if i != j)):
        raise RuntimeError('one-flip Gram bounds failed')
    matrix = Matrix(basis)
    smith, left, right = smith_normal_decomp(DomainMatrix.from_Matrix(matrix).convert_to(ZZ))
    smith, left, right = smith.to_Matrix(), left.to_Matrix(), right.to_Matrix()
    invariants = tuple(abs(int(smith[i, i])) for i in range(21))
    if invariants != (1,) * 21:
        raise RuntimeError('kernel basis not saturated; other integer vectors cannot be excluded')
    inverse = right[:, :21]*Matrix.diag(*(smith[i, i] for i in range(21)))*left
    if any(not value.is_Integer for value in inverse):
        raise RuntimeError('saturation inverse is not integral')
    inverse = tuple(tuple(int(inverse[i, j]) for j in range(21)) for i in range(43))
    # A reader can check B*inverse=I using ONLY literal Python integers.
    # This proves saturation without trusting the normal-form algorithm.
    if any(sum(basis[i][k]*inverse[k][j] for k in range(43)) != int(i == j)
           for i in range(21) for j in range(21)):
        raise RuntimeError('integer right-inverse saturation certificate failed')
    # On the zero-sum subspace, Fourier magnitude of O is >=sqrt(44)-2.
    # Therefore ||sum alpha_t g_t||^2 >=(24-2sqrt(44))*||alpha||^2
    # >10*||alpha||^2. Complementary c with sum 9 forces ||g||^2<=34,
    # so integral alpha has squared norm <=3. Three signed basis vectors
    # have norm >=3*20-6*4=36 and cannot occur. Only singles/pairs remain.
    candidates = [dict(terms=((t, 1),), word=g) for t, g in enumerate(basis, 1)]
    checked_pairs = 0
    for i, j in combinations(range(21), 2):
        for sign in (-1, 1):
            checked_pairs += 1
            norm = gram[i][i]+gram[j][j]+2*sign*gram[i][j]
            if norm > 34:
                continue
            g = tuple(x+sign*y for x, y in zip(basis[i], basis[j]))
            if all(abs(v) <= 1 for v in g):
                if sum(v*v for v in g) != norm:
                    raise RuntimeError('Gram norm disagrees with literal trit norm')
                candidates.append(dict(terms=((i+1, 1), (j+1, sign)), word=g))
    return dict(o=o, profile=profile, basis=basis, gram=gram,
                smith_invariants=invariants, integral_right_inverse=inverse,
                primitive=True, kernel_dimension=21,
                coefficient_squared_norm_upper_bound=3, checked_signed_pairs=checked_pairs,
                candidates=candidates, global_sign_quotient='Row swap and global alternation',
                scope='Identical one-flip perfect O rows, even positive L86 target, even-row sums 9.')
