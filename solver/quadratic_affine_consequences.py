"""Exact affine consequences of a Boolean quadratic system; research helper.

Each polynomial is packed across equations. The kernel of the quadratic
coefficient matrix gives all equation combinations with zero quadratic part.
No random-system rank assumption or nonlinear monomial relaxation is used.
This module is not dispatched by the production sign solver.
"""


def affine_combinations(quadratic, variables, equations):
    """Basis of equation masks whose quadratic terms cancel identically."""
    if type(variables) is not int or type(equations) is not int or variables < 0 or equations < 0:
        raise ValueError('nonnegative integer dimensions required')
    if len(quadratic) != variables * variables:
        raise ValueError('square coefficient table required')
    if variables < 0 or equations < 0 or any(
            type(x) is not int or x < 0 or x >> equations for x in quadratic):
        raise ValueError('nonnegative bounded coefficient fields required')
    if any(quadratic[i * variables + i] for i in range(variables)) or any(
            quadratic[i * variables + j] != quadratic[j * variables + i]
            for i in range(variables) for j in range(i + 1, variables)):
        raise ValueError('symmetric zero-diagonal polar coefficients required')
    basis = {}
    kernel = []
    for equation in range(equations):
        row = 0
        position = 0
        for i in range(variables):
            for j in range(i + 1, variables):
                row |= ((quadratic[i * variables + j] >> equation) & 1) << position
                position += 1
        combination = 1 << equation
        while row:
            pivot = row.bit_length() - 1
            if pivot not in basis:
                basis[pivot] = (row, combination)
                break
            other, mask = basis[pivot]
            row ^= other
            combination ^= mask
        if not row:
            kernel.append(combination)
    return tuple(kernel)


def restrict_affine(combinations, constant, linear):
    """Return complete affine parametrization and rank, or a contradiction.

    All arithmetic is in F2. Reconstruct x = base XOR sum(toggles[j] * y[j]).
    Input combinations must come from affine_combinations for this system.
"""
    basis = {}
    for combination in combinations:
        rhs = bin(constant & combination).count('1') & 1
        row = sum((bin(value & combination).count('1') & 1) << j
                  for j, value in enumerate(linear))
        while row:
            pivot = row.bit_length() - 1
            if pivot not in basis:
                basis[pivot] = (row, rhs)
                break
            other, target = basis[pivot]
            row ^= other
            rhs ^= target
        if not row and rhs:
            return dict(consistent=False, rank=len(basis), base=None, toggles=())

    def complete(value, homogeneous=False):
        for pivot in sorted(basis):
            row, rhs = basis[pivot]
            bit = (bin(row & value).count('1') & 1) ^ (0 if homogeneous else rhs)
            value ^= bit << pivot
        return value

    base = complete(0)
    toggles = tuple(complete(1 << j, True) for j in range(len(linear)) if j not in basis)
    return dict(consistent=True, rank=len(basis), base=base, toggles=toggles)
