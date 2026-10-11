"""Exact conditioning of an affine binary sign space on one whole row.

For s=c+Bf and selected row s_R=t, eliminate B_R f=t+c_R while
keeping the RHS symbolic in t. Dependent rows give checks on t; each
compatible t has exactly 2**(dim(f)-rank(B_R)) preimages. Restricting t
to constant weight is exhaustive for that cardinality constraint.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class ConditionedSigns:
    side: int
    pattern_variables: tuple
    projection_rank: int
    free_count: int
    checks: tuple  # parity(pattern & mask) == rhs
    expressions: tuple  # (remaining-free mask, pattern mask, constant)


def condition_sign_row(system, side):
    if not system.consistent or type(side) is not int or side not in (0, 1):
        raise ValueError('consistent system and side 0 or 1 required')
    original = system.expressions()
    selected = tuple(i for i, (s, _) in enumerate(system.variables) if s == side)
    basis = {}
    checks = []
    for pattern_bit, column in enumerate(selected):
        row, constant = original[column]
        label = 1 << pattern_bit
        while row:
            pivot = row.bit_length()-1
            if pivot not in basis:
                basis[pivot] = row, label, constant
                break
            prior, rhs_label, rhs_constant = basis[pivot]
            row ^= prior
            label ^= rhs_label
            constant ^= rhs_constant
        if not row:
            checks.append((label, constant))
    remaining = [i for i in range(len(system.free_columns)) if i not in basis]
    free_expressions = [(0, 0, 0)]*len(system.free_columns)
    for bit, column in enumerate(remaining):
        free_expressions[column] = (1 << bit, 0, 0)
    for pivot in sorted(basis):
        row, label, constant = basis[pivot]
        mask = 0
        rest = row ^ (1 << pivot)
        while rest:
            bit = rest & -rest
            gm, tm, c = free_expressions[bit.bit_length()-1]
            mask ^= gm
            label ^= tm
            constant ^= c
            rest ^= bit
        free_expressions[pivot] = mask, label, constant
    expressions = []
    for row, constant in original:
        free_mask = pattern_mask = 0
        while row:
            bit = row & -row
            gm, tm, c = free_expressions[bit.bit_length()-1]
            free_mask ^= gm
            pattern_mask ^= tm
            constant ^= c
            row ^= bit
        expressions.append((free_mask, pattern_mask, constant))
    return ConditionedSigns(side, tuple(system.variables[i] for i in selected), len(basis),
                            len(remaining), tuple(checks), tuple(expressions))
