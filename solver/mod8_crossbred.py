"""Certified degree-three consequences linear in the selected tail variables.

Work in the Boolean ring, so multiplication of monomial masks is bitwise OR.
Every derived equation is an XOR of f_i and z_j*f_i. No semi-regularity,
random-system or Hilbert-series assumption is used to exclude a prefix.
Experimental: not selected by the production search runner.
"""
from itertools import combinations


def boolean_equations(coefficients):
    constant, linear, quadratic = coefficients
    q = len(linear)
    if (not 0 <= q <= 24 or len(quadratic) != q*q
            or any(type(v) is not int or not 0 <= v < 2**32
                   for v in (constant, *linear, *quadratic))):
        raise ValueError('uint32 Boolean quadratic coefficients through dimension 24 required')
    width = max((constant, *linear, *quadratic)).bit_length()
    equations = []
    for bit in range(width):
        terms = {0} if constant >> bit & 1 else set()
        for i in range(q):
            if linear[i] >> bit & 1:
                terms.add(1 << i)
            for j in range(i+1, q):
                if quadratic[i*q+j] >> bit & 1:
                    terms.add((1 << i) | (1 << j))
        equations.append(frozenset(terms))
    return tuple(equations)


def multiply_terms(terms, factor):
    result = set()
    for term in terms:
        value = term | factor
        if value in result:
            result.remove(value)
        else:
            result.add(value)
    return result


def replay_certificate(originals, q, proof):
    result = set()
    while proof:
        low = proof & -proof
        number = low.bit_length()-1
        proof ^= low
        equation, multiplier = divmod(number, q+1)
        factor = 0 if not multiplier else 1 << (multiplier-1)
        result.symmetric_difference_update(multiply_terms(originals[equation], factor))
    return frozenset(result)


def degree_three_tail_consequences(coefficients, tail=6, *, multiplier_axes=None):
    """Exact intersection of the selected row space with tail-degree <= 1.

    A subset of multipliers weakens the filter, never its validity. Proof bit
    numbers always use the full q+1 stride, even with a subset of axes.
    """
    originals = boolean_equations(coefficients)
    q = len(coefficients[1])
    if type(tail) is not int or not 0 <= tail <= min(q, 8):
        raise ValueError('tail between zero and min(q,8) required')
    axes = tuple(range(q)) if multiplier_axes is None else tuple(multiplier_axes)
    if len(set(axes)) != len(axes) or any(type(j) is not int or not 0 <= j < q for j in axes):
        raise ValueError('distinct literal multiplier axes required')
    prefix = q-tail
    tail_mask = ((1 << q)-1) ^ ((1 << prefix)-1)
    monomials = [0]
    for degree in (1, 2, 3):
        monomials.extend(sum(1 << j for j in group)
                         for group in combinations(range(q), degree))
    good = [m for m in monomials if bin(m & tail_mask).count('1') <= 1]
    bad = [m for m in monomials if bin(m & tail_mask).count('1') >= 2]
    monomials = good + bad
    positions = {term: j for j, term in enumerate(monomials)}
    basis = {}
    row_number = 0
    for equation, original in enumerate(originals):
        for axis in (-1, *axes):
            factor = 0 if axis < 0 else 1 << axis
            row = 0
            for term in multiply_terms(original, factor):
                row ^= 1 << positions[term]
            proof = 1 << (equation*(q+1)+axis+1)
            row_number += 1
            while row:
                pivot = row.bit_length()-1
                if pivot not in basis:
                    basis[pivot] = row, proof
                    break
                other, certificate = basis[pivot]
                row ^= other
                proof ^= certificate
    consequences = []
    for pivot, (row, proof) in sorted(basis.items()):
        if pivot >= len(good):
            continue
        terms = []
        remaining = row
        while remaining:
            low = remaining & -remaining
            remaining ^= low
            terms.append(monomials[low.bit_length()-1])
        if replay_certificate(originals, q, proof) != frozenset(terms):
            raise RuntimeError('independent symbolic certificate replay failed')
        consequences.append(dict(terms=tuple(terms), proof=hex(proof)))
    # Pack all constraints in parallel into wide Python integers. A block
    # represents one tail coefficient (block zero is the constant).
    count = len(consequences)
    packed = {}
    prefix_mask = (1 << prefix)-1
    for equation, consequence in enumerate(consequences):
        for term in consequence['terms']:
            suffix = term >> prefix
            block = suffix.bit_length() if suffix else 0
            key = term & prefix_mask
            packed[key] = packed.get(key, 0) ^ (1 << (block*count+equation))
    return dict(q=q, tail=tail, prefix=prefix, input_equations=len(originals),
                multiplied_rows=row_number, monomials=len(monomials),
                multiplier_axes=axes,
                eliminated_columns=len(bad), consequence_count=count,
                consequences=tuple(consequences), packed=tuple(packed.items()),
                all_certificates_independently_replayed=True)


def specialize_tail_system(derived, prefix_bits):
    """Return (constant syndrome, tail columns) after exact specialization."""
    if type(prefix_bits) is not int or not 0 <= prefix_bits < 1 << derived['prefix']:
        raise ValueError('literal prefix assignment required')
    value = 0
    for monomial, coefficient in derived['packed']:
        if prefix_bits & monomial == monomial:
            value ^= coefficient
    count = derived['consequence_count']
    mask = (1 << count)-1
    return value & mask, tuple((value >> (j*count)) & mask
                               for j in range(1, derived['tail']+1))


def affine_tail_indices(constant, columns):
    """ALL solutions of the specialized GF2 system; empty means inconsistent."""
    basis = {}
    nullspace = []
    for j, column in enumerate(columns):
        label = 1 << j
        while column:
            pivot = column.bit_length()-1
            if pivot not in basis:
                basis[pivot] = column, label
                break
            row, other = basis[pivot]
            column ^= row
            label ^= other
        if not column:
            nullspace.append(label)
    solution = 0
    while constant:
        pivot = constant.bit_length()-1
        if pivot not in basis:
            return ()
        row, label = basis[pivot]
        constant ^= row
        solution ^= label
    results = [solution]
    for label in nullspace:
        results += [value ^ label for value in results]
    return tuple(sorted(results))
