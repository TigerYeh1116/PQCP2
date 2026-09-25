"""Exact necessary orientation equations for factor-2 lifts with odd half-length.

For L=2d, c[i]=x[i]+x[i+d] and h[i]=(x[i]-x[i+d])/2.
Then P_x(u)=P_c(u)/2+2*N_h(u), where N is NEGACYCLIC:
wrapping around length d changes the sign. Thus the pair residual is
R[u]=(T[u]-T[u+d])/4. For odd d, N(d-u)=-N(u), so shifts
1..(d-1)/2 suffice. A zero of c contributes h[i]=(-1)**b[i];
all other h entries are zero. Original orientation bits refer to index i,
NOT adjacent-pair expansion or the reordered CRT representation.

If M signed products sum to R, exactly (M-R)/2 are negative.
Taking their product gives a necessary XOR equation on the b variables,
including the known negative wrap signs. Gaussian elimination is exact.
Consistency is NOT a solution: full residual and content constraints must
still be enforced. No search, known solutions, floating point, or Z3 here.
"""

from dataclasses import dataclass
from typing import Sequence, Tuple


@dataclass(frozen=True)
class OrientationSystem:
    variables: Tuple[Tuple[int, int], ...]
    equations: Tuple[Tuple[int, int, int, int], ...]  # shift, mask, rhs, terms
    basis: Tuple[Tuple[int, int, int], ...]  # pivot, mask, rhs
    consistent: bool

    @property
    def rank(self) -> int:
        return len(self.basis)

    @property
    def free_columns(self) -> Tuple[int, ...]:
        pivots = {pivot for pivot, _, _ in self.basis}
        return tuple(i for i in range(len(self.variables)) if i not in pivots)

    def expressions(self) -> Tuple[Tuple[int, int], ...]:
        """Return each original bit as (free-bit XOR mask, constant).

        This parameterizes ALL solutions to the necessary linear system,
        not a sampled subset. The mapping is bijective on free assignments.
        """
        if not self.consistent:
            raise ValueError("inconsistent orientation system")
        result = [(0, 0)] * len(self.variables)
        for i, column in enumerate(self.free_columns):
            result[column] = (1 << i, 0)
        for pivot, row, rhs in self.basis:
            mask, constant = 0, rhs
            rest = row ^ (1 << pivot)
            while rest:
                bit = rest & -rest
                lower_mask, lower_constant = result[bit.bit_length() - 1]
                mask ^= lower_mask
                constant ^= lower_constant
                rest ^= bit
            result[pivot] = mask, constant
        return tuple(result)


def odd_orientation_system(
    compressed_a: Sequence[int], compressed_b: Sequence[int], residual: Sequence[int],
) -> OrientationSystem:
    """Build conservative mod-4 constraints for an odd-d negacyclic lift.

    Caller must separately check that the compressed pair matches folded T.
    Even-half lengths are deliberately left to the existing implementation.
    """
    ca, cb, target = tuple(compressed_a), tuple(compressed_b), tuple(residual)
    d = len(ca)
    if (d < 3 or d % 2 == 0 or len(cb) != d or len(target) != d
            or any(type(v) is not int or v not in (-2, 0, 2) for v in ca + cb)
            or any(type(v) is not int for v in target)):
        raise ValueError("odd equal half-length >= 3, ternary sums and integer residual required")
    variables = tuple((s, i) for s, word in enumerate((ca, cb))
                      for i, value in enumerate(word) if value == 0)
    ids = {v: i for i, v in enumerate(variables)}
    basis = {}
    equations = []
    consistent = True
    for u in range(1, (d + 1) // 2):
        mask = terms = wraps = 0
        for s, word in enumerate((ca, cb)):
            for i in range(d):
                j = (i + u) % d
                if word[i] == word[j] == 0:
                    terms += 1
                    wraps ^= int(i + u >= d)
                    mask ^= (1 << ids[s, i]) ^ (1 << ids[s, j])
        wanted = target[u]
        if abs(wanted) > terms or (terms - wanted) % 2:
            consistent = False
            break
        rhs = (((terms - wanted) // 2) & 1) ^ wraps
        equations.append((u, mask, rhs, terms))
        while mask:
            pivot = mask.bit_length() - 1
            if pivot not in basis:
                basis[pivot] = mask, rhs
                break
            other, bit = basis[pivot]
            mask ^= other
            rhs ^= bit
        if not mask and rhs:
            consistent = False
            break
    return OrientationSystem(variables, tuple(equations),
                             tuple((p, *basis[p]) for p in sorted(basis)), consistent)
