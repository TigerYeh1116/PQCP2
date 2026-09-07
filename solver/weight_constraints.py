"""Necessary Hamming-weight constraints for even-length binary (L, 4)-PQCPs.

For signs ``x_i=(-1)^a_i`` and ``y_i=(-1)^b_i``, summing every periodic
autocorrelation shift gives ``sum_u S(u) = X^2 + Y^2``, where
``X=L-2 wt(A)`` and ``Y=L-2 wt(B)``.  A valid target has ``S(0)=2L`` and
one symmetric pair of values ``epsilon`` with epsilon in {-4, +4}; hence

    (L/2-wt(A))^2 + (L/2-wt(B))^2 = L/2 +/- 2.

The functions here enumerate every necessary fixed-content family.  FKM
chooses one such family for each restart, and the current SA neighborhood
exchanges a zero and a one within one sequence so both weights remain fixed
until the next restart.
"""

from typing import Tuple


WeightPair = Tuple[int, int]


def admissible_weight_pairs(length: int) -> Tuple[WeightPair, ...]:
    """Return every ordered weight pair compatible with the target total sum.

    An empty result is a proof that no binary (L,4)-PQCP can satisfy the
    current Project 2 target profile at this even length.
    """
    _validate_even_length(length)
    half = length // 2
    targets = (half - 2, half + 2)
    return tuple(
        (weight_a, weight_b)
        for weight_a in range(length + 1)
        for weight_b in range(length + 1)
        if (half - weight_a) ** 2 + (half - weight_b) ** 2 in targets
    )


def canonical_weight_pairs(length: int) -> Tuple[WeightPair, ...]:
    """Return one representative under A/B swap and individual complementation.

    Complementing either binary sequence preserves its periodic
    autocorrelation, and swapping A/B preserves their sum.  Thus restricting
    FKM *initializations* to these representatives loses no PQCP equivalence
    class while avoiding duplicated weight families.
    """
    return tuple(sorted({_canonical_pair(pair, length) for pair in admissible_weight_pairs(length)}))


def is_admissible_weight_pair(length: int, weight_a: int, weight_b: int) -> bool:
    """Test the exact necessary total-autocorrelation identity."""
    return (weight_a, weight_b) in admissible_weight_pairs(length)


def _canonical_pair(pair: WeightPair, length: int) -> WeightPair:
    weight_a, weight_b = pair
    variants = (
        (weight_a, weight_b), (length - weight_a, weight_b),
        (weight_a, length - weight_b), (length - weight_a, length - weight_b),
    )
    return min(tuple(sorted(variant)) for variant in variants)


def _validate_even_length(length: int) -> None:
    if not isinstance(length, int) or isinstance(length, bool) or length <= 0 or length % 2:
        raise ValueError("the PQCP weight identity requires a positive even length")
