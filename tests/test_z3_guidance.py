"""Correctness tests for correlation-guided exact local completion."""

from itertools import product

import pytest

from solver.correlation import full_correlation_profile
from solver.golay import complete_one_flip_each
from solver.verifier import verify_pqcp
from solver.z3_guidance import (
    admissible_short_moves,
    correlation_hamming_lower_bound,
    guided_completion,
)


def _distance(left, right):
    return sum(a != b for a, b in zip(left[0] + left[1], right[0] + right[1]))


def _flip(pair, flips):
    a, b = list(pair[0]), list(pair[1])
    for sequence, position in flips:
        (a if sequence == "a" else b)[position] ^= 1
    return tuple(a), tuple(b)


def test_profile_lower_bound_never_exceeds_exact_small_l_distance():
    length = 4
    all_pairs = [
        (tuple(a), tuple(b))
        for a in product((0, 1), repeat=length)
        for b in product((0, 1), repeat=length)
    ]
    solutions = [pair for pair in all_pairs if verify_pqcp(*pair).is_valid]
    for pair in all_pairs:
        exact = min(_distance(pair, solution) for solution in solutions)
        bound = correlation_hamming_lower_bound(full_correlation_profile(*pair))
        assert bound <= exact


def test_r_bit_profile_delta_is_multiple_of_four_and_bounded_by_four_r():
    center = complete_one_flip_each(8, seed=3)
    assert center is not None
    before = full_correlation_profile(*center)
    flips = (("a", 0), ("a", 3), ("b", 1))
    for radius in range(1, len(flips) + 1):
        after = full_correlation_profile(*_flip(center, flips[:radius]))
        for old, new in zip(before, after):
            delta = new - old
            assert delta % 4 == 0
            assert abs(delta) <= 4 * radius


def test_short_moves_preserve_only_mathematically_admissible_weights():
    solution = complete_one_flip_each(8, seed=3)
    assert solution is not None
    allowed = {(sum(solution[0]), sum(solution[1]))}
    for move in admissible_short_moves(*solution):
        changed = _flip(solution, move)
        from solver.weight_constraints import admissible_weight_pairs
        assert (sum(changed[0]), sum(changed[1])) in admissible_weight_pairs(8)
    assert allowed


def test_guidance_recovers_two_bit_weight_preserving_perturbation():
    solution = complete_one_flip_each(8, seed=3)
    assert solution is not None
    moves = admissible_short_moves(*solution)
    perturbed = _flip(solution, moves[0])
    result = guided_completion(*perturbed, top_k=2)
    assert result.solved
    assert verify_pqcp(result.a, result.b).is_valid


def test_guidance_recovers_four_bit_two_stage_perturbation():
    solution = complete_one_flip_each(8, seed=3)
    assert solution is not None
    first = admissible_short_moves(*solution)[0]
    intermediate = _flip(solution, first)
    occupied = set(first)
    second = next(
        move for move in admissible_short_moves(*intermediate)
        if not occupied.intersection(move) and not verify_pqcp(*_flip(intermediate, move)).is_valid
    )
    perturbed = _flip(intermediate, second)
    result = guided_completion(*perturbed, top_k=50)
    assert result.solved
    assert verify_pqcp(result.a, result.b).is_valid


@pytest.mark.parametrize("top_k", (0, -1, True))
def test_guidance_rejects_invalid_top_k(top_k):
    with pytest.raises(ValueError):
        guided_completion((0, 0, 0, 0), (0, 0, 1, 1), top_k=top_k)
