"""Optional PyTorch gradient guidance for exact same-parity swap repair.

Gradients only rank discrete moves.  Every proposed move is re-evaluated by
the exact incremental correlation implementation, and only a strict decrease
of the fixed-target integer energy is accepted.  This module is experimental
and is not imported by the production pipeline.
"""

from dataclasses import dataclass
from typing import Sequence, Tuple

import torch

from .compression import CorrelationState
from .moves import (
    WeightPreservingSwap, apply_weight_preserving_swap,
    trial_weight_preserving_swap,
)
from .objective import pqcp_objective
from .target_profiles import TargetContentProfile, profile_matches_pair_content
from .torch_relaxation import torch_pair_correlation
from .verifier import verify_pqcp


@dataclass(frozen=True)
class TorchGuidedRepairResult:
    """Best exact candidate reached by gradient-ranked discrete descent."""

    a: Tuple[int, ...]
    b: Tuple[int, ...]
    initial_score: int
    best_score: int
    accepted_steps: int
    gradient_calls: int
    exact_evaluations: int
    verified: bool


def rank_same_parity_swaps(
    a: Sequence[int],
    b: Sequence[int],
    profile: TargetContentProfile,
    limit: int = 32,
) -> Tuple[WeightPreservingSwap, ...]:
    """Rank legal swaps by first-order change in fixed-target squared error."""
    if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
        raise ValueError("limit must be a positive integer")
    pair = tuple(a), tuple(b)
    if not profile_matches_pair_content(profile, *pair):
        raise ValueError("candidate does not match target content profile")
    tensors = [
        torch.tensor(
            [1.0 if bit == 0 else -1.0 for bit in values],
            dtype=torch.float64,
            requires_grad=True,
        )
        for values in pair
    ]
    correlation = torch_pair_correlation(tensors[0], tensors[1])
    desired = torch.zeros(profile.L, dtype=torch.float64)
    desired[profile.k] = desired[profile.L - profile.k] = float(profile.target_value)
    residual = correlation[1:profile.L // 2 + 1] - desired[1:profile.L // 2 + 1]
    torch.sum(residual * residual).backward()

    ranked = []
    for name, values, tensor in zip(("a", "b"), pair, tensors):
        gradient = tensor.grad
        if gradient is None:  # pragma: no cover - autograd contract
            raise RuntimeError("PyTorch did not produce a sign gradient")
        for parity in (0, 1):
            zeros = [index for index in range(parity, profile.L, 2) if values[index] == 0]
            ones = [index for index in range(parity, profile.L, 2) if values[index] == 1]
            for zero in zeros:
                for one in ones:
                    # sign(zero) changes +1 -> -1 and sign(one) -1 -> +1.
                    predicted_delta = -2.0 * float(gradient[zero]) + 2.0 * float(gradient[one])
                    ranked.append((predicted_delta, name, zero, one))
    ranked.sort()
    return tuple(
        WeightPreservingSwap(name, zero, one)
        for _, name, zero, one in ranked[:limit]
    )


def torch_gradient_repair(
    a: Sequence[int],
    b: Sequence[int],
    profile: TargetContentProfile,
    max_steps: int = 20,
    candidate_limit: int = 32,
) -> TorchGuidedRepairResult:
    """Perform deterministic exact descent using PyTorch only for move ranking."""
    if not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps < 0:
        raise ValueError("max_steps must be a non-negative integer")
    state = CorrelationState(a, b)
    if not profile_matches_pair_content(profile, state.a, state.b):
        raise ValueError("candidate does not match target content profile")
    initial_score = best_score = pqcp_objective(state.profile)
    best_a, best_b = state.a, state.b
    accepted = calls = evaluations = 0

    def energy(values: Sequence[int]) -> int:
        target_value = profile.target_value
        return sum(
            (values[shift] - (target_value if shift == profile.k else 0)) ** 2
            for shift in range(1, profile.L // 2 + 1)
        )

    current_energy = energy(state.profile)
    for _ in range(max_steps):
        moves = rank_same_parity_swaps(state.a, state.b, profile, candidate_limit)
        calls += 1
        selected = None
        selected_evaluation = None
        selected_energy = current_energy
        for move in moves:
            evaluation = trial_weight_preserving_swap(state, move)
            evaluations += 1
            candidate_energy = energy(evaluation.profile)
            if candidate_energy < selected_energy:
                selected, selected_evaluation, selected_energy = move, evaluation, candidate_energy
        if selected is None or selected_evaluation is None:
            break
        apply_weight_preserving_swap(state, selected)
        accepted += 1
        current_energy = selected_energy
        score = selected_evaluation.score
        if score < best_score:
            best_a, best_b, best_score = state.a, state.b, score
            if best_score == 0:
                break
    verification = verify_pqcp(best_a, best_b)
    return TorchGuidedRepairResult(
        best_a, best_b, initial_score, best_score, accepted,
        calls, evaluations, verification.is_valid,
    )


def torch_gradient_kick(
    a: Sequence[int],
    b: Sequence[int],
    profile: TargetContentProfile,
    steps: int = 4,
    candidate_limit: int = 32,
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Apply a short gradient-ranked perturbation, allowing exact uphill moves.

    This is an experimental stagnation escape rather than a repair proof.  At
    each step PyTorch supplies a shortlist and the exact fixed-target energy
    chooses its least costly non-tabu state.  Content invariants remain exact.
    """
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 0:
        raise ValueError("steps must be a non-negative integer")
    state = CorrelationState(a, b)
    visited = {(state.a, state.b)}

    def energy(values: Sequence[int]) -> int:
        return sum(
            (values[shift] - (profile.target_value if shift == profile.k else 0)) ** 2
            for shift in range(1, profile.L // 2 + 1)
        )

    for _ in range(steps):
        options = []
        for move in rank_same_parity_swaps(state.a, state.b, profile, candidate_limit):
            evaluation = trial_weight_preserving_swap(state, move)
            # Materialize only the two exchanged bits to detect immediate
            # cycles without mutating the exact correlation state.
            values_a, values_b = list(state.a), list(state.b)
            values = values_a if move.sequence == "a" else values_b
            values[move.zero_position], values[move.one_position] = 1, 0
            candidate_pair = tuple(values_a), tuple(values_b)
            if candidate_pair not in visited:
                options.append((energy(evaluation.profile), move, candidate_pair))
        if not options:
            break
        _, selected, candidate_pair = min(options, key=lambda item: item[0])
        apply_weight_preserving_swap(state, selected)
        visited.add(candidate_pair)
    return state.a, state.b
