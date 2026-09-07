"""Optional PyTorch relaxation for fixed-weight PQCP candidate refinement.

This module is deliberately separate from the exact solver.  It represents
binary signs by ``tanh(theta / temperature)`` and differentiates through the
Project 2 periodic autocorrelation.  At every observation point the logits
are projected to exactly ``weight_a`` and ``weight_b`` bits, so every returned
candidate is binary and belongs to the same necessary content family as its
input.  Final scores are always recomputed by the pure-Python implementation.

PyTorch is an optional dependency: importing the rest of ``solver`` does not
import this module or require torch.
"""

from dataclasses import dataclass
import random
from typing import List, Optional, Sequence, Tuple

import torch

from .correlation import full_correlation_profile, normalize_binary_sequence
from .objective import pqcp_objective
from .verifier import verify_pqcp
from .target_profiles import TargetContentProfile, profile_matches_pair_content


BinaryPair = Tuple[Tuple[int, ...], Tuple[int, ...]]


@dataclass(frozen=True)
class RelaxationParameters:
    """Controls for one deterministic CPU continuous-relaxation run."""

    steps: int = 400
    learning_rate: float = 0.04
    initial_logit: float = 1.25
    initial_temperature: float = 1.5
    final_temperature: float = 0.25
    binary_penalty: float = 0.25
    weight_penalty: float = 2.0
    observation_interval: int = 5
    jitter: float = 0.05
    seed: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.steps, int) or isinstance(self.steps, bool) or self.steps < 0:
            raise ValueError("steps must be a non-negative integer")
        if self.learning_rate <= 0 or self.initial_logit <= 0:
            raise ValueError("learning_rate and initial_logit must be positive")
        if self.initial_temperature <= 0 or self.final_temperature <= 0:
            raise ValueError("temperatures must be positive")
        if self.binary_penalty < 0 or self.weight_penalty < 0 or self.jitter < 0:
            raise ValueError("penalties and jitter must be non-negative")
        if not isinstance(self.observation_interval, int) or self.observation_interval <= 0:
            raise ValueError("observation_interval must be a positive integer")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError("seed must be an integer")


@dataclass(frozen=True)
class RelaxationResult:
    """Best exact projected candidate observed during continuous optimization."""

    best_a: Tuple[int, ...]
    best_b: Tuple[int, ...]
    best_profile: Tuple[int, ...]
    best_score: int
    initial_score: int
    steps: int
    verified: bool
    score_history: Tuple[int, ...]
    selected_shift_history: Tuple[int, ...]


def torch_pair_correlation(sign_a: torch.Tensor, sign_b: torch.Tensor) -> torch.Tensor:
    """Return differentiable periodic pair autocorrelations for all shifts.

    ``sign_a`` and ``sign_b`` are one-dimensional floating tensors.  The
    formula is exactly ``sum_i x_i*x_(i+u) + sum_i y_i*y_(i+u)``.
    """
    if sign_a.ndim != 1 or sign_b.ndim != 1 or sign_a.shape != sign_b.shape:
        raise ValueError("sign tensors must be one-dimensional and have equal shape")
    if sign_a.numel() == 0:
        raise ValueError("sign tensors must be non-empty")
    return torch.stack([
        torch.sum(sign_a * torch.roll(sign_a, shifts=-shift))
        + torch.sum(sign_b * torch.roll(sign_b, shifts=-shift))
        for shift in range(sign_a.numel())
    ])


def project_logits_fixed_weight(logits: torch.Tensor, weight: int) -> Tuple[int, ...]:
    """Project logits to bits with exactly ``weight`` ones.

    Bit one denotes sign -1, so the ``weight`` smallest sign logits become
    ones.  Stable sorting makes ties deterministic.
    """
    if logits.ndim != 1 or logits.numel() == 0:
        raise ValueError("logits must be a non-empty one-dimensional tensor")
    length = logits.numel()
    if not isinstance(weight, int) or isinstance(weight, bool) or not 0 <= weight <= length:
        raise ValueError("weight must be in 0..L")
    order = torch.argsort(logits.detach(), stable=True).tolist()
    selected = set(order[:weight])
    return tuple(1 if index in selected else 0 for index in range(length))


def project_logits_target_content(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    profile: TargetContentProfile,
) -> BinaryPair:
    """Project four parity classes to one exact target-content profile.

    Bit one denotes sign ``-1``.  Selecting the smallest logits independently
    within A-even, A-odd, B-even, and B-odd therefore preserves both ordinary
    and alternating sign sums required by ``profile``.
    """
    if logits_a.ndim != 1 or logits_b.ndim != 1 or logits_a.shape != logits_b.shape:
        raise ValueError("logits must be equal one-dimensional tensors")
    if logits_a.numel() != profile.L:
        raise ValueError("logit length must equal profile.L")

    def project(values: torch.Tensor, even_weight: int, odd_weight: int) -> Tuple[int, ...]:
        result = [0] * profile.L
        for parity, weight in ((0, even_weight), (1, odd_weight)):
            positions = list(range(parity, profile.L, 2))
            ordered = sorted(
                positions,
                key=lambda index: (float(values[index].detach()), index),
            )
            for index in ordered[:weight]:
                result[index] = 1
        return tuple(result)

    return (
        project(logits_a, profile.a_even_ones, profile.a_odd_ones),
        project(logits_b, profile.b_even_ones, profile.b_odd_ones),
    )


def relax_candidate_for_profile(
    a: Sequence[int],
    b: Sequence[int],
    profile: TargetContentProfile,
    parameters: Optional[RelaxationParameters] = None,
) -> RelaxationResult:
    """Refine a candidate inside one exact Project 2 target-content subproblem.

    Unlike :func:`relax_candidate`, this loss does not switch between target
    shifts or signs.  Every observed binary point is projected to the four
    exact parity counts of ``profile``.  Thus it can be handed back to the
    same same-parity SA trajectory or to profile-constrained Z3 without
    changing the subproblem.  Continuous tensors are never accepted as
    solutions; pure-Python correlation and the independent verifier remain
    authoritative.
    """
    params = parameters or RelaxationParameters()
    bits_a = normalize_binary_sequence(a)
    bits_b = normalize_binary_sequence(b)
    if len(bits_a) != len(bits_b) or len(bits_a) != profile.L:
        raise ValueError("a and b must both have profile.L bits")
    if not profile_matches_pair_content(profile, bits_a, bits_b):
        raise ValueError("input candidate does not match target content profile")

    torch.manual_seed(params.seed)
    rng = random.Random(params.seed)
    dtype = torch.float64
    initial_a = [params.initial_logit * (1.0 if bit == 0 else -1.0) for bit in bits_a]
    initial_b = [params.initial_logit * (1.0 if bit == 0 else -1.0) for bit in bits_b]
    if params.jitter:
        initial_a = [value + rng.uniform(-params.jitter, params.jitter) for value in initial_a]
        initial_b = [value + rng.uniform(-params.jitter, params.jitter) for value in initial_b]
    theta_a = torch.tensor(initial_a, dtype=dtype, requires_grad=True)
    theta_b = torch.tensor(initial_b, dtype=dtype, requires_grad=True)
    optimizer = torch.optim.Adam((theta_a, theta_b), lr=params.learning_rate)

    desired = torch.zeros(profile.L, dtype=dtype)
    desired[0] = float(2 * profile.L)
    desired[profile.k] = desired[profile.L - profile.k] = float(profile.target_value)
    best_a, best_b = bits_a, bits_b
    best_profile = tuple(full_correlation_profile(best_a, best_b))
    initial_score = best_score = pqcp_objective(best_profile)
    scores: List[int] = [best_score]

    parity_targets = (
        profile.a_even_ones, profile.a_odd_ones,
        profile.b_even_ones, profile.b_odd_ones,
    )
    for step in range(params.steps):
        fraction = step / max(1, params.steps - 1)
        temperature = params.initial_temperature * (
            params.final_temperature / params.initial_temperature
        ) ** fraction
        sign_a = torch.tanh(theta_a / temperature)
        sign_b = torch.tanh(theta_b / temperature)
        continuous_profile = torch_pair_correlation(sign_a, sign_b)
        # Symmetry means shifts above L/2 duplicate information.  Retaining
        # u=0 also pushes relaxed signs toward the correct total norm.
        residual = continuous_profile[:profile.L // 2 + 1] - desired[:profile.L // 2 + 1]
        target_loss = torch.mean(residual * residual)
        one_prob_a = (1.0 - sign_a) / 2.0
        one_prob_b = (1.0 - sign_b) / 2.0
        parity_sums = (
            torch.sum(one_prob_a[0::2]), torch.sum(one_prob_a[1::2]),
            torch.sum(one_prob_b[0::2]), torch.sum(one_prob_b[1::2]),
        )
        content_loss = sum(
            (value - target) ** 2
            for value, target in zip(parity_sums, parity_targets)
        ) / profile.L
        binary_loss = torch.mean((1.0 - sign_a * sign_a) ** 2) + torch.mean(
            (1.0 - sign_b * sign_b) ** 2
        )
        loss = target_loss + params.weight_penalty * content_loss + params.binary_penalty * binary_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % params.observation_interval == 0 or step + 1 == params.steps:
            candidate_a, candidate_b = project_logits_target_content(theta_a, theta_b, profile)
            candidate_profile = tuple(full_correlation_profile(candidate_a, candidate_b))
            candidate_score = pqcp_objective(candidate_profile)
            scores.append(candidate_score)
            if candidate_score < best_score:
                best_a, best_b = candidate_a, candidate_b
                best_profile, best_score = candidate_profile, candidate_score
                if best_score == 0:
                    break

    verification = verify_pqcp(best_a, best_b)
    if tuple(verification.profile) != best_profile:
        raise RuntimeError("profile-aware PyTorch disagrees with exact verifier profile")
    return RelaxationResult(
        best_a, best_b, best_profile, best_score, initial_score,
        step + 1 if params.steps else 0, verification.is_valid,
        tuple(scores), tuple(profile.k for _ in scores[1:]),
    )


def relax_candidate_batch_for_profile(
    candidates: Sequence[BinaryPair],
    profile: TargetContentProfile,
    parameters: Optional[RelaxationParameters] = None,
) -> RelaxationResult:
    """Vectorize several independent profile-aware relaxations on one CPU tensor.

    This is intended for restart seeding: batching amortizes autograd and PACF
    construction overhead that dominates a single length-44 tensor.  Every
    row must already satisfy the same exact content profile, and every row is
    independently projected before it can compete for the returned best.
    """
    params = parameters or RelaxationParameters()
    if not candidates:
        raise ValueError("candidates must be non-empty")
    normalized = tuple(
        (normalize_binary_sequence(a), normalize_binary_sequence(b))
        for a, b in candidates
    )
    if any(len(a) != profile.L or len(b) != profile.L for a, b in normalized):
        raise ValueError("every candidate must have profile.L bits")
    if any(not profile_matches_pair_content(profile, a, b) for a, b in normalized):
        raise ValueError("every candidate must match the target content profile")

    torch.manual_seed(params.seed)
    generator = torch.Generator().manual_seed(params.seed)
    dtype = torch.float64
    base_a = torch.tensor([
        [params.initial_logit * (1.0 if bit == 0 else -1.0) for bit in a]
        for a, _ in normalized
    ], dtype=dtype)
    base_b = torch.tensor([
        [params.initial_logit * (1.0 if bit == 0 else -1.0) for bit in b]
        for _, b in normalized
    ], dtype=dtype)
    if params.jitter:
        base_a += (2.0 * torch.rand(base_a.shape, generator=generator, dtype=dtype) - 1.0) * params.jitter
        base_b += (2.0 * torch.rand(base_b.shape, generator=generator, dtype=dtype) - 1.0) * params.jitter
    theta_a = base_a.requires_grad_()
    theta_b = base_b.requires_grad_()
    optimizer = torch.optim.Adam((theta_a, theta_b), lr=params.learning_rate)

    exact_rows = [
        (a, b, tuple(full_correlation_profile(a, b))) for a, b in normalized
    ]
    initial_scores = [pqcp_objective(row[2]) for row in exact_rows]
    best_index = min(range(len(exact_rows)), key=lambda index: initial_scores[index])
    best_a, best_b, best_profile = exact_rows[best_index]
    initial_score = best_score = initial_scores[best_index]
    scores: List[int] = [best_score]
    desired = torch.zeros(profile.L, dtype=dtype)
    desired[0] = float(2 * profile.L)
    desired[profile.k] = desired[profile.L - profile.k] = float(profile.target_value)
    parity_targets = torch.tensor([
        profile.a_even_ones, profile.a_odd_ones,
        profile.b_even_ones, profile.b_odd_ones,
    ], dtype=dtype)

    for step in range(params.steps):
        fraction = step / max(1, params.steps - 1)
        temperature = params.initial_temperature * (
            params.final_temperature / params.initial_temperature
        ) ** fraction
        sign_a = torch.tanh(theta_a / temperature)
        sign_b = torch.tanh(theta_b / temperature)
        correlations = _torch_pair_correlation_batch(sign_a, sign_b)
        residual = correlations[:, :profile.L // 2 + 1] - desired[:profile.L // 2 + 1]
        target_loss = torch.mean(residual * residual, dim=1)
        one_a, one_b = (1.0 - sign_a) / 2.0, (1.0 - sign_b) / 2.0
        parity_sums = torch.stack((
            torch.sum(one_a[:, 0::2], dim=1), torch.sum(one_a[:, 1::2], dim=1),
            torch.sum(one_b[:, 0::2], dim=1), torch.sum(one_b[:, 1::2], dim=1),
        ), dim=1)
        content_loss = torch.sum((parity_sums - parity_targets) ** 2, dim=1) / profile.L
        binary_loss = torch.mean((1.0 - sign_a * sign_a) ** 2, dim=1) + torch.mean(
            (1.0 - sign_b * sign_b) ** 2, dim=1
        )
        loss = torch.sum(
            target_loss + params.weight_penalty * content_loss
            + params.binary_penalty * binary_loss
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % params.observation_interval == 0 or step + 1 == params.steps:
            for row in range(len(normalized)):
                candidate_a, candidate_b = project_logits_target_content(
                    theta_a[row], theta_b[row], profile
                )
                candidate_profile = tuple(full_correlation_profile(candidate_a, candidate_b))
                candidate_score = pqcp_objective(candidate_profile)
                if candidate_score < best_score:
                    best_a, best_b = candidate_a, candidate_b
                    best_profile, best_score = candidate_profile, candidate_score
            scores.append(best_score)
            if best_score == 0:
                break

    verification = verify_pqcp(best_a, best_b)
    if tuple(verification.profile) != best_profile:
        raise RuntimeError("batched PyTorch disagrees with exact verifier profile")
    return RelaxationResult(
        best_a, best_b, best_profile, best_score, initial_score,
        step + 1 if params.steps else 0, verification.is_valid,
        tuple(scores), tuple(profile.k for _ in scores[1:]),
    )


def _torch_pair_correlation_batch(sign_a: torch.Tensor, sign_b: torch.Tensor) -> torch.Tensor:
    """Return differentiable PACF profiles for tensors shaped ``(batch,L)``."""
    if sign_a.ndim != 2 or sign_b.shape != sign_a.shape:
        raise ValueError("batched sign tensors must have equal (batch,L) shape")
    return torch.stack([
        torch.sum(sign_a * torch.roll(sign_a, shifts=-shift, dims=1), dim=1)
        + torch.sum(sign_b * torch.roll(sign_b, shifts=-shift, dims=1), dim=1)
        for shift in range(sign_a.shape[1])
    ], dim=1)


def relax_candidate(
    a: Sequence[int],
    b: Sequence[int],
    parameters: Optional[RelaxationParameters] = None,
) -> RelaxationResult:
    """Refine one complete pair through a continuous fixed-content relaxation.

    The differentiable target is a soft minimum over every legal target
    profile: one symmetric nonzero orbit with common value +4 or -4.  A
    binarization penalty discourages fractional signs and a conservative
    content penalty keeps the continuous iterate close to the input weights.
    Exact fixed-weight projection is observed throughout; only projected
    binary candidates can become ``best``.
    """
    params = parameters or RelaxationParameters()
    bits_a = normalize_binary_sequence(a)
    bits_b = normalize_binary_sequence(b)
    if len(bits_a) != len(bits_b):
        raise ValueError("a and b must have equal lengths")
    length = len(bits_a)
    weight_a, weight_b = sum(bits_a), sum(bits_b)

    torch.manual_seed(params.seed)
    rng = random.Random(params.seed)
    dtype = torch.float64
    initial_a = [params.initial_logit * (1.0 if bit == 0 else -1.0) for bit in bits_a]
    initial_b = [params.initial_logit * (1.0 if bit == 0 else -1.0) for bit in bits_b]
    if params.jitter:
        initial_a = [value + rng.uniform(-params.jitter, params.jitter) for value in initial_a]
        initial_b = [value + rng.uniform(-params.jitter, params.jitter) for value in initial_b]
    theta_a = torch.tensor(initial_a, dtype=dtype, requires_grad=True)
    theta_b = torch.tensor(initial_b, dtype=dtype, requires_grad=True)
    optimizer = torch.optim.Adam((theta_a, theta_b), lr=params.learning_rate)

    best_a, best_b = bits_a, bits_b
    best_profile = tuple(full_correlation_profile(best_a, best_b))
    initial_score = best_score = pqcp_objective(best_profile)
    scores: List[int] = [best_score]
    selected_shifts: List[int] = []

    target_profiles = _target_profiles(length, dtype)
    for step in range(params.steps):
        fraction = step / max(1, params.steps - 1)
        temperature = params.initial_temperature * (
            params.final_temperature / params.initial_temperature
        ) ** fraction
        sign_a = torch.tanh(theta_a / temperature)
        sign_b = torch.tanh(theta_b / temperature)
        profile = torch_pair_correlation(sign_a, sign_b)
        # Normalize by L: loss scale and learning rate remain comparable for
        # L=4 through the official Project lengths.
        errors = torch.mean((target_profiles - profile.unsqueeze(0)) ** 2, dim=1)
        # Smooth minimum early, increasingly sharp as signs cool.
        softmin_temperature = max(0.05, 2.0 * (1.0 - fraction) + 0.05)
        target_loss = -softmin_temperature * torch.logsumexp(
            -errors / softmin_temperature, dim=0
        )
        continuous_weight_a = torch.sum((1.0 - sign_a) / 2.0)
        continuous_weight_b = torch.sum((1.0 - sign_b) / 2.0)
        weight_loss = (
            (continuous_weight_a - weight_a) ** 2
            + (continuous_weight_b - weight_b) ** 2
        ) / length
        binary_loss = torch.mean((1.0 - sign_a * sign_a) ** 2) + torch.mean(
            (1.0 - sign_b * sign_b) ** 2
        )
        loss = target_loss + params.weight_penalty * weight_loss + params.binary_penalty * binary_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % params.observation_interval == 0 or step + 1 == params.steps:
            candidate_a = project_logits_fixed_weight(theta_a, weight_a)
            candidate_b = project_logits_fixed_weight(theta_b, weight_b)
            candidate_profile = tuple(full_correlation_profile(candidate_a, candidate_b))
            candidate_score = pqcp_objective(candidate_profile)
            scores.append(candidate_score)
            selected_shifts.append(_closest_target_shift(candidate_profile))
            if candidate_score < best_score:
                best_a, best_b = candidate_a, candidate_b
                best_profile, best_score = candidate_profile, candidate_score
                if best_score == 0:
                    break

    verification = verify_pqcp(best_a, best_b)
    if tuple(verification.profile) != best_profile:
        raise RuntimeError("PyTorch candidate disagrees with independent full profile")
    return RelaxationResult(
        best_a, best_b, best_profile, best_score, initial_score,
        step + 1 if params.steps else 0, verification.is_valid,
        tuple(scores), tuple(selected_shifts),
    )


def _target_profiles(length: int, dtype: torch.dtype) -> torch.Tensor:
    """Build all Project 2 target profiles, counting actual shift indices."""
    targets = []
    for shift in range(1, (length + 1) // 2):
        if shift == length - shift:
            continue
        for value in (-4.0, 4.0):
            target = [0.0] * length
            target[0] = float(2 * length)
            target[shift] = target[length - shift] = value
            targets.append(target)
    if not targets:
        # Small degenerate lengths have no two-distinct-shift target orbit.
        target = [0.0] * length
        target[0] = float(2 * length)
        targets.append(target)
    return torch.tensor(targets, dtype=dtype)


def _closest_target_shift(profile: Sequence[int]) -> int:
    """Return representative shift of the closest legal squared-error target."""
    representatives = range(1, (len(profile) + 1) // 2)
    return max(representatives, key=lambda shift: abs(profile[shift]), default=0)
