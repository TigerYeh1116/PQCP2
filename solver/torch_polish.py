"""Optional batched exact parity-preserving discrete polish on the same device.

This adapts the even-PACP slides' quantize-then-batched-SA idea to PQCP.
It uses the project's fixed-target multiscale energy, NOT the PACP target.
Swaps change two opposite signs in one sequence at equal-parity positions,
preserving all four content counts. This module is not enabled by default.

Random uniforms are supplied explicitly, avoiding hidden global MPS RNG state.
All per-step correlation updates and acceptance calculations stay on device.
Binary PACF arithmetic is integer-exact in float32 at the project's lengths;
Metropolis probabilities are navigation heuristics, never verification.
"""

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import torch

from .correlation import full_correlation_profile
from .structured_energy import structured_energy_breakdown
from .target_profiles import TargetContentProfile, pair_content, target_content_profiles


def closest_content_target(a: Sequence[int], b: Sequence[int], *,
                           compression: bool = True) -> TargetContentProfile:
    """Choose the lowest-energy valid target for an UNCHANGED binary center.

    All shifts 1..L/2-1 compatible with its four content counts are considered.
    Decimation representatives alone would be unjustified here: replacing k
    by its representative without decimating the fixed center changes distances.
    This is a navigation choice among original Project 2 targets, not pruning.
    No bits are transformed, and no mathematically equivalent solutions are
    manufactured. Ties use dataclass order for deterministic selection.
    """
    profile = full_correlation_profile(a, b)
    content = pair_content(a, b)
    feasible = [p for p in target_content_profiles(len(a), decimation_reduced=False)
                if (p.a_even_ones, p.a_odd_ones, p.b_even_ones, p.b_odd_ones) == content]
    if not feasible:
        raise ValueError("candidate has no compatible DC/Nyquist target")
    weights = ({2: 2, 4: 4} if len(a) % 4 == 0 else {2: 2}) if compression else {}
    return min(feasible, key=lambda p: (
        structured_energy_breakdown(profile, p.k, p.eta, tuple(weights)).weighted_total(weights), p))


def swap_correlation_delta(sequence: torch.Tensor, p: torch.Tensor,
                           q: torch.Tensor) -> torch.Tensor:
    """Exact delta for a simultaneous two-sign flip, with p==q a no-op.

    For u!=0, add each individual flip's -2*x[p]*(x[p+u]+x[p-u]),
    then correct their shared terms by 4*x[p]*x[q] times the number of
    directed p/q edges at u. Both edges count at u=L/2. At u=0 no term
    changes. Inputs: sequence (batch,L), p/q (batch,) device int64 indices.
    This primitive also supports unequal parity or equal signs for testing;
    the search caller restricts actual moves to opposite signs/same parity.
    """
    length = sequence.shape[-1]
    shifts = torch.arange(1, length // 2 + 1, device=sequence.device)[None, :]
    p, q = p[:, None], q[:, None]
    xp, xq = sequence.gather(1, p), sequence.gather(1, q)
    plus_p, minus_p = (p + shifts) % length, (p - shifts) % length
    plus_q, minus_q = (q + shifts) % length, (q - shifts) % length
    half = -2 * xp * (sequence.gather(1, plus_p) + sequence.gather(1, minus_p))
    half = half - 2 * xq * (sequence.gather(1, plus_q) + sequence.gather(1, minus_q))
    shared = (plus_p == q).to(sequence.dtype) + (plus_q == p).to(sequence.dtype)
    half = (half + 4 * xp * xq * shared) * (p != q).to(sequence.dtype)
    zero = torch.zeros_like(xp)
    # For odd lengths the last independent shift has a separate mirror.
    mirror = half[:, :-1] if length % 2 == 0 else half
    return torch.cat((zero, half, mirror.flip(-1)), dim=-1)


def discrete_energy(profile: torch.Tensor, target: torch.Tensor,
                    *, compression: bool = True) -> torch.Tensor:
    """Existing E_full + 2 E_2 + 4 E_4, in exact 1/16 binary units."""
    length = profile.shape[-1]
    residual = profile - target
    value = residual[:, 1:length // 2 + 1].square().sum(-1) / 16
    if compression:
        for factor, weight in ((2, 2), (4, 4)):
            if length % factor == 0:
                value = value + weight * residual.reshape(-1, factor, length // factor).sum(1).square().sum(-1) / 16
    return value


@dataclass
class PolishResult:
    """Best discrete-score candidate per lane, plus auditable work counters."""

    signs: torch.Tensor
    profile: torch.Tensor
    scores: torch.Tensor
    proposals: int
    legal_swaps: int
    accepted_swaps: int
    kicks: int = 0
    forced_swaps: int = 0
    current_signs: Optional[torch.Tensor] = None
    current_profile: Optional[torch.Tensor] = None
    current_energy: Optional[torch.Tensor] = None


@torch.no_grad()
def polish_batch(model, signs: torch.Tensor, uniforms: torch.Tensor, *,
                 initial_temperature: float = 24.6, final_temperature: float = 0.6,
                 compression: bool = True, trace: Optional[list] = None,
                 proposal_policy: str = "positions", kick_interval: int = 0,
                 kick_moves: int = 3) -> PolishResult:
    """Run a bounded device-side SA block, returning every lane's best candidate.

    ``uniforms`` has shape (steps,batch,4), with values in [0,1): sequence/parity
    choice, two positions within that parity, and a Metropolis draw. Equal-sign
    proposals are harmless no-ops (reported separately from legal swaps).
    Bests are selected by exact pqcp_objective, while navigation uses fixed-target
    energy. A PQCP at ANY k is retained even if its proposal is energy-rejected;
    once found that lane freezes for the remainder of this bounded block.
    The caller must independently verify and dedup returned zero-score pairs.
    ``trace`` is only for small correctness tests; production leaves it None.
    The default linear temperature matches the existing C kernel's denominator
    64*(6*(1-step/steps)+0.15), converted to our energy/16 units: 24.6 -> 0.6.
    Shape (steps,batch,candidates,4) compares multiple proposals and navigates
    using the lowest-energy legal one, as in the existing C search. ALL proposals
    are checked for a better exact objective, including non-selected solutions.
    ``proposal_policy='opposite'`` samples a negative and positive member of
    the selected parity directly, using maintained index pools. Empty/full
    parity groups remain no-ops; no rejection loop or CPU per-step work is used.
    The default 'positions' retains the original uniform-position experiment.
    Optional kicks reuse C's three random legal swaps after stagnation, rather
    than introducing a different neighborhood. Kicks are disabled by default.
    """
    import math

    if signs.ndim != 3 or signs.shape[1:] != (2, model.L) or signs.shape[0] != len(model.profiles):
        raise ValueError("polish signs must match the model's batch,2,L")
    if model.L < 4 or model.L % 2:
        raise ValueError("parity-preserving polish requires even L >= 4")
    if uniforms.ndim == 3:
        uniforms = uniforms.unsqueeze(2)
    if (uniforms.ndim != 4 or uniforms.shape[1] != signs.shape[0] or uniforms.shape[-1] != 4
            or uniforms.shape[0] == 0 or uniforms.shape[2] == 0):
        raise ValueError("uniforms must have shape steps,batch[,candidates],4 with positive counts")
    if proposal_policy not in ("positions", "opposite"):
        raise ValueError("proposal_policy must be positions or opposite")
    if (not isinstance(kick_interval, int) or isinstance(kick_interval, bool) or kick_interval < 0
            or not isinstance(kick_moves, int) or isinstance(kick_moves, bool) or kick_moves <= 0):
        raise ValueError("kick interval must be nonnegative and kick moves positive integers")
    if uniforms.device != signs.device or model.targets.device != signs.device:
        raise ValueError("polish model, signs and randomness must share a device")
    if not ((signs == -1) | (signs == 1)).all().item():
        raise ValueError("polish requires binary +/-1 signs")
    if not ((uniforms >= 0) & (uniforms < 1)).all().item():
        raise ValueError("uniforms must be finite and in [0,1)")
    if any(not math.isfinite(t) or t <= 0 for t in (initial_temperature, final_temperature)):
        raise ValueError("temperatures must be finite and positive")
    batch, _, length = signs.shape
    if not torch.equal((signs.reshape(batch, 2, length // 2, 2) == -1).sum(-2), model.weights):
        raise ValueError("initial signs do not match the model's parity content")
    current = signs.clone()
    profile = model.correlation(current)
    energy = discrete_energy(profile, model.targets, compression=compression)
    best, best_profile = current.clone(), profile.clone()
    scores = model.discrete_scores(profile)
    lanes = torch.arange(batch, device=signs.device)
    candidates = uniforms.shape[2]
    targets = model.targets[:, None, :].expand(-1, candidates, -1).reshape(-1, length)
    if proposal_policy == "opposite":
        parity_signs = current.reshape(batch, 2, length // 2, 2).transpose(-1, -2).reshape(batch, 4, length // 2)
        pools = parity_signs.argsort(dim=-1, stable=True)
        weights = model.weights.reshape(batch, 4)
    legal_total = torch.zeros((), device=signs.device, dtype=torch.int64)
    accepted_total = torch.zeros_like(legal_total)
    kicks_total, forced_total = torch.zeros_like(legal_total), torch.zeros_like(legal_total)
    if kick_interval:
        lowest_energy = energy.clone()
        stagnant = torch.zeros(batch, device=signs.device, dtype=torch.int64)
        kicks_left = torch.zeros_like(stagnant)
    for step, draw in enumerate(uniforms):
        group = (draw[..., 0] * 4).to(torch.int64)
        which, parity = group // 2, group % 2
        if proposal_policy == "opposite":
            negatives = weights[lanes[:, None], group]
            negative_rank = (draw[..., 1] * negatives).to(torch.int64).clamp(max=length // 2 - 1)
            positive_rank = (negatives + (draw[..., 2] * (length // 2 - negatives)).to(torch.int64)).clamp(max=length // 2 - 1)
            p = 2 * pools[lanes[:, None], group, negative_rank] + parity
            q = 2 * pools[lanes[:, None], group, positive_rank] + parity
        else:
            p = 2 * (draw[..., 1] * (length // 2)).to(torch.int64) + parity
            q = 2 * (draw[..., 2] * (length // 2)).to(torch.int64) + parity
        sequence = current[lanes[:, None], which]
        xp, xq = sequence.gather(2, p[..., None]).squeeze(-1), sequence.gather(2, q[..., None]).squeeze(-1)
        legal = (xp != xq) & (scores[:, None] != 0)
        delta = swap_correlation_delta(sequence.reshape(-1, length), p.flatten(), q.flatten()).reshape(batch, candidates, length)
        proposed_profile = profile[:, None, :] + delta * legal[..., None]
        proposed_energy = discrete_energy(proposed_profile.reshape(-1, length), targets, compression=compression).reshape(batch, candidates)
        chosen = proposed_energy.masked_fill(~legal, float("inf")).argmin(-1)
        if kick_interval:
            forcing = (kicks_left > 0) & (scores != 0)
            first_legal = legal.to(torch.int64).argmax(-1)
            chosen = torch.where(forcing, first_legal, chosen)
        chosen_energy = proposed_energy[lanes, chosen]
        temperature = initial_temperature + (final_temperature - initial_temperature) * step / len(uniforms)
        change = chosen_energy - energy
        accept = legal[lanes, chosen] & ((change <= 0) | (draw[lanes, chosen, 3] < torch.exp(-change.clamp_min(0) / temperature)))
        if kick_interval:
            accept = accept | (forcing & legal[lanes, chosen])
        # Avoid repeated-index scatter: each separate operation has one index
        # per lane. The equal-position case is rejected by the legal mask.
        flat = current.reshape(batch, 1, 2 * length).expand(-1, candidates, -1)
        proposed = flat.scatter(2, (which * length + p)[..., None], -xp[..., None])
        proposed = proposed.scatter(2, (which * length + q)[..., None], -xq[..., None]).reshape(batch, candidates, 2, length)
        proposed = torch.where(legal[..., None, None], proposed, current[:, None])
        proposed_scores = model.discrete_scores(proposed_profile.reshape(-1, length)).reshape(batch, candidates)
        best_slot = proposed_scores.argmin(-1)
        best_proposal_score = proposed_scores[lanes, best_slot]
        improve = best_proposal_score < scores
        best = torch.where(improve[:, None, None], proposed[lanes, best_slot], best)
        best_profile = torch.where(improve[:, None], proposed_profile[lanes, best_slot], best_profile)
        scores = torch.minimum(scores, best_proposal_score)
        if proposal_policy == "opposite":
            # Swap the two pool entries when the corresponding signs swap:
            # the first 'weight' slots continue to contain exactly the negatives.
            chosen_group = group[lanes, chosen]
            flat_pool = pools.reshape(batch, -1)
            updated_pool = flat_pool.scatter(1, (chosen_group * (length // 2) + negative_rank[lanes, chosen])[:, None],
                                            (q[lanes, chosen] // 2)[:, None])
            updated_pool = updated_pool.scatter(1, (chosen_group * (length // 2) + positive_rank[lanes, chosen])[:, None],
                                               (p[lanes, chosen] // 2)[:, None])
            pools = torch.where(accept[:, None], updated_pool, flat_pool).reshape_as(pools)
        current = torch.where(accept[:, None, None], proposed[lanes, chosen], current)
        profile = torch.where(accept[:, None], proposed_profile[lanes, chosen], profile)
        energy = torch.where(accept, chosen_energy, energy)
        legal_total += legal.sum()
        accepted_total += accept.sum()
        if kick_interval:
            forced = forcing & accept
            forced_total += forced.sum()
            kicks_left -= forced.to(kicks_left.dtype)
            improved_energy = energy < lowest_energy
            lowest_energy = torch.minimum(energy, lowest_energy)
            stagnant = torch.where(improved_energy | forcing, 0, stagnant + 1)
            start_kick = (stagnant > kick_interval) & (kicks_left == 0) & (scores != 0)
            kicks_total += start_kick.sum()
            kicks_left = torch.where(start_kick, kick_moves, kicks_left)
            stagnant = torch.where(start_kick, 0, stagnant)
        if trace is not None:
            trace.append((current.cpu().tolist(), profile.cpu().tolist(), best.cpu().tolist(), best_profile.cpu().tolist()))
    return PolishResult(best, best_profile, scores, batch * candidates * len(uniforms),
                        int(legal_total.item()), int(accepted_total.item()),
                        int(kicks_total.item()), int(forced_total.item()),
                        current, profile, energy)


@torch.no_grad()
def polish_stream(model, signs: torch.Tensor, *, steps: int, candidates: int,
                  seed: int, block_steps: int = 1000,
                  initial_temperature: float = 24.6,
                  final_temperature: float = 0.6,
                  compression: bool = True,
                  proposal_policy: str = "opposite",
                  kick_interval: int = 1000,
                  kick_moves: int = 3,
                  progress: Optional[Callable[[int, int, int], None]] = None) -> PolishResult:
    """Run a deterministic, memory-bounded discrete MPS trajectory.

    Random draws are generated in blocks, while the accepted current state and
    each lane's historical best survive block boundaries.  The earlier API
    allocated ``steps*batch*candidates*4`` floats at once, so a reference-depth
    run could consume more than 0.5 GB before its first move.  The global
    temperature schedule is unchanged by blocking.

    Stagnation counters restart at block boundaries.  Requiring each block to
    cover at least one kick interval prevents accidentally disabling the
    configured escape mechanism.
    """
    if not isinstance(steps, int) or isinstance(steps, bool) or steps <= 0:
        raise ValueError("steps must be a positive integer")
    if not isinstance(candidates, int) or isinstance(candidates, bool) or candidates <= 0:
        raise ValueError("candidates must be a positive integer")
    if not isinstance(block_steps, int) or isinstance(block_steps, bool) or block_steps <= 0:
        raise ValueError("block_steps must be a positive integer")
    if kick_interval and block_steps < kick_interval:
        raise ValueError("block_steps must be at least kick_interval")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    current = signs
    best_signs = signs.clone()
    best_profile = model.correlation(signs)
    best_scores = model.discrete_scores(best_profile)
    current_profile = best_profile
    current_energy = discrete_energy(current_profile, model.targets, compression=compression)
    totals = [0, 0, 0, 0, 0]
    completed = 0
    while completed < steps and not bool((best_scores == 0).any().item()):
        count = min(block_steps, steps - completed)
        start_temperature = initial_temperature + (
            final_temperature - initial_temperature
        ) * completed / steps
        end_temperature = initial_temperature + (
            final_temperature - initial_temperature
        ) * (completed + count) / steps
        draws = torch.rand(
            count, signs.shape[0], candidates, 4, generator=generator,
        ).to(signs.device)
        result = polish_batch(
            model, current, draws,
            initial_temperature=start_temperature,
            final_temperature=end_temperature,
            compression=compression, proposal_policy=proposal_policy,
            kick_interval=kick_interval, kick_moves=kick_moves,
        )
        improve = result.scores < best_scores
        best_signs = torch.where(improve[:, None, None], result.signs, best_signs)
        best_profile = torch.where(improve[:, None], result.profile, best_profile)
        best_scores = torch.minimum(best_scores, result.scores)
        current = result.current_signs
        current_profile = result.current_profile
        current_energy = result.current_energy
        for index, value in enumerate((
            result.proposals, result.legal_swaps, result.accepted_swaps,
            result.kicks, result.forced_swaps,
        )):
            totals[index] += value
        completed += count
        if progress is not None:
            progress(completed, int(best_scores.min().item()), totals[0])
        del draws, result
    return PolishResult(
        best_signs, best_profile, best_scores,
        totals[0], totals[1], totals[2], totals[3], totals[4],
        current, current_profile, current_energy,
    )


__all__ = (
    "PolishResult", "closest_content_target", "discrete_energy", "polish_batch",
    "polish_stream", "swap_correlation_delta",
)
