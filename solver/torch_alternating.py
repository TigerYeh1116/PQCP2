"""Experimental odd-half compression/direction alternating GPU polish.

For L=2d with d odd, each antipodal pair has one even and one odd index.
Choose two unequal antipodal pairs in the SAME sequence, with opposite signs
at their even endpoints. Swap those even endpoints AND their odd partners.
This four-bit move preserves each antipodal sum and both parity weights.
The partner swap delta must use the intermediate sequence: adding two deltas
computed on the old word would miss shared edges. All deltas use the existing
exact two-flip primitive; the existing acceptance/energy/verifier are retained.

Support phases use ordinary parity-preserving swaps and may change compression.
This is a heuristic navigation path, not pruning or a completeness claim.
"""

from typing import Optional, Tuple, TYPE_CHECKING
from threading import Event

import torch

if TYPE_CHECKING:
    from .torch_polish import PolishResult
    from .torch_search import BatchedPQCP


@torch.no_grad()
def orientation_proposals(current: torch.Tensor, draw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (proposed pairs, exact PACF deltas, legal mask) on the device.

    current is batch,2,L and draw is batch,candidates,4, all on one device.
    Input validation is done by polish_batch. Sampling is without rejection
    loops; absent positive/negative eligible endpoints yield safe no-ops.
    """
    from .torch_polish import swap_correlation_delta
    batch, _, length = current.shape
    half = length // 2
    if length % 4 != 2:
        raise ValueError("orientation moves require odd half-length")
    even = 2 * torch.arange(half, device=current.device)
    partners = (even + half) % length
    endpoints, opposites = current[..., even], current[..., partners]
    unequal = endpoints != opposites
    negative = unequal & (endpoints == -1)
    positive = unequal & (endpoints == 1)
    negatives, positives = negative.sum(-1), positive.sum(-1)
    neg_pool = (~negative).to(torch.int64).argsort(dim=-1, stable=True)
    pos_pool = (~positive).to(torch.int64).argsort(dim=-1, stable=True)
    lanes = torch.arange(batch, device=current.device)[:, None]
    which = (draw[..., 0] * 2).to(torch.int64)
    nr = (draw[..., 1] * negatives[lanes, which]).to(torch.int64)
    pr = (draw[..., 2] * positives[lanes, which]).to(torch.int64)
    p, q = 2 * neg_pool[lanes, which, nr], 2 * pos_pool[lanes, which, pr]
    pp, qq = (p + half) % length, (q + half) % length
    legal = (negatives[lanes, which] > 0) & (positives[lanes, which] > 0)
    sequence = current[lanes, which]
    flat = sequence.reshape(-1, length)
    first = swap_correlation_delta(flat, p.flatten(), q.flatten())
    intermediate = flat.scatter(1, p.flatten()[:, None], -flat.gather(1, p.flatten()[:, None]))
    intermediate = intermediate.scatter(1, q.flatten()[:, None], -flat.gather(1, q.flatten()[:, None]))
    second = swap_correlation_delta(intermediate, pp.flatten(), qq.flatten())
    delta = (first + second).reshape(batch, draw.shape[1], length)
    candidates = current[:, None].expand(-1, draw.shape[1], -1, -1).reshape(batch, draw.shape[1], 2 * length)
    proposed = candidates
    for position in (p, q, pp, qq):
        index = (which * length + position)[..., None]
        proposed = proposed.scatter(2, index, -candidates.gather(2, index))
    proposed = proposed.reshape(batch, draw.shape[1], 2, length)
    return (torch.where(legal[..., None, None], proposed, current[:, None]),
            delta * legal[..., None], legal)


def fold_error(profile: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Unweighted exact squared factor-2 residual (navigation, not a new target)."""
    length = profile.shape[-1]
    return (profile - target).reshape(-1, 2, length // 2).sum(1).square().sum(-1)


@torch.no_grad()
def alternating_polish_stream(model: "BatchedPQCP", signs: torch.Tensor, *,
                              steps: int, candidates: int, seed: int,
                              phase_steps: int = 32, initial_temperature: float = 24.6,
                              final_temperature: float = 0.6, trace: Optional[list] = None,
                              stop_event: Optional[Event] = None) -> "PolishResult":
    """Bounded deterministic alternation; no hidden RNG and no disk access.

    Trace, when requested, contains phase-labelled block traces for tests only.
    The best objective AND best compressed-error candidates survive phases.
    The latter prevents a useful exact compression being lost to score ranking.
    Cancellation is checked between bounded phases; this is not a checkpoint
    format or a bit-exact resume facility for an interrupted polish block.
    """
    from .torch_polish import PolishResult, discrete_energy, polish_batch
    if model.L % 4 != 2:
        raise ValueError("alternating polish requires L=2 mod 4")
    if any(type(v) is not int or v <= 0 for v in (steps, candidates, phase_steps)):
        raise ValueError("steps, candidates and phase_steps must be positive integers")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    current = signs.clone()
    best = signs.clone()
    best_profile = model.correlation(signs)
    scores = model.discrete_scores(best_profile)
    compressed, compressed_profile = best.clone(), best_profile.clone()
    compressed_error = fold_error(best_profile, model.targets)
    compressed_energy = discrete_energy(best_profile, model.targets)
    current_profile, energy = best_profile, compressed_energy
    totals = [0, 0, 0, 0, 0]
    direction_proposals = direction_accepted = completed = 0
    while completed < steps and not bool((scores == 0).any().item()):
        if stop_event is not None and stop_event.is_set():
            break
        count = min(phase_steps, steps - completed)
        direction = (completed // phase_steps) % 2 == 1
        draws = torch.rand(count, signs.shape[0], candidates, 4, generator=generator).to(signs.device)
        block_trace = [] if trace is not None else None
        start = initial_temperature + (final_temperature - initial_temperature) * completed / steps
        end = initial_temperature + (final_temperature - initial_temperature) * (completed + count) / steps
        result = polish_batch(model, current, draws, initial_temperature=start,
                              final_temperature=end, proposal_policy="orientation" if direction else "opposite",
                              kick_interval=0, track_compression=True, trace=block_trace)
        improve = result.scores < scores
        best = torch.where(improve[:, None, None], result.signs, best)
        best_profile = torch.where(improve[:, None], result.profile, best_profile)
        scores = torch.minimum(scores, result.scores)
        error = fold_error(result.compression_profile, model.targets)
        comp_energy = discrete_energy(result.compression_profile, model.targets)
        better = (error < compressed_error) | ((error == compressed_error) & (comp_energy < compressed_energy))
        compressed = torch.where(better[:, None, None], result.compression_signs, compressed)
        compressed_profile = torch.where(better[:, None], result.compression_profile, compressed_profile)
        compressed_error = torch.where(better, error, compressed_error)
        compressed_energy = torch.where(better, comp_energy, compressed_energy)
        current, current_profile, energy = result.current_signs, result.current_profile, result.current_energy
        for i, value in enumerate((result.proposals, result.legal_swaps, result.accepted_swaps,
                                   result.kicks, result.forced_swaps)):
            totals[i] += value
        if direction:
            direction_proposals += result.proposals
            direction_accepted += result.accepted_swaps
        if trace is not None:
            trace.append(("direction" if direction else "support", block_trace))
        completed += count
    return PolishResult(best, best_profile, scores, *totals, current, current_profile, energy,
                        compression_signs=compressed, compression_profile=compressed_profile,
                        direction_proposals=direction_proposals, direction_accepted=direction_accepted)
