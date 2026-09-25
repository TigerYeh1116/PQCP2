"""Experimental direct ternary search for L=2d, d odd, on CPU/MPS/CUDA.

Work with v[i]=(x[i]+x[i+d])/2 in {-1,0,1}. Exact compression requires
P_va(r)+P_vb(r)=(T(r)+T(r+d))/4. At r=0 this forces exactly d nonzero
symbols across the two length-d words. For a side with e,o binary ones,
n=count(-1), z=e+o-2n and p=d-n-z. A lift with the prescribed parity
content exists iff 0<=e-n<=z (and the analogous odd count). Since d is odd,
every antipodal pair contains one index of each parity; this criterion is
independent of symbol positions. We enumerate ALL such paired contents,
then sample them across restarts. No single arbitrary content is imposed.

This is bounded batched discrete optimization, NOT a new continuous loss or
an exhaustive proof. Swapping ternary symbols keeps the assigned content.
Float32 here represents exact small integers (project L<=94); exported
matches are always recomputed with the independent integer implementation.
Neither a low energy nor an exact compressed match is a PQCP certificate.
"""

from dataclasses import dataclass
import random
from threading import Event
from typing import List, Optional, Sequence, Tuple

import torch

from .compressed_completion import CompressedLiftJob
from .compressed_fkm import factor2_compressed_contents, lift_factor2_compressed
from .structured_energy import fold_profile, integer_periodic_autocorrelation_profile, target_profile
from .target_profiles import TargetContentProfile, pair_content, target_content_profiles

Content = Tuple[int, int, int]


def paired_contents(target: TargetContentProfile) -> Tuple[Tuple[Content, Content], ...]:
    """All lift-feasible paired ternary contents obeying S(L/2)=0 exactly."""
    if target.L < 6 or target.L > 94 or target.L % 4 != 2:
        raise ValueError("direct compression requires 6<=L<=94 and L=2 mod 4")
    d = target.L // 2
    sides = []
    for e, o in ((target.a_even_ones, target.a_odd_ones),
                 (target.b_even_ones, target.b_odd_ones)):
        sides.append([c for c in factor2_compressed_contents(target.L, e, o)
                      if 0 <= e - c[0] <= c[1] and 0 <= o - c[0] <= c[1]])
    return tuple((a, b) for a in sides[0] for b in sides[1] if a[1] + b[1] == d)


def representative_lift(word: Sequence[int], even: int, odd: int) -> Tuple[int, ...]:
    """Construct a traceable source lift, not a claim of a valid PQCP.

    Input uses the existing raw compressed alphabet -2,0,2. The first
    e-n zero pairs get a one at their EVEN endpoint, wherever that endpoint
    lies (i or i+d). This avoids the adjacent-pair expansion mistake.
    """
    d = len(word)
    if d % 2 != 1 or any(type(x) is not int or x not in (-2, 0, 2) for x in word):
        raise ValueError("odd-length -2/0/2 compressed word required")
    negative, zeros = word.count(-2), word.count(0)
    remaining = even - negative
    if not 0 <= remaining <= zeros or odd != negative + zeros - remaining:
        raise ValueError("compressed word cannot realize the requested parity content")
    directions = []
    for i, value in enumerate(word):
        if value == 0:
            even_bit = int(remaining > 0)
            remaining -= even_bit
            directions.append(even_bit if i % 2 == 0 else 1 - even_bit)
    return lift_factor2_compressed(word, directions)


def ternary_profile(words: torch.Tensor) -> torch.Tensor:
    """Exact periodic pair sums for batch,2,d ternary words."""
    d = words.shape[-1]
    index = torch.arange(d, device=words.device)
    shifted = words[..., (index[:, None] + index[None, :]) % d]
    return (words[..., None, :] * shifted).sum(-1).sum(1)


def ternary_swap_delta(words: torch.Tensor, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Exact O(d) PACF delta for arbitrary-valued swaps, including shift zero.

    Set t=x[q]-x[p]. Linear terms are t*(x[p+u]+x[p-u]-x[q+u]-x[q-u]).
    Quadratic terms are -t^2 for each p/q edge and +2*t^2 at shift zero.
    This is NOT the binary sign-flip formula applied to a zero symbol.
    """
    d = words.shape[-1]
    u = torch.arange(d, device=words.device)[None, :]
    p, q = p[:, None], q[:, None]
    t = words.gather(1, q) - words.gather(1, p)
    linear = t * (words.gather(1, (p + u) % d) + words.gather(1, (p - u) % d)
                  - words.gather(1, (q + u) % d) - words.gather(1, (q - u) % d))
    edges = ((p + u) % d == q).to(words.dtype) + ((q + u) % d == p).to(words.dtype)
    return linear + t.square() * (2 * (u == 0).to(words.dtype) - edges)


@dataclass
class CompressedSearchResult:
    words: torch.Tensor
    profiles: torch.Tensor
    energies: torch.Tensor
    jobs: List[CompressedLiftJob]
    proposals: int
    accepted: int
    steps: int


@torch.no_grad()
def search_compressed(profiles: Sequence[TargetContentProfile], *, device: torch.device,
                      seed: int, steps: int = 128, candidates: int = 8,
                      block_steps: int = 32, stop_event: Optional[Event] = None,
                      trace: Optional[list] = None) -> CompressedSearchResult:
    """Batched multistart ternary swap search, with explicit bounded work.

    Initial words are random permutations of exact feasible contents, never
    read from saved solutions. Annealing temperature 4 -> .1 is an untuned
    heuristic in squared normalized-PACF units. Every proposal is checked for
    best energy, including rejected proposals; exact lanes freeze until export.
    Cancellation is checked between bounded blocks. Trace is test-only.
    """
    if not profiles or any(p.L != profiles[0].L for p in profiles):
        raise ValueError("nonempty equal-length target profiles required")
    if any(type(n) is not int or n <= 0 for n in (steps, candidates, block_steps)):
        raise ValueError("work limits must be positive integers")
    valid = frozenset(target_content_profiles(profiles[0].L, decimation_reduced=False))
    if any(p not in valid for p in profiles):
        raise ValueError("profile does not meet the Project 2 necessary conditions")
    rng = random.Random(seed)
    initial, contents = [], {}
    for target in profiles:
        if target not in contents:
            contents[target] = paired_contents(target)
        choices = contents[target]
        if not choices:
            raise ValueError("target has no half-shift-compatible ternary contents")
        pair = []
        for n, z, p in rng.choice(choices):
            word = [-1] * n + [0] * z + [1] * p
            rng.shuffle(word)
            pair.append(word)
        initial.append(pair)
    current = torch.tensor(initial, dtype=torch.float32, device=device)
    batch, _, d = current.shape
    target = torch.tensor([[v // 4 for v in fold_profile(target_profile(t.L, t.k, t.eta), 2)]
                           for t in profiles], dtype=torch.float32, device=device)
    profile = ternary_profile(current)
    energy = (profile[:, 1:d//2+1] - target[:, 1:d//2+1]).square().sum(-1)
    best, best_profile, best_energy = current.clone(), profile.clone(), energy.clone()
    lanes = torch.arange(batch, device=device)[:, None]
    accepted = torch.zeros((), device=device, dtype=torch.int64)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    completed = 0
    while completed < steps:
        if stop_event is not None and stop_event.is_set():
            break
        count = min(block_steps, steps - completed)
        draws = torch.rand(count, batch, candidates, 4, generator=generator).to(device)
        for draw in draws:
            which = (2 * draw[..., 0]).long()
            p, q = (d * draw[..., 1]).long(), (d * draw[..., 2]).long()
            selected = current[lanes, which]
            xp, xq = selected.gather(2, p[..., None]), selected.gather(2, q[..., None])
            legal = (xp[..., 0] != xq[..., 0]) & (best_energy[:, None] != 0)
            delta = ternary_swap_delta(selected.reshape(-1, d), p.flatten(), q.flatten())
            trial_profile = profile[:, None, :] + delta.reshape(batch, candidates, d) * legal[..., None]
            trial_energy = (trial_profile[..., 1:d//2+1] - target[:, None, 1:d//2+1]).square().sum(-1)
            flat = current[:, None].expand(-1, candidates, -1, -1).reshape(batch, candidates, 2*d)
            trials = flat.scatter(2, (which*d+p)[..., None], xq)
            trials = trials.scatter(2, (which*d+q)[..., None], xp).reshape(batch, candidates, 2, d)
            trials = torch.where(legal[..., None, None], trials, current[:, None])
            slot = trial_energy.masked_fill(~legal, float("inf")).argmin(-1)
            row = lanes[:, 0]
            candidate_energy = trial_energy[row, slot]
            better = candidate_energy < best_energy
            best = torch.where(better[:, None, None], trials[row, slot], best)
            best_profile = torch.where(better[:, None], trial_profile[row, slot], best_profile)
            best_energy = torch.minimum(best_energy, candidate_energy)
            temperature = 4.0 + (.1 - 4.0) * completed / steps
            change = candidate_energy - energy
            accept = legal[row, slot] & ((change <= 0) | (draw[row, slot, 3] < torch.exp(-change.clamp_min(0) / temperature)))
            current = torch.where(accept[:, None, None], trials[row, slot], current)
            profile = torch.where(accept[:, None], trial_profile[row, slot], profile)
            energy = torch.where(accept, candidate_energy, energy)
            accepted += accept.sum()
            completed += 1
            if trace is not None:
                trace.append((current.cpu().tolist(), profile.cpu().tolist()))
    jobs = []
    known = set()
    # Independent integer checks also cover nonzero bests, not just hits.
    for pair, observed, score, target_content in zip(
            best.int().cpu().tolist(), best_profile.int().cpu().tolist(),
            best_energy.int().cpu().tolist(), profiles):
        exact = [sum(v) for v in zip(*(integer_periodic_autocorrelation_profile(w) for w in pair))]
        desired = [v // 4 for v in fold_profile(target_profile(target_content.L, target_content.k, target_content.eta), 2)]
        if exact != observed or score != sum((x-y)**2 for x, y in zip(exact[1:d//2+1], desired[1:d//2+1])):
            raise RuntimeError("ternary GPU profile disagrees with integer recomputation")
        if score != 0:
            continue
        if exact != desired:
            raise RuntimeError("zero compressed energy failed full folded-target check")
        ca, cb = (tuple(2*v for v in w) for w in pair)
        a = representative_lift(ca, target_content.a_even_ones, target_content.a_odd_ones)
        b = representative_lift(cb, target_content.b_even_ones, target_content.b_odd_ones)
        if pair_content(a, b) != (target_content.a_even_ones, target_content.a_odd_ones,
                                 target_content.b_even_ones, target_content.b_odd_ones):
            raise RuntimeError("ternary export changed parity content")
        job = CompressedLiftJob(ca, cb, target_content, a, b)
        if job.canonical_key not in known:
            known.add(job.canonical_key)
            jobs.append(job)
    return CompressedSearchResult(best, best_profile, best_energy, jobs,
                                  completed * batch * candidates, int(accepted.item()), completed)
