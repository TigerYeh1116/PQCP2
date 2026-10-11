"""Joint C/G search in the exact half-shift-zero binary state space.

Four colors per antipodal cell: c+, c-, g+, g-. Complementary supports,
four parity contents, and total half-shift zero hold after every proposal.
The integer energy Ec+Eg is exactly twice the full fixed-target energy.
Swap/transfer scheduling is heuristic; a bounded miss proves no nonexistence.
"""
from dataclasses import dataclass
import math
import random

import torch

from .correlation import full_correlation_profile, normalize_binary_sequence
from .objective import pqcp_objective
from .mps_memory import autorelease_pool
from .structured_energy import fold_profile, integer_periodic_autocorrelation_profile as pacf, target_profile
from .target_profiles import pair_content, target_content_profiles
from .torch_compressed_search import search_paired_contents, ternary_profile, ternary_swap_delta
from .verifier import verify_pqcp


def joint_binary_words(coarse, difference, target):
    d = target.L//2
    coarse, difference = tuple(tuple(row) for row in coarse), tuple(tuple(row) for row in difference)
    if (target.L % 4 != 2 or len(coarse) != 2 or len(difference) != 2
            or any(len(row) != d or any(type(v) is not int or v not in (-1, 0, 1) for v in row)
                   for row in coarse+difference)
            or any((c != 0)+(g != 0) != 1 for cs, gs in zip(coarse, difference) for c, g in zip(cs, gs))
            or sum(v != 0 for row in coarse for v in row) != d):
        raise ValueError('complementary ternary rows of total coarse weight d required')
    words = []
    for c, g in zip(coarse, difference):
        h = tuple((-1 if i % 2 else 1)*v for i, v in enumerate(g))
        signs = tuple(a+b for a, b in zip(c, h))+tuple(a-b for a, b in zip(c, h))
        words.append(tuple(int(v == -1) for v in signs))
    words = tuple(words)
    if pair_content(*words) != (target.a_even_ones, target.a_odd_ones, target.b_even_ones, target.b_odd_ones):
        raise RuntimeError('joint construction changed parity contents')
    return words


def binary_joint_rows(words, target):
    """Inverse antipodal encoding, with independent content/half-shift audits."""
    if len(words) != 2:
        raise ValueError('two binary rows required')
    words = tuple(tuple(normalize_binary_sequence(row)) for row in words)
    if any(len(row) != target.L for row in words) or target.L % 4 != 2:
        raise ValueError('binary words must match an odd-half target')
    d = target.L//2
    signs = tuple(tuple(1-2*v for v in row) for row in words)
    c = tuple(tuple((row[i]+row[i+d])//2 for i in range(d)) for row in signs)
    g = tuple(tuple((-1 if i % 2 else 1)*(row[i]-row[i+d])//2 for i in range(d)) for row in signs)
    if joint_binary_words(c, g, target) != words:
        raise RuntimeError('inverse joint encoding failed round trip')
    return c, g


def _replace_delta(words, p, q, vp, vq):
    """Exact two-site trit update, including quadratic cross terms and origin."""
    d = words.shape[-1]
    u = torch.arange(d, device=words.device)[None, :]
    p, q, vp, vq = (v[:, None] for v in (p, q, vp, vq))
    alpha, beta = vp-words.gather(1, p), vq-words.gather(1, q)
    value = alpha*(words.gather(1, (p+u) % d)+words.gather(1, (p-u) % d))
    value += beta*(words.gather(1, (q+u) % d)+words.gather(1, (q-u) % d))
    edges = ((p+u) % d == q).to(words.dtype)+((q+u) % d == p).to(words.dtype)
    return value+alpha*beta*edges+(alpha.square()+beta.square())*(u == 0)


def joint_swap_trials(c, g, draw):
    batch, _, d = c.shape
    count = draw.shape[1]
    lanes = torch.arange(batch, device=c.device)[:, None]
    which = (2*draw[..., 0]).long()
    p, q = (d*draw[..., 1]).long(), (d*draw[..., 2]).long()
    trials, deltas, values = [], [], []
    for words in (c, g):
        selected = words[lanes, which]
        xp, xq = selected.gather(-1, p[..., None]), selected.gather(-1, q[..., None])
        delta = ternary_swap_delta(selected.reshape(-1, d), p.flatten(), q.flatten()).reshape(batch, count, d)
        flat = words[:, None].expand(-1, count, -1, -1).reshape(batch, count, 2*d)
        trial = flat.scatter(-1, (which*d+p)[..., None], xq).scatter(-1, (which*d+q)[..., None], xp)
        trials.append(trial.reshape(batch, count, 2, d)); deltas.append(delta); values.append((xp, xq))
    legal = ((values[0][0] != values[0][1]) | (values[1][0] != values[1][1]))[..., 0]
    return trials[0], trials[1], deltas[0], deltas[1], legal


def joint_transfer_trials(c, g, draw, lower, upper):
    """One row exchanges g+/g- for c+/c-, the other performs the inverse."""
    batch, _, d = c.shape
    count = draw.shape[1]
    p = (d*torch.stack((draw[..., 1], draw[..., 4]), -1)).long()
    q = (d*torch.stack((draw[..., 2], draw[..., 5]), -1)).long()
    create_a = draw[..., 0] < .5
    create = torch.stack((create_a, ~create_a), -1)
    sign_c = torch.where(draw[..., 6] < .5, 1., -1.)[..., None]
    sign_g = torch.where(draw[..., 7] < .5, 1., -1.)[..., None]
    candidates, deltas, old = [], [], []
    for words, selected, sign in ((c, create, sign_c), (g, ~create, sign_g)):
        expanded = words[:, None].expand(-1, count, -1, -1)
        xp, xq = expanded.gather(-1, p[..., None])[..., 0], expanded.gather(-1, q[..., None])[..., 0]
        vp = torch.where(selected, sign, 0.)
        vq = -vp
        candidates.append(expanded.scatter(-1, p[..., None], vp[..., None]).scatter(-1, q[..., None], vq[..., None]))
        delta = _replace_delta(expanded.reshape(-1, d), p.flatten(), q.flatten(), vp.flatten(), vq.flatten())
        deltas.append(delta.reshape(batch, count, 2, d).sum(2)); old.append((xp, xq))
    cx, cq = old[0]; gx, gq = old[1]
    legal = ((p != q) & torch.where(create, (cx == 0) & (cq == 0) & (gx*gq == -1),
                                   (cx*cq == -1) & (gx == 0) & (gq == 0))).all(-1)
    counts = (candidates[0] == -1).sum(-1)
    legal &= ((counts >= lower[:, None]) & (counts <= upper[:, None])).all(-1)
    return candidates[0], candidates[1], deltas[0], deltas[1], legal


@dataclass
class JointSearchResult:
    coarse: torch.Tensor
    difference: torch.Tensor
    coarse_profiles: torch.Tensor
    difference_profiles: torch.Tensor
    energies: torch.Tensor
    candidates: list
    proposals: int
    accepted: int
    transfer_accepted: int
    steps: int


def _draw_blocks(steps, block_steps, batch, candidates, generator, device, stop_event):
    """The pool stays open while the caller consumes each bounded block."""
    completed = 0
    while completed < steps:
        if stop_event is not None and stop_event.is_set():
            return
        count = min(block_steps, steps-completed)
        with autorelease_pool(device):
            yield torch.rand(count, batch, candidates, 8, generator=generator).to(device)
        completed += count


@torch.no_grad()
def search_joint(profiles, *, device, seed, steps=128, candidates=8, block_steps=32,
                 transfer_interval=4, export_limit=128, stop_event=None, trace=None,
                 content_pruning=True, binary_content_pruning=True, initial_words=None,
                 initial_temperature=4., final_temperature=.1, direct_transfers=False):
    if (not profiles or any(p.L != profiles[0].L for p in profiles) or profiles[0].L % 4 != 2
            or not 6 <= profiles[0].L <= 94 or type(seed) is not int
            or any(type(n) is not int or n <= 0 for n in (steps, candidates, block_steps))
            or type(transfer_interval) is not int or transfer_interval < 0
            or type(export_limit) is not int or export_limit < 0
            or type(content_pruning) is not bool or type(binary_content_pruning) is not bool
            or type(direct_transfers) is not bool):
        raise ValueError('valid odd-half-length target batch and bounded work required')
    if (any(not isinstance(t, (int, float)) or isinstance(t, bool) or not math.isfinite(t) or t <= 0
            for t in (initial_temperature, final_temperature))
            or initial_words is not None and len(initial_words) != len(profiles)):
        raise ValueError('finite positive temperatures and one initial pair per lane required')
    valid = frozenset(target_content_profiles(profiles[0].L, decimation_reduced=False))
    if any(p not in valid for p in profiles):
        raise ValueError('Project 2 target profiles required')
    batch, d = len(profiles), profiles[0].L//2
    rng = random.Random(seed)
    initial_c, initial_g, bounds = [], [], []
    for lane, target in enumerate(profiles):
        allowed = search_paired_contents(target, content_pruning=content_pruning, binary_content_pruning=binary_content_pruning)
        if not allowed:
            raise ValueError('no half-shift feasible contents for target')
        cs, gs = [], []
        for (negative, zero, positive), even in zip(rng.choice(allowed), (target.a_even_ones, target.b_even_ones)):
            ng = even-negative
            colors = [0]*positive+[1]*negative+[2]*(zero-ng)+[3]*ng
            rng.shuffle(colors)
            cs.append(tuple(1 if v == 0 else -1 if v == 1 else 0 for v in colors))
            gs.append(tuple(1 if v == 2 else -1 if v == 3 else 0 for v in colors))
        if initial_words is not None:
            cs, gs = binary_joint_rows(initial_words[lane], target)
            contents = tuple((row.count(-1), row.count(0), row.count(1)) for row in cs)
            if contents not in allowed:
                raise ValueError('initial binary pair violates verified content bounds')
        initial_c.append(cs); initial_g.append(gs)
        bounds.append(tuple((min(pair[side][0] for pair in allowed), max(pair[side][0] for pair in allowed)) for side in (0, 1)))
    c, g = (torch.tensor(rows, dtype=torch.float32, device=device) for rows in (initial_c, initial_g))
    lower = torch.tensor([[v[0] for v in row] for row in bounds], device=device)
    upper = torch.tensor([[v[1] for v in row] for row in bounds], device=device)
    tc, tg = [], []
    for target in profiles:
        full = target_profile(target.L, target.k, target.eta)
        tc.append(tuple(v//4 for v in fold_profile(full, 2)))
        tg.append(tuple((-1 if u % 2 else 1)*(full[u]-full[u+d])//4 for u in range(d)))
    tc, tg = (torch.tensor(rows, dtype=c.dtype, device=device) for rows in (tc, tg))
    pc, pg = ternary_profile(c), ternary_profile(g)
    def energy(ac, ag):
        return ((ac[..., 1:d//2+1]-tc[:, None, 1:d//2+1]).square().sum(-1)
               +(ag[..., 1:d//2+1]-tg[:, None, 1:d//2+1]).square().sum(-1))
    current_energy = energy(pc[:, None], pg[:, None])[:, 0]
    best_c, best_g, best_pc, best_pg, best_energy = (v.clone() for v in (c, g, pc, pg, current_energy))
    row = torch.arange(batch, device=device)
    accepted = torch.zeros((), dtype=torch.int64, device=device)
    transfer_accepted = accepted.clone()
    generator = torch.Generator(device='cpu').manual_seed(seed)
    completed = 0
    for draws in _draw_blocks(steps, block_steps, batch, candidates, generator, device, stop_event):
        for draw in draws:
            transfer = bool(transfer_interval and (completed+1) % transfer_interval == 0)
            if transfer and direct_transfers:
                from .torch_joint_transfer import direct_transfer_draw
                draw = direct_transfer_draw(c, g, draw, lower, upper)
            trials_c, trials_g, dc, dg, legal = (joint_transfer_trials(c, g, draw, lower, upper) if transfer
                                                else joint_swap_trials(c, g, draw))
            legal &= best_energy[:, None] != 0
            trial_pc, trial_pg = pc[:, None]+dc, pg[:, None]+dg
            scores = energy(trial_pc, trial_pg)
            slot = scores.masked_fill(~legal, float('inf')).argmin(-1)
            score = scores[row, slot]
            chosen = legal[row, slot]
            better = chosen & (score < best_energy)
            best_c = torch.where(better[:, None, None], trials_c[row, slot], best_c)
            best_g = torch.where(better[:, None, None], trials_g[row, slot], best_g)
            best_pc = torch.where(better[:, None], trial_pc[row, slot], best_pc)
            best_pg = torch.where(better[:, None], trial_pg[row, slot], best_pg)
            best_energy = torch.where(better, score, best_energy)
            change = score-current_energy
            temperature = initial_temperature+(final_temperature-initial_temperature)*completed/steps
            accept = chosen & ((change <= 0) | (draw[row, slot, 3] < torch.exp(-change.clamp_min(0)/temperature)))
            c = torch.where(accept[:, None, None], trials_c[row, slot], c)
            g = torch.where(accept[:, None, None], trials_g[row, slot], g)
            pc = torch.where(accept[:, None], trial_pc[row, slot], pc)
            pg = torch.where(accept[:, None], trial_pg[row, slot], pg)
            current_energy = torch.where(accept, score, current_energy)
            accepted += accept.sum()
            if transfer: transfer_accepted += accept.sum()
            completed += 1
            if trace is not None:
                trace.append((c.cpu().tolist(), g.cpu().tolist(), pc.cpu().tolist(), pg.cpu().tolist()))
    out_c, out_g = best_c.int().cpu().tolist(), best_g.int().cpu().tolist()
    out_pc, out_pg = best_pc.int().cpu().tolist(), best_pg.int().cpu().tolist()
    out_energy = best_energy.int().cpu().tolist()
    selected = set(sorted(range(batch), key=lambda i: (out_energy[i], i))[:export_limit])
    selected.update(i for i, value in enumerate(out_energy) if value == 0)
    exports = []
    for lane, (cs, gs, observed_c, observed_g, value, target) in enumerate(zip(out_c, out_g, out_pc, out_pg, out_energy, profiles)):
        exact_c = tuple(a+b for a, b in zip(*(pacf(w) for w in cs)))
        exact_g = tuple(a+b for a, b in zip(*(pacf(w) for w in gs)))
        wanted_c, wanted_g = tc[lane].int().cpu().tolist(), tg[lane].int().cpu().tolist()
        ec = sum((a-b)**2 for a, b in zip(exact_c[1:d//2+1], wanted_c[1:d//2+1]))
        eg = sum((a-b)**2 for a, b in zip(exact_g[1:d//2+1], wanted_g[1:d//2+1]))
        if tuple(observed_c) != exact_c or tuple(observed_g) != exact_g or value != ec+eg or value % 2:
            raise RuntimeError('joint GPU profiles or energy failed integer audit')
        words = joint_binary_words(cs, gs, target)
        if lane not in selected:
            continue
        full = tuple(full_correlation_profile(*words))
        desired = target_profile(target.L, target.k, target.eta)
        if full[d] != 0 or sum((a-b)**2 for a, b in zip(full[1:d+1], desired[1:d+1])) != 8*value:
            raise RuntimeError('joint/full energy identity failed independent binary audit')
        verified = verify_pqcp(*words).is_valid if value == 0 else False
        if value == 0 and not verified:
            raise RuntimeError('joint zero failed independent full verifier')
        exports.append(dict(lane=lane, A=words[0], B=words[1], profile=full, score=pqcp_objective(full),
            target_energy=value//2, coarse_energy=ec, difference_energy=eg,
            k=target.k, sign=target.eta, verified=verified, method='joint_half_shift_colors'))
    return JointSearchResult(best_c, best_g, best_pc, best_pg, best_energy, exports,
        batch*candidates*completed, int(accepted.item()), int(transfer_accepted.item()), completed)
