"""Douglas--Rachford between exact-count colors and C/G Fourier spheres.

Every true full solution lies in both sets. Color assignments are certified
nearest projections of the represented float coordinates; Fourier projections
use the exact radial formula evaluated numerically. Nonconvex convergence is
not guaranteed. Only independent integer/full verification establishes a hit.
"""
from dataclasses import dataclass
import random

import torch

from .colour_projection import project_colour_row
from .correlation import full_correlation_profile
from .mps_memory import autorelease_pool
from .objective import pqcp_objective
from .structured_energy import target_profile
from .target_profiles import target_content_profiles
from .torch_compressed_search import search_paired_contents, ternary_profile
from .torch_joint_search import joint_binary_words
from .verifier import verify_pqcp


def project_palette_tensor(values, counts):
    batch, components, rows, d = values.shape
    if (components, rows) != (2, 2) or len(counts) != batch:
        raise ValueError('batch, C/G, two rows, odd length coordinates required')
    raw = values.detach().cpu().tolist()
    result, certified = [], 0
    for value, lane_counts in zip(raw, counts):
        cs, gs = [], []
        for side in range(2):
            labels, checked = project_colour_row(value[0][side], value[1][side], lane_counts[side])
            certified += int(checked)
            cs.append([1 if v == 0 else -1 if v == 1 else 0 for v in labels])
            gs.append([1 if v == 2 else -1 if v == 3 else 0 for v in labels])
        result.append([cs, gs])
    return torch.tensor(result, dtype=values.dtype, device=values.device), certified


def spectral_joint_projection(values, cosine, sine, targets):
    d = values.shape[-1]
    real, imag = values @ cosine.T, -(values @ sine.T)
    power = (real.square()+imag.square()).sum(-2)
    scale = (targets/power.clamp_min(1e-20)).sqrt()
    real, imag = real*scale[..., None, :], imag*scale[..., None, :]
    zero = power == 0
    real[..., 0, :] = torch.where(zero, targets.sqrt(), real[..., 0, :])
    multiplicity = torch.ones(cosine.shape[0], dtype=values.dtype, device=values.device)*2
    multiplicity[0] = 1
    return ((real*multiplicity) @ cosine-(imag*multiplicity) @ sine)/d


@dataclass
class JointFeasibilityResult:
    candidates: list
    energies: list
    iterations: int
    native_certificates: int
    exact_integer_fallbacks: int


@torch.no_grad()
def search_joint_feasibility(profiles, *, device, seed, iterations=2000, observation_interval=25,
                             export_limit=128, stop_event=None):
    if (not profiles or any(p.L != profiles[0].L for p in profiles) or profiles[0].L % 4 != 2
            or not 6 <= profiles[0].L <= 94 or type(seed) is not int
            or type(iterations) is not int or iterations < 1 or type(observation_interval) is not int
            or observation_interval < 1 or type(export_limit) is not int or export_limit < 0):
        raise ValueError('valid target batch and bounded feasibility iterations required')
    if any(p not in target_content_profiles(profiles[0].L, decimation_reduced=False) for p in profiles):
        raise ValueError('Project 2 targets required')
    batch, d, rng = len(profiles), profiles[0].L//2, random.Random(seed)
    initial, counts, wanted_c, wanted_g = [], [], [], []
    for target in profiles:
        allowed = search_paired_contents(target)
        if not allowed: raise ValueError('no feasible contents')
        row_counts, cs, gs = [], [], []
        for (nc, zero, pc), even in zip(rng.choice(allowed), (target.a_even_ones, target.b_even_ones)):
            ng = even-nc
            group = (pc, nc, zero-ng, ng)
            row_counts.append(group)
            colors = [color for color, count in enumerate(group) for _ in range(count)]
            rng.shuffle(colors)
            cs.append([1 if v == 0 else -1 if v == 1 else 0 for v in colors])
            gs.append([1 if v == 2 else -1 if v == 3 else 0 for v in colors])
        counts.append(row_counts); initial.append([cs, gs])
        full = target_profile(target.L, target.k, target.eta)
        wanted_c.append(tuple((full[u]+full[u+d])//4 for u in range(d)))
        wanted_g.append(tuple((-1 if u % 2 else 1)*(full[u]-full[u+d])//4 for u in range(d)))
    theta = torch.tensor(initial, dtype=torch.float32, device=device)
    generator = torch.Generator().manual_seed(seed)
    theta += .1*torch.randn(theta.shape, generator=generator).to(device)
    frequencies = torch.arange(d//2+1, dtype=theta.dtype, device=device)
    angles = 2*torch.pi*frequencies[:, None]*torch.arange(d, dtype=theta.dtype, device=device)[None]/d
    cosine, sine = angles.cos(), angles.sin()
    spectral_targets = torch.stack([torch.stack((d+2*p.eta*cosine[:, p.k % d],
        d+2*p.eta*(-1 if p.k % 2 else 1)*cosine[:, p.k % d])) for p in profiles])
    tc, tg = (torch.tensor(rows, dtype=theta.dtype, device=device) for rows in (wanted_c, wanted_g))
    best = torch.tensor(initial, dtype=theta.dtype, device=device)
    initial_c, initial_g = ternary_profile(best[:, 0]), ternary_profile(best[:, 1])
    best_energy = ((initial_c[:, 1:d//2+1]-tc[:, 1:d//2+1]).square().sum(-1)
                  +(initial_g[:, 1:d//2+1]-tg[:, 1:d//2+1]).square().sum(-1))
    certified = calls = completed = 0
    for iteration in range(iterations):
        if stop_event is not None and stop_event.is_set(): break
        with autorelease_pool(device):
            projected, certificates = project_palette_tensor(theta, counts)
            certified += certificates; calls += 2*batch
            # Every discrete iterate is checked: interval sampling could miss
            # a transient full solution. Retain the argument for API callers.
            c, g = ternary_profile(projected[:, 0]), ternary_profile(projected[:, 1])
            energy = (c[:, 1:d//2+1]-tc[:, 1:d//2+1]).square().sum(-1)+(g[:, 1:d//2+1]-tg[:, 1:d//2+1]).square().sum(-1)
            better = energy < best_energy
            best = torch.where(better[:, None, None, None], projected, best)
            best_energy = torch.minimum(best_energy, energy)
            reflected = 2*projected-theta
            theta += spectral_joint_projection(reflected, cosine, sine, spectral_targets)-projected
            completed += 1
    # Audit the best discrete point of every lane in original binary length.
    words, energies, candidates = best.int().cpu().tolist(), best_energy.cpu().tolist(), []
    selected = set(sorted(range(batch), key=lambda i: (energies[i], i))[:export_limit])
    selected.update(i for i, value in enumerate(energies) if value == 0)
    for lane, (value, target) in enumerate(zip(words, profiles)):
        binary = joint_binary_words(value[0], value[1], target)
        profile = tuple(full_correlation_profile(*binary)); desired = target_profile(target.L, target.k, target.eta)
        exact = sum((a-b)**2 for a, b in zip(profile[1:d+1], desired[1:d+1]))//16
        if energies[lane] != 2*exact:
            raise RuntimeError('feasibility candidate failed integer energy audit')
        energies[lane] = 2*exact
        if lane not in selected: continue
        valid = verify_pqcp(*binary).is_valid if exact == 0 else False
        if exact == 0 and not valid: raise RuntimeError('feasibility zero failed full verifier')
        candidates.append(dict(lane=lane, A=binary[0], B=binary[1], profile=profile, score=pqcp_objective(profile),
            target_energy=exact, k=target.k, sign=target.eta, verified=valid, method='joint_spectral_feasibility'))
    return JointFeasibilityResult(candidates, energies, completed, certified, calls-certified)
