"""MPS expansion of independently verified PQCP neighborhoods.

Known solutions supply higher-order positional structure that a low scalar
objective does not encode.  This module never returns the known pair itself:
it applies one or more same-parity opposite-bit swaps (therefore preserving
all four necessary content counts), evaluates every resulting pair on MPS,
and ranks it by the exact fixed-target multiscale energy.  The output is only
a seed bank; full correlation and the independent verifier remain mandatory.
"""

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Sequence, Tuple

import torch

from .correlation import full_correlation_profile
from .objective import pqcp_objective
from .target_profiles import TargetContentProfile, pair_content
from .torch_polish import discrete_energy
from .torch_search import BatchedPQCP, require_device
from .verifier import verify_pqcp


Pair = Tuple[Tuple[int, ...], Tuple[int, ...]]


@dataclass(frozen=True)
class SeedExpansionConfig:
    """Controls for deterministic, fixed-content MPS neighborhood expansion."""

    L: int
    seed: int = 123
    device: str = "mps"
    clones_per_solution: int = 8
    max_swap_radius: int = 8
    limit: int = 2000

    def __post_init__(self) -> None:
        for name in ("L", "clones_per_solution", "max_swap_radius", "limit"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(name + " must be a positive integer")
        if self.L < 4 or self.L % 2:
            raise ValueError("seed expansion requires even L >= 4")
        if self.device not in ("mps", "cpu"):
            raise ValueError("device must be mps or cpu")


def load_verified_solution_seeds(path: Path, length: int) -> Tuple[Pair, ...]:
    """Read established ``a=/b=`` records and independently verify all of them."""
    source = Path(path)
    if not source.is_file():
        return ()
    pairs = []
    seen = set()
    for raw_a, raw_b in re.findall(
        r"^a=([01]+)\nb=([01]+)$", source.read_text(encoding="utf-8"), re.MULTILINE
    ):
        a, b = tuple(map(int, raw_a)), tuple(map(int, raw_b))
        if len(a) != length or len(b) != length:
            raise ValueError("recorded solution length mismatch")
        verification = verify_pqcp(a, b)
        if not verification.is_valid:
            raise ValueError("recorded seed fails independent verifier")
        key = tuple(sorted((a, b)))
        if key not in seen:
            seen.add(key)
            pairs.append((a, b))
    return tuple(pairs)


@torch.no_grad()
def expand_solution_neighborhoods(
    pairs: Sequence[Pair], config: SeedExpansionConfig,
) -> Tuple[dict, ...]:
    """Return diverse nonzero-radius MPS-ranked seeds around verified pairs."""
    if not pairs:
        return ()
    device = require_device(config.device)
    expanded_pairs = []
    profiles = []
    radii = []
    sources = []
    for source_index, (a, b) in enumerate(pairs):
        verification = verify_pqcp(a, b)
        if len(a) != config.L or len(b) != config.L or not verification.is_valid:
            raise ValueError("seed pair must be a verified PQCP of configured length")
        k = min(verification.nonzero_shifts)
        eta = verification.profile[k] // 4
        content = pair_content(a, b)
        for clone in range(config.clones_per_solution):
            # Even rotations preserve parity content and the same target k.
            rotation = 2 * ((clone * 104729 + config.seed) % (config.L // 2))
            rotated_a = a[rotation:] + a[:rotation]
            rotated_b = b[rotation:] + b[:rotation]
            expanded_pairs.append((rotated_a, rotated_b))
            profiles.append(TargetContentProfile(config.L, k, eta, *content))
            radii.append(1 + clone % config.max_swap_radius)
            sources.append(source_index)
    bits = torch.tensor(expanded_pairs, dtype=torch.float32, device=device)
    groups = bits.reshape(-1, 2, config.L // 2, 2).transpose(-1, -2).reshape(-1, 4, config.L // 2)
    generator = torch.Generator(device="cpu").manual_seed(config.seed + 7_000_003)
    lane = torch.arange(len(expanded_pairs), device=device)
    radii_tensor = torch.tensor(radii, device=device)
    half = config.L // 2
    for move in range(config.max_swap_radius):
        active = radii_tensor > move
        draw = torch.rand(len(expanded_pairs), 3, generator=generator).to(device)
        group = (draw[:, 0] * 4).to(torch.int64)
        selected = groups[lane, group]
        one_score = draw[:, 1, None].expand(-1, half)  # offset below makes positions distinct
        zero_score = draw[:, 2, None].expand(-1, half)
        offsets = torch.rand(len(expanded_pairs), half, generator=generator).to(device)
        one = (offsets + one_score).masked_fill(selected == 0, -1).argmax(-1)
        offsets = torch.rand(len(expanded_pairs), half, generator=generator).to(device)
        zero = (offsets + zero_score).masked_fill(selected == 1, -1).argmax(-1)
        flat = groups.reshape(len(expanded_pairs), -1)
        one_index, zero_index = group * half + one, group * half + zero
        updated = flat.scatter(1, one_index[:, None], torch.zeros_like(one_index[:, None], dtype=flat.dtype))
        updated = updated.scatter(1, zero_index[:, None], torch.ones_like(zero_index[:, None], dtype=flat.dtype))
        groups = torch.where(active[:, None], updated, flat).reshape_as(groups)
    bits = groups.reshape(-1, 2, 2, half).transpose(-1, -2).reshape(-1, 2, config.L)
    model = BatchedPQCP(profiles, device)
    signs = 1 - 2 * bits
    observed = model.correlation(signs)
    energies = discrete_energy(observed, model.targets, compression=True)
    scores = model.discrete_scores(observed)
    cpu_bits = bits.to(torch.int32).cpu().tolist()
    cpu_profiles = observed.to(torch.int32).cpu().tolist()
    order = sorted(range(len(expanded_pairs)), key=lambda i: (
        int(energies[i].item()), int(scores[i].item()), sources[i], radii[i], i
    ))[:config.limit]
    records = []
    for index in order:
        a, b = cpu_bits[index]
        exact = full_correlation_profile(a, b)
        if exact != cpu_profiles[index]:
            raise RuntimeError("MPS expansion profile failed full recomputation")
        target = profiles[index]
        records.append({
            "A": a, "B": b, "profile": exact, "score": pqcp_objective(exact),
            "iteration": index, "k": target.k, "sign": target.eta,
            "source_solution": sources[index], "swap_radius": radii[index],
            "target_energy": int(energies[index].item()),
        })
    return tuple(records)


__all__ = (
    "Pair", "SeedExpansionConfig", "expand_solution_neighborhoods",
    "load_verified_solution_seeds",
)
