"""MPS-native fixed-content population search for Project 2 PQCPs.

The previous PyTorch path optimized fractional signs and only rank-quantized
them during observation.  Low continuous loss therefore did not reliably
identify a useful binary basin.  This module keeps every evaluated candidate
binary and preserves the four proven even/odd one counts by construction.

For each canonical target/content profile, several independent islands sample
many candidates with Gumbel-top-k.  Exact binary PACF and the existing fixed-
target multiscale energy are evaluated in large MPS batches.  Elite bit
frequencies update each island's logits with smoothing and mutation noise.
This is a navigation heuristic only.  A zero is accepted exclusively after
full Python correlation recomputation and the independent verifier.
"""

from dataclasses import dataclass
import math
from typing import Optional, Tuple

import torch

from .compressed_fkm import compressed_fkm_lift_candidates
from .structured_energy import target_profile
from .target_profiles import TargetContentProfile, canonical_target_content_profiles


@dataclass(frozen=True)
class PopulationSearchConfig:
    """Bounded controls for independent cross-entropy search islands."""

    L: int
    seed: int = 123
    device: str = "mps"
    islands_per_profile: int = 4
    population_size: int = 256
    elite_count: int = 32
    learning_rate: float = 0.35
    probability_floor: float = 0.04
    mutation_noise: float = 0.12
    fkm_bias: float = 0.35
    compression: bool = True
    refinement: bool = True
    immigrant_fraction: float = 0.25

    def __post_init__(self) -> None:
        for name in ("L", "islands_per_profile", "population_size", "elite_count"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(name + " must be a positive integer")
        if self.L < 4 or self.L % 2:
            raise ValueError("population search requires even L >= 4")
        if self.elite_count > self.population_size:
            raise ValueError("elite_count cannot exceed population_size")
        if self.device not in ("mps", "cpu"):
            raise ValueError("device must be mps or cpu")
        for name in ("learning_rate", "probability_floor", "mutation_noise", "fkm_bias"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(name + " must be finite and nonnegative")
        if not 0 < self.learning_rate <= 1:
            raise ValueError("learning_rate must be in (0,1]")
        if not 0 < self.probability_floor < 0.5:
            raise ValueError("probability_floor must be in (0,0.5)")
        if not 0 < self.immigrant_fraction <= 1:
            raise ValueError("immigrant_fraction must be in (0,1]")


@dataclass(frozen=True)
class PopulationBatch:
    """One exact evaluated generation, retained on CPU only when requested."""

    bits: torch.Tensor
    profiles: torch.Tensor
    energy: torch.Tensor
    scores: torch.Tensor


def _expanded_profiles(config: PopulationSearchConfig) -> Tuple[TargetContentProfile, ...]:
    base = canonical_target_content_profiles(config.L)
    if not base:
        raise ValueError("no necessary target/content profiles for L={}".format(config.L))
    return tuple(profile for profile in base for _ in range(config.islands_per_profile))


class TorchPopulationSearch:
    """Exact-binary CEM islands whose bulk evaluation stays on one device."""

    def __init__(self, config: PopulationSearchConfig):
        self.config = config
        if config.device == "mps":
            if not torch.backends.mps.is_available():
                raise RuntimeError("MPS is unavailable")
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.profiles = _expanded_profiles(config)
        self.distributions = len(self.profiles)
        self.half = config.L // 2
        self.weights = torch.tensor([
            [p.a_even_ones, p.a_odd_ones, p.b_even_ones, p.b_odd_ones]
            for p in self.profiles
        ], dtype=torch.int64, device=self.device)
        self.targets = torch.tensor([
            target_profile(config.L, p.k, p.eta) for p in self.profiles
        ], dtype=torch.float32, device=self.device)
        index = torch.arange(config.L, device=self.device)
        self.shifts = (index[None, :] + index[:config.L // 2 + 1, None]) % config.L
        self.logits = torch.zeros(
            self.distributions, 4, self.half, dtype=torch.float32, device=self.device
        )
        self._apply_fkm_bias()
        self.generator = torch.Generator(device="cpu").manual_seed(config.seed)
        self.generation = 0
        self.candidate_evaluations = 0
        self.best = None
        self.elites = None

    def _apply_fkm_bias(self) -> None:
        """Weakly bias A toward genuine compressed FKM lifts; never fix bits."""
        if self.config.fkm_bias == 0:
            return
        pools = {}
        for island, profile in enumerate(self.profiles):
            key = (profile.a_even_ones, profile.a_odd_ones)
            if key not in pools:
                pools[key] = compressed_fkm_lift_candidates(
                    profile.L, *key, seed=self.config.seed,
                    limit=self.config.islands_per_profile,
                )
            pool = pools[key]
            if not pool:
                continue
            a = pool[island % len(pool)]
            self.logits[island, 0] = torch.tensor(
                [(2 * a[index] - 1) * self.config.fkm_bias for index in range(0, profile.L, 2)],
                device=self.device,
            )
            self.logits[island, 1] = torch.tensor(
                [(2 * a[index] - 1) * self.config.fkm_bias for index in range(1, profile.L, 2)],
                device=self.device,
            )

    @torch.no_grad()
    def sample(self, count: Optional[int] = None, *, uniform: bool = False) -> torch.Tensor:
        """Return bits shaped ``(island,population,2,L)`` with exact content."""
        c = self.config
        count = c.population_size if count is None else count
        draws = torch.rand(
            self.distributions, count, 4, self.half,
            generator=self.generator,
        ).clamp_(1e-7, 1 - 1e-7).to(self.device)
        gumbel = -torch.log(-torch.log(draws))
        order = torch.argsort((0 if uniform else self.logits[:, None]) + gumbel, dim=-1, descending=True)
        ranks = torch.empty_like(order).scatter_(
            -1, order,
            torch.arange(self.half, device=self.device).expand_as(order),
        )
        grouped = ranks < self.weights[:, None, :, None]
        bits = torch.empty(
            self.distributions, count, 2, c.L,
            dtype=torch.float32, device=self.device,
        )
        bits[:, :, 0, 0::2], bits[:, :, 0, 1::2] = grouped[:, :, 0], grouped[:, :, 1]
        bits[:, :, 1, 0::2], bits[:, :, 1, 1::2] = grouped[:, :, 2], grouped[:, :, 3]
        return bits

    @torch.no_grad()
    def evaluate(self, bits: torch.Tensor) -> PopulationBatch:
        """Evaluate exact binary PACF, fixed-target energy and PQCP objective."""
        c = self.config
        if (bits.ndim != 4 or bits.shape[0] != self.distributions
                or tuple(bits.shape[2:]) != (2, c.L)
                or bits.device.type != self.device.type):
            raise ValueError("population bits have the wrong shape or device")
        signs = 1 - 2 * bits
        flat = signs.reshape(-1, 2, c.L)
        half_profile = (flat.unsqueeze(-2) * flat[..., self.shifts]).sum((-1, -3))
        profile = torch.cat((half_profile, half_profile[:, 1:-1].flip(-1)), -1)
        profile = profile.reshape(self.distributions, bits.shape[1], c.L)
        return self._scored(bits, profile)

    def _scored(self, bits: torch.Tensor, profile: torch.Tensor) -> PopulationBatch:
        """Score an exact full or incrementally updated PACF on-device."""
        c = self.config
        residual = profile - self.targets[:, None]
        energy = residual[:, :, 1:c.L // 2 + 1].square().sum(-1) / 16
        if c.compression:
            for factor, weight in ((2, 2), (4, 4)):
                if c.L % factor == 0:
                    folded = residual.reshape(
                        self.distributions, bits.shape[1], factor, c.L // factor
                    ).sum(2)
                    energy += weight * folded.square().sum(-1) / 16
        side = profile[:, :, 1:]
        distance = torch.minimum(side.abs(), (side.abs() - 4).abs()).sum(-1)
        count = ((side != 0).sum(-1) - 2).abs()
        scores = distance + count + (profile[:, :, 0] - 2 * c.L).abs()
        return PopulationBatch(bits, profile, energy, scores)

    @staticmethod
    def _join(left: PopulationBatch, right: PopulationBatch) -> PopulationBatch:
        return PopulationBatch(*(torch.cat((getattr(left, name), getattr(right, name)), 1)
                                 for name in ("bits", "profiles", "energy", "scores")))

    @staticmethod
    def _take(batch: PopulationBatch, index: torch.Tensor) -> PopulationBatch:
        rows = torch.arange(index.shape[0], device=index.device)[:, None]
        return PopulationBatch(*(getattr(batch, name)[rows, index]
                                 for name in ("bits", "profiles", "energy", "scores")))

    def _offspring(self, count: int) -> PopulationBatch:
        """Mutate intact A/B parents, updating only O(L) affected terms.

        The usual offspring makes one opposite-bit, same-parity swap. Every
        eighth generation makes two, and every 32nd makes four to probe beyond
        one-swap minima. These are heuristic seed mutations, never pruning.
        Exact elite parents survive independently. Random immigrants supply
        new structures, so a poor first seed cannot lock every trajectory.
        """
        from .torch_polish import swap_correlation_delta

        valid = self.elites.scores != 0
        valid_order = (~valid).to(torch.int32).argsort(dim=1, stable=True)
        valid_count = valid.sum(-1)
        ranks = (torch.rand(self.distributions, count, generator=self.generator).to(self.device)
                 * valid_count[:, None]).to(torch.int64)
        parent_index = valid_order.gather(1, ranks)
        parent = self._take(self.elites, parent_index)
        bits, profile = parent.bits.clone(), parent.profiles.clone()
        radii = 4 if self.generation % 32 == 0 else (2 if self.generation % 8 == 0 else 1)
        flat = bits.reshape(-1, 2, self.config.L)
        rows = torch.arange(flat.shape[0], device=self.device)
        for _ in range(radii):
            draw = torch.rand(self.distributions, count, 4, self.half,
                              generator=self.generator).to(self.device)
            # Choose uniformly among mutable parity groups, then choose one
            # 1 and one 0 uniformly within that group. Full/empty groups are
            # excluded, including the L=4 boundary cases.
            mutable = (self.weights > 0) & (self.weights < self.half)
            group = draw[..., 0].masked_fill(~mutable[:, None], -1).argmax(-1).flatten()
            which, parity = group // 2, group % 2
            positions = 2 * torch.arange(self.half, device=self.device)[None, :] + parity[:, None]
            sequence = flat[rows, which]
            parity_bits = sequence.gather(1, positions)
            noise = torch.rand(flat.shape[0], self.half, generator=self.generator).to(self.device)
            p = 2 * noise.masked_fill(parity_bits != 1, -1).argmax(-1) + parity
            q = 2 * noise.masked_fill(parity_bits != 0, -1).argmax(-1) + parity
            legal = ((sequence[rows, p] != sequence[rows, q])
                     & mutable.any(-1).repeat_interleave(count)
                     & (valid_count > 0).repeat_interleave(count))
            delta = swap_correlation_delta(1 - 2 * sequence, p, q) * legal[:, None]
            profile = profile + delta.reshape_as(profile)
            xp, xq = sequence[rows, p].clone(), sequence[rows, q].clone()
            flat[rows, which, p] = torch.where(legal, xq, xp)
            flat[rows, which, q] = torch.where(legal, xp, xq)
        return self._scored(bits, profile)

    def _select_elites(self, batch: PopulationBatch) -> PopulationBatch:
        """Retain distinct intact pairs per target island by exact energy.

        Dot products of binary signs detect exact duplicates without hashing
        collisions or transferring a population to CPU. This is local archive
        diversity, not mathematical orbit pruning. If fewer than K distinct
        pairs exist, duplicate slots are harmless and keep a fixed shape.
        """
        order = batch.energy.argsort(dim=1, stable=True)
        ordered = self._take(batch, order)
        signs = (1 - 2 * ordered.bits).flatten(2)
        duplicates = torch.bmm(signs, signs.transpose(1, 2)) == 2 * self.config.L
        duplicate = duplicates.tril(diagonal=-1).any(-1)
        priority = ordered.energy.masked_fill(duplicate | (ordered.scores == 0), float("inf"))
        selected = priority.argsort(dim=1, stable=True)[:, :self.config.elite_count]
        return self._take(ordered, selected)

    @torch.no_grad()
    def step(self) -> PopulationBatch:
        """Sample, exactly evaluate and update every island from its elites."""
        c = self.config
        if c.refinement and self.elites is not None:
            immigrants = max(1, math.ceil(c.population_size * c.immigrant_fraction))
            # Half the immigrants ignore learned logits completely, giving
            # every fixed-content pair nonzero sampling probability even
            # after an island's learned distribution concentrates.
            fresh = max(1, immigrants // 2)
            fresh_bits = self.sample(fresh, uniform=True)
            if fresh < immigrants:
                fresh_bits = torch.cat((fresh_bits, self.sample(immigrants - fresh)), 1)
            batch = self.evaluate(fresh_bits)
            if immigrants < c.population_size:
                batch = self._join(batch, self._offspring(c.population_size - immigrants))
            self.elites = self._select_elites(self._join(self.elites, batch))
        else:
            batch = self.evaluate(self.sample())
            if c.refinement:
                self.elites = self._select_elites(batch)
            else:
                elite_index = torch.topk(batch.energy, c.elite_count, dim=1,
                                         largest=False, sorted=False).indices
                self.elites = self._take(batch, elite_index)
        elite = self.elites.bits
        grouped = torch.stack((
            elite[:, :, 0, 0::2], elite[:, :, 0, 1::2],
            elite[:, :, 1, 0::2], elite[:, :, 1, 1::2],
        ), dim=2)
        # Found solutions are observations only, never teachers or parents.
        # Some tiny profile families contain only solutions; leave their
        # sampling distribution unbiased instead of learning those pairs.
        eligible = (self.elites.scores != 0).to(grouped.dtype)
        frequency = (grouped * eligible[:, :, None, None]).sum(1) / eligible.sum(1).clamp_min(1)[:, None, None]
        floor = c.probability_floor
        probability = frequency.clamp(floor, 1 - floor)
        desired = torch.log(probability / (1 - probability))
        desired = torch.where(eligible.any(1)[:, None, None], desired, 0)
        self.logits.lerp_(desired, c.learning_rate)
        if c.mutation_noise:
            noise = torch.randn(self.logits.shape, generator=self.generator).to(self.device)
            self.logits.add_(noise, alpha=c.mutation_noise)
        self.logits.clamp_(-6, 6)
        self.generation += 1
        self.candidate_evaluations += self.distributions * c.population_size
        flat_scores = batch.scores.flatten()
        index = int(flat_scores.argmin().item())
        distribution, member = divmod(index, c.population_size)
        score = int(flat_scores[index].item())
        if self.best is None or score < self.best["score"]:
            self.best = {
                "score": score, "generation": self.generation,
                "distribution": distribution, "member": member,
                "profile_case": self.profiles[distribution],
                "A": batch.bits[distribution, member, 0].to(torch.int32).cpu().tolist(),
                "B": batch.bits[distribution, member, 1].to(torch.int32).cpu().tolist(),
                "profile": batch.profiles[distribution, member].to(torch.int32).cpu().tolist(),
                "energy": int(batch.energy[distribution, member].item()),
            }
        return batch


__all__ = ("PopulationBatch", "PopulationSearchConfig", "TorchPopulationSearch")
