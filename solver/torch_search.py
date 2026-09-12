"""Batched MPS continuous search for the Project 2 two-sidelobe PQCP target.

The PACP slides motivate batched Adam, quantization, elite storage and rebirth.
Their targets (all sidelobes +/-2, or only the half shift +/-4) do NOT apply.
Here T(0)=2L, T(k)=T(L-k)=4*eta, and all other entries are zero, 0<k<L/2.
DC/Nyquist content profiles, genuine compressed FKM, the integer objective,
independent verifier and deduplicating L.txt writer are reused unchanged.

Continuous loss guides search only. Every observed binary candidate preserves
four parity weights by rank projection. Zero-score candidates at ANY target
shift are checked by the independent integer verifier before persistence.
MPS is required by default; CPU is available explicitly for numerical tests.
"""

from dataclasses import asdict, dataclass, replace
from collections import OrderedDict
import json
import math
import os
from pathlib import Path
import random
from time import perf_counter
from typing import Callable, Dict, Optional, Sequence, Tuple

import torch

from .checkpoint import atomic_write_json
from .compressed_fkm import compressed_fkm_lift_candidates
from .compressed_pairing import pair_by_compressed_signature
from .correlation import full_correlation_profile
from .half_shift import construct_half_shift_zero_partner
from .objective import pqcp_objective, pqcp_objective_breakdown
from .search_runner import append_verified_solution_if_new
from .structured_energy import structured_energy_breakdown, target_profile
from .target_profiles import (
    TargetContentProfile,
    canonical_target_content_profiles,
    pair_content,
    target_content_profiles,
)
from .verifier import verify_pqcp


@dataclass(frozen=True)
class TorchSearchConfig:
    """Memory-bounded controls; changing device never silently falls back."""

    L: int
    seed: int = 123
    device: str = "mps"
    batch_size: int = 512
    steps_per_restart: int = 1000
    observation_interval: int = 25
    stagnation_steps: int = 300
    learning_rate: float = 0.04
    initial_temperature: float = 1.5
    final_temperature: float = 0.4
    binary_penalty: float = 1.0
    content_penalty: float = 2.0
    compression: bool = True
    fkm_pool_size: int = 8
    archive_size: int = 512
    elite_count: int = 8
    continuous_kernel: str = "direct"
    optimization_mode: str = "relaxed"
    loss_mode: str = "legacy"
    loss_backend: str = "profile"
    fast_observation: bool = False
    observation_backend: str = "torch"
    frequency_pruning: bool = False
    symmetry_pruning: bool = False
    half_shift_seeding: bool = True
    compression_signature_pairing: bool = True
    mathematical_loss: str = "none"

    def __post_init__(self) -> None:
        for name in ("L", "batch_size", "steps_per_restart", "observation_interval",
                     "stagnation_steps", "fkm_pool_size", "archive_size", "elite_count"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(name + " must be a positive integer")
        if self.L < 4 or self.L % 2:
            raise ValueError("PyTorch PQCP search requires even L >= 4")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError("seed must be an integer")
        if self.device not in ("mps", "cpu"):
            raise ValueError("device must be mps (production) or cpu (explicit testing)")
        if self.continuous_kernel not in ("direct", "dft"):
            raise ValueError("continuous_kernel must be direct or dft")
        if self.optimization_mode not in ("straight_through", "relaxed", "douglas_rachford"):
            raise ValueError("optimization_mode must be straight_through, relaxed or douglas_rachford")
        if self.optimization_mode == "douglas_rachford" and self.continuous_kernel != "dft":
            raise ValueError("douglas_rachford requires the DFT continuous_kernel")
        if self.loss_mode not in ("legacy", "balanced", "projected"):
            raise ValueError("loss_mode must be legacy, balanced or projected")
        if self.loss_backend not in ("profile", "spectral"):
            raise ValueError("loss_backend must be profile or spectral")
        if self.loss_backend == "spectral" and self.continuous_kernel != "dft":
            raise ValueError("spectral loss_backend requires the dft continuous_kernel")
        if not isinstance(self.fast_observation, bool):
            raise ValueError("fast_observation must be boolean")
        if not isinstance(self.frequency_pruning, bool):
            raise ValueError("frequency_pruning must be boolean")
        if not isinstance(self.symmetry_pruning, bool):
            raise ValueError("symmetry_pruning must be boolean")
        if not isinstance(self.half_shift_seeding, bool):
            raise ValueError("half_shift_seeding must be boolean")
        if not isinstance(self.compression_signature_pairing, bool):
            raise ValueError("compression_signature_pairing must be boolean")
        if self.mathematical_loss not in ("none", "psd_cap", "divisor_lift", "lattice", "variance", "combined", "lattice_bootstrap"):
            raise ValueError("invalid mathematical_loss")
        if self.mathematical_loss in ("psd_cap", "combined") and self.continuous_kernel != "dft":
            raise ValueError("PSD mathematical loss requires the dft continuous_kernel")
        if self.observation_backend not in ("torch", "metal"):
            raise ValueError("observation_backend must be torch or metal")
        if self.observation_backend == "metal" and (self.device != "mps" or self.L > 94):
            raise ValueError("metal observation_backend requires MPS and L<=94")
        for name in ("learning_rate", "initial_temperature", "final_temperature"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(name + " must be finite and positive")
        for name in ("binary_penalty", "content_penalty"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(name + " must be finite and non-negative")


def require_device(name: str) -> torch.device:
    """Fail clearly when Metal is unavailable; never disguise CPU as MPS."""
    if name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS 無法使用：請在支援 Metal 的 Mac 與正確的 .venv 執行；不會自動改用 CPU。")
        if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
            raise RuntimeError("請移除 PYTORCH_ENABLE_MPS_FALLBACK=1，確保搜尋計算實際在 MPS 上執行。")
    elif name != "cpu":
        raise ValueError("unsupported device")
    return torch.device(name)


class _PeriodicHalfCorrelation(torch.autograd.Function):
    """Use the analytic PACF derivative without repeated-index atomic scatter.

    d rho(x,u)/d x[p] = x[p+u] + x[p-u], including 2*x[p] at u=0.
    Gather's generic backward accumulates repeated indices in a GPU-dependent
    order. This explicit reduction gives repeatable MPS checkpoint replay and
    avoids that scatter; finite differences test the derivative independently.
    """

    @staticmethod
    def forward(ctx, signs, plus, minus):
        ctx.save_for_backward(signs, plus, minus)
        return (signs.unsqueeze(-2) * signs[..., plus]).sum(dim=(-1, -3))

    @staticmethod
    def backward(ctx, gradient):
        signs, plus, minus = ctx.saved_tensors
        neighbors = signs[..., plus] + signs[..., minus]
        return (gradient[:, None, :, None] * neighbors).sum(-2), None, None


class BatchedPQCP(torch.nn.Module):
    """Vectorized cyclic sums and PQCP loss for tensors shaped (batch,2,L).

    Pure real float32 arithmetic avoids complex FFT/device compatibility and
    float64. Cyclic indexing is cached on the selected device. Hard +/-1
    inputs have exact integer products/sums for the project's small lengths;
    Python full recomputation remains authoritative at all persistence gates.
    """

    def __init__(self, profiles: Sequence[TargetContentProfile], device: torch.device,
                 *, continuous_kernel: str = "direct"):
        super().__init__()
        if not profiles or len({p.L for p in profiles}) != 1:
            raise ValueError("a nonempty batch of equal-length profiles is required")
        self.L = profiles[0].L
        self.profiles = tuple(profiles)
        if continuous_kernel not in ("direct", "dft"):
            raise ValueError("unsupported continuous correlation kernel")
        self.continuous_kernel = continuous_kernel
        index = torch.arange(self.L, device=device)
        self.register_buffer("shifts", (index[None, :] + index[:self.L // 2 + 1, None]) % self.L)
        self.register_buffer("reverse_shifts", (index[None, :] - index[:self.L // 2 + 1, None]) % self.L)
        self.register_buffer("targets", torch.tensor(
            [target_profile(p.L, p.k, p.eta) for p in profiles], dtype=torch.float32, device=device))
        self.register_buffer("weights", torch.tensor([
            [[p.a_even_ones, p.a_odd_ones], [p.b_even_ones, p.b_odd_ones]]
            for p in profiles], dtype=torch.int64, device=device))
        self.register_buffer("ranks", torch.arange(self.L // 2, device=device))
        if continuous_kernel == "dft":
            # Constants alone are constructed on CPU in double precision, then
            # copied once. All search matmuls and gradients run on the device.
            position = torch.arange(self.L, dtype=torch.float64)
            angle = (2 * math.pi / self.L) * position[:, None] * position[None, :self.L // 2 + 1]
            cosine, sine = angle.cos(), angle.sin()
            multiplicity = torch.full((self.L // 2 + 1,), 2.0, dtype=torch.float64)
            multiplicity[0] = multiplicity[-1] = 1
            self.register_buffer("dft_real", cosine.to(device=device, dtype=torch.float32))
            self.register_buffer("dft_imag", sine.to(device=device, dtype=torch.float32))
            self.register_buffer("dft_inverse_half", (
                multiplicity[:, None] * cosine[:self.L // 2 + 1].T / self.L
            ).to(device=device, dtype=torch.float32))
            self.register_buffer("dft_multiplicity", multiplicity.to(device=device, dtype=torch.float32))
            target_spectrum = torch.tensor([
                [2 * p.L + 8 * p.eta * math.cos(2 * math.pi * f * p.k / p.L)
                 for f in range(self.L // 2 + 1)] for p in profiles
            ], dtype=torch.float32, device=device)
            self.register_buffer("target_spectrum", target_spectrum)
            frequency = torch.arange(self.L // 2 + 1)
            for mode in ("base", "legacy", "balanced"):
                spectral_weight = multiplicity / (32 * self.L)
                if mode != "base":
                    for factor in (2, 4):
                        if self.L % factor == 0:
                            coefficient = factor * factor if mode == "legacy" else 1
                            spectral_weight += coefficient * multiplicity * (frequency % factor == 0) / (16 * self.L)
                self.register_buffer("spectral_weight_" + mode, spectral_weight.to(device=device, dtype=torch.float32))

    def correlation(self, signs: torch.Tensor) -> torch.Tensor:
        """S(u)=sum_i x_i*x_(i+u)+y_i*y_(i+u), with u and L-u equal."""
        half = _PeriodicHalfCorrelation.apply(signs, self.shifts, self.reverse_shifts)
        return torch.cat((half, half[:, 1:-1].flip(-1)), dim=-1)

    def continuous_correlation(self, signs: torch.Tensor) -> torch.Tensor:
        """Evaluate the SAME PACF via optional real DFT matrix multiplication.

        For real x, rho(u)=L^-1 sum_f |DFT(x)[f]|^2 cos(2*pi*f*u/L).
        Negative frequencies duplicate positive ones, except f=0 and L/2.
        The two sequences' powers add. This replaces a B*2*(L/2+1)*L
        gather workspace with B*2*(L/2+1) spectral activations. It changes
        floating summation order, hence may change optimization trajectories.
        It is never used for discrete scoring, solution detection or storage;
        those retain exact direct correlation plus the independent verifier.
        """
        if self.continuous_kernel == "direct":
            return self.correlation(signs)
        power = ((signs @ self.dft_real).square() + (signs @ self.dft_imag).square()).sum(-2)
        half = power @ self.dft_inverse_half
        return torch.cat((half, half[:, 1:-1].flip(-1)), dim=-1)

    @torch.no_grad()
    def project(self, logits: torch.Tensor) -> torch.Tensor:
        """Select the exact number of -1 signs independently in each parity."""
        values = logits.reshape(-1, 2, self.L // 2, 2).transpose(-1, -2)
        order = torch.argsort(values, dim=-1, stable=True)
        sorted_signs = 1.0 - 2.0 * (self.ranks < self.weights.unsqueeze(-1)).to(logits.dtype)
        projected = torch.empty_like(values).scatter_(-1, order, sorted_signs)
        return projected.transpose(-1, -2).reshape_as(logits)

    def loss_components(self, signs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Per-pair full, folded, origin, content and binary squared errors.

        Full/folded energies use the existing 1/16 correlation units and the
        existing weights 2 and 4. Folding includes u=0 and correctly combines
        colliding target shifts. Origin/content/binary terms constrain relaxed
        signs; they vanish for an exact binary solution of this profile.
        """
        residual = self.continuous_correlation(signs) - self.targets
        out = {"full": residual[:, 1:self.L // 2 + 1].square().sum(-1) / 16,
               "origin": residual[:, 0].square() / 16}
        for factor in (2, 4):
            if self.L % factor == 0:
                folded = residual.reshape(-1, factor, self.L // factor).sum(1)
                out["fold{}".format(factor)] = folded.square().sum(-1) / 16
        parity_sums = signs.reshape(-1, 2, self.L // 2, 2).sum(-2)
        expected = self.L // 2 - 2 * self.weights
        out["content"] = (parity_sums - expected).square().sum((-1, -2)) / 4
        out["binary"] = (1 - signs.square()).square().sum((-1, -2))
        return out

    def projection_distance(self, signs: torch.Tensor) -> torch.Tensor:
        """Squared distance to the nearest four-content binary vertex.

        For short parity groups use parallel rank comparisons rather than a
        per-step stable sort (expensive on MPS). Ties use the same lower-index
        rule as ``project``. Only the chosen vertex is detached, not signs.
        Storage is O(batch*L^2), bounded for this project's L<=94. Larger
        experimental lengths use sorting to avoid a quadratic workspace;
        observation retains the existing general rank projection.
        """
        if self.L > 94:
            return (signs - self.project(signs)).square().sum((-1, -2))
        with torch.no_grad():
            values = signs.reshape(-1, 2, self.L // 2, 2).transpose(-1, -2)
            left, right = values.unsqueeze(-1), values.unsqueeze(-2)
            earlier = self.ranks[None, :] < self.ranks[:, None]
            rank = ((right < left) | ((right == left) & earlier)).sum(-1)
            vertex = 1 - 2 * (rank < self.weights.unsqueeze(-1)).to(signs.dtype)
            vertex = vertex.transpose(-1, -2).reshape_as(signs)
        return (signs - vertex).square().sum((-1, -2))

    def loss(self, signs: torch.Tensor, config: TorchSearchConfig) -> torch.Tensor:
        """Return one differentiable loss per lane, never the exact PQCP score.

        ``legacy`` preserves the original full + origin + 2 fold2 + 4 fold4.
        ``balanced`` divides each folded squared norm by its factor m:
        F_m F_m^T = m I, so ||F_m e||^2/m is the energy of the orthogonal
        projection onto that periodic subspace, not an m-fold amplification.
        These nested subspaces remain deliberately weighted, not independent.

        ``projected`` additionally replaces the quartic binary penalty with
        min_{z in V} ||x-z||^2, where V contains exactly the lane's four parity
        contents. Rank projection is its exact minimizer: assigning -1 instead
        of +1 changes a coordinate's squared distance by 4*x. Away from ties
        the gradient is 2*(x-project(x)); detaching this minimizer is exact,
        not a straight-through approximation. At ties we select one branch.

        All modes keep the same target, origin and soft content conditions.
        Full uses u=1..L/2 (one per reflection orbit); the target always has
        TWO nonzero actual shifts k,L-k. No folded loss replaces verification.
        With positive binary weight, zero loss on real signs implies a binary
        pair with the assigned exact target (up to numerical roundoff).
        """
        if config.loss_backend == "spectral":
            return self.spectral_loss(signs, config)
        parts = self.loss_components(signs)
        result = parts["full"] + parts["origin"]
        if config.compression:
            result = result + (2 if config.loss_mode == "legacy" else 0.5) * parts["fold2"]
            if "fold4" in parts:
                result = result + (4 if config.loss_mode == "legacy" else 0.25) * parts["fold4"]
        binary = parts["binary"]
        if config.loss_mode == "projected":
            binary = self.projection_distance(signs)
        result = result + config.content_penalty * parts["content"] + config.binary_penalty * binary
        if config.mathematical_loss != "none":
            from .torch_mathematical_loss import mathematical_structure_loss
            result = result + mathematical_structure_loss(
                self, signs, config.mathematical_loss
            )
        return result

    def spectral_loss(self, signs: torch.Tensor, config: TorchSearchConfig) -> torch.Tensor:
        """Evaluate the SAME loss by Parseval, without reconstructing PACF.

        R(f)=|X(f)|^2+|Y(f)|^2-T_hat(f), T_hat(f)=2L+8 eta cos(2pi fk/L).
        For real, periodic, symmetric e=C-T:
        H+O = (||e||^2+e(0)^2+e(L/2)^2)/32,
        ||e||^2 = sum_f multiplicity(f)*R(f)^2/L.
        Folding by m selects frequencies f divisible by m, with energy
        G_m=m/(16L)*sum_(m|f) multiplicity(f)*R(f)^2.
        Thus existing legacy/balanced coefficients and every penalty are
        preserved. e(0), e(L/2) are cheap direct sign products. This reduces
        computation, not the target or discrete verification precision.
        Floating summation changes can change a stochastic search trajectory.
        """
        real, imag = signs @ self.dft_real, signs @ self.dft_imag
        residual = (real.square() + imag.square()).sum(-2) - self.target_spectrum
        mode = "base" if not config.compression else "legacy" if config.loss_mode == "legacy" else "balanced"
        weight = getattr(self, "spectral_weight_" + mode)
        origin = signs.square().sum((-1, -2)) - 2 * self.L
        half = 2 * (signs[:, :, :self.L // 2] * signs[:, :, self.L // 2:]).sum((-1, -2))
        result = (residual.square() * weight).sum(-1) + (origin.square() + half.square()) / 32
        parity_sums = signs.reshape(-1, 2, self.L // 2, 2).sum(-2)
        content = (parity_sums - (self.L // 2 - 2 * self.weights)).square().sum((-1, -2)) / 4
        binary = (self.projection_distance(signs) if config.loss_mode == "projected"
                  else (1 - signs.square()).square().sum((-1, -2)))
        result = result + config.content_penalty * content + config.binary_penalty * binary
        if config.mathematical_loss != "none":
            from .torch_mathematical_loss import mathematical_structure_loss
            result = result + mathematical_structure_loss(
                self, signs, config.mathematical_loss
            )
        return result

    def discrete_scores(self, profile: torch.Tensor) -> torch.Tensor:
        """Batch equivalent of pqcp_objective for symmetric binary profiles."""
        side = profile[:, 1:]
        value_distance = torch.minimum(side.abs(), (side.abs() - 4).abs()).sum(-1)
        count_distance = ((side != 0).sum(-1) - 2).abs()
        return value_distance + count_distance + (profile[:, 0] - 2 * self.L).abs()

    def fixed_target_energies(self, profile: torch.Tensor, *, compression: bool = True) -> torch.Tensor:
        """Return exact discrete energy to every lane's assigned PQCP target.

        This is ``E_full + 2 E_2 + 4 E_4`` in integer correlation units.
        Unlike the target-agnostic objective, it tells the completion layer
        whether a lane actually approached its assigned ``(k, eta, content)``
        basin.  It is a ranking heuristic only; verification remains exact.
        """
        residual = profile - self.targets
        value = residual[:, 1:self.L // 2 + 1].square().sum(-1) / 16
        if compression:
            for factor, weight in ((2, 2), (4, 4)):
                if self.L % factor == 0:
                    folded = residual.reshape(-1, factor, self.L // factor).sum(1)
                    value = value + weight * folded.square().sum(-1) / 16
        return value


class TorchSearch:
    """Independent Adam lanes, FKM seeds, bounded archive and safe persistence."""

    def __init__(self, config: TorchSearchConfig, root: Path = Path("."), *, initialize: bool = True):
        self.config, self.root = config, Path(root)
        self.device = require_device(config.device)
        # Complementing either sequence and exchanging A/B preserve both
        # individual PACFs and hence the Project target.  Production
        # existence search can therefore keep exactly one content profile
        # from each such orbit.  The default remains opt-in here so older
        # checkpoints and controlled ablations retain their trajectory.
        self.available = (
            canonical_target_content_profiles(config.L)
            if config.symmetry_pruning
            else target_content_profiles(config.L, decimation_reduced=True)
        )
        if config.frequency_pruning:
            from .frequency_constraints import quarter_frequency_feasible
            self.available = tuple(p for p in self.available if quarter_frequency_feasible(p))
        if not self.available:
            raise ValueError("L={} 沒有符合既有普通／alternating sum 必要條件的 profile。".format(config.L))
        self.generation = self.epoch = self.round_step = self.restarts = 0
        self.elapsed = 0.0
        self.new_solutions = self.verified_hits = self.candidate_evaluations = 0
        self.best = None
        self.archive = {}
        self.last_observed_epoch = -1
        self._fkm_pools = {}
        self._compression_pair_cache = {}
        self._exact_payload_cache = OrderedDict()
        if initialize:
            self._initialize_batch()

    def _initial_pairs(self, profiles, generation: int):
        rng = random.Random(self.config.seed + 1000003 * generation)
        pairs = []
        for p in profiles:
            key = (p.a_even_ones, p.a_odd_ones)
            if key not in self._fkm_pools:
                self._fkm_pools[key] = compressed_fkm_lift_candidates(
                    p.L, *key, seed=self.config.seed, limit=self.config.fkm_pool_size)
            pool = self._fkm_pools[key]
            if not pool:
                raise RuntimeError("壓縮 FKM 無法為合法 content 產生 seed")
            exact_pairs = ()
            if self.config.compression_signature_pairing:
                if p not in self._compression_pair_cache:
                    b_pool = compressed_fkm_lift_candidates(
                        p.L, p.b_even_ones, p.b_odd_ones,
                        seed=self.config.seed + 1_000_000_007,
                        limit=self.config.fkm_pool_size,
                    )
                    matches = pair_by_compressed_signature(
                        pool, b_pool, p, limit=self.config.fkm_pool_size,
                        general_psd_pruning=self.config.frequency_pruning,
                    )
                    self._compression_pair_cache[p] = tuple(
                        (match.a, match.b) for match in matches
                    )
                exact_pairs = self._compression_pair_cache[p]
            if exact_pairs:
                # A full signature match satisfies every factor-2/factor-4
                # compressed target coordinate exactly before optimization.
                a, b = rng.choice(exact_pairs)
            elif self.config.half_shift_seeding:
                # The half shift is self-symmetric and cannot be one of the
                # two nonzero target shifts. Select an FKM A whose exact-
                # content B partner satisfies S(L/2)=0, then construct that
                # partner combinatorially. This is an exact necessary
                # condition; no known PQCP or empirical score is consulted.
                start = rng.randrange(len(pool))
                a = b = None
                for offset in range(len(pool)):
                    candidate = pool[(start + offset) % len(pool)]
                    partner = construct_half_shift_zero_partner(
                        candidate, p.b_even_ones, p.b_odd_ones, rng
                    )
                    if partner is not None:
                        a, b = candidate, partner
                        break
                if a is None or b is None:
                    raise RuntimeError(
                        "壓縮 FKM seed pool 無法滿足必要的 S(L/2)=0 條件"
                    )
            else:
                # Explicit ablation/rollback path: retain exact parity
                # content but do not force the half-shift coordinate.
                b_key = (p.b_even_ones, p.b_odd_ones)
                if b_key not in self._fkm_pools:
                    self._fkm_pools[b_key] = compressed_fkm_lift_candidates(
                        p.L, *b_key, seed=self.config.seed,
                        limit=self.config.fkm_pool_size,
                    )
                b_pool = self._fkm_pools[b_key]
                if not b_pool:
                    raise RuntimeError("壓縮 FKM 無法為 B content 產生 seed")
                a, b = rng.choice(pool), rng.choice(b_pool)
            pairs.append([a, b])
        return torch.tensor(pairs, dtype=torch.float32)

    def _initialize_batch(self) -> None:
        """Rebirth recreates parameters AND Adam moments, preserving no stale momentum."""
        c = self.config
        order = list(self.available)
        random.Random(c.seed).shuffle(order)
        profiles = tuple(order[(self.generation * c.batch_size + i) % len(order)]
                         for i in range(c.batch_size))
        self.model = BatchedPQCP(profiles, self.device, continuous_kernel=c.continuous_kernel)
        bits = self._initial_pairs(profiles, self.generation)
        generator = torch.Generator(device="cpu").manual_seed(
            (c.seed + self.generation * 1000003) % (2 ** 63))
        # Local RNG keeps CPU verification/completion independent of search.
        jitter = torch.randn(bits.shape, generator=generator) * 0.15
        self.theta = torch.nn.Parameter(((1 - 2 * bits) * 0.8 + jitter).to(self.device))
        self.optimizer = torch.optim.Adam([self.theta], lr=c.learning_rate)
        self.round_step = 0
        self.restarts += c.batch_size
        self.lane_best = [None] * c.batch_size
        self.last_improved = [self.epoch] * c.batch_size

    def step(self) -> None:
        """Perform one batched forward/backward/Adam update on the selected device.

        ``straight_through`` fixes the former relaxation gap.  Its forward
        pass is the exact content-preserving +/-1 rank projection, so Adam is
        always scored on a real binary candidate.  Backpropagation uses the
        derivative of ``tanh(theta / temperature)`` (the standard
        straight-through construction); projection ordering itself is not
        differentiated.  ``relaxed`` preserves the old continuous-forward
        experiment and remains available for checkpoint compatibility and
        controlled ablation.
        """
        c = self.config
        if c.optimization_mode == "douglas_rachford":
            from .torch_feasibility import douglas_rachford_step
            with torch.no_grad():
                self.theta.copy_(douglas_rachford_step(
                    self.model, self.theta, metal=c.observation_backend == "metal"))
            self.epoch += 1
            self.round_step += 1
            return
        fraction = self.round_step / max(1, c.steps_per_restart - 1)
        temperature = c.initial_temperature * (c.final_temperature / c.initial_temperature) ** fraction
        period = max(1, c.steps_per_restart // 4)
        lr_fraction = (self.round_step % period) / period
        self.optimizer.param_groups[0]["lr"] = c.learning_rate * (0.1 + 0.9 * (1 + math.cos(math.pi * lr_fraction)) / 2)
        self.optimizer.zero_grad(set_to_none=True)
        relaxed = torch.tanh(self.theta / temperature)
        if c.optimization_mode == "straight_through":
            hard = self.model.project(self.theta)
            signs = relaxed + (hard - relaxed).detach()
        else:
            signs = relaxed
        # One structural cosine cycle first, then three exact-target cycles.
        # This is a two-phase objective, not a fractional weight adjustment.
        loss_config = c
        if c.mathematical_loss == "lattice_bootstrap":
            loss_config = replace(
                c, mathematical_loss="lattice" if self.round_step < period else "none"
            )
        loss = self.model.loss(signs, loss_config).sum()
        loss.backward()
        self.optimizer.step()
        self.epoch += 1
        self.round_step += 1

    def _payload(self, a, b, lane: int, expected_score: int,
                 expected_target_energy: Optional[int] = None) -> dict:
        """Recompute integers before trusting any accelerator score or profile."""
        p = self.model.profiles[lane]
        cache_key = (tuple(a), tuple(b), p)
        if self.config.fast_observation and expected_score != 0 and cache_key in self._exact_payload_cache:
            cached = self._exact_payload_cache[cache_key]
            if cached["score"] != expected_score:
                raise RuntimeError("cached exact score disagrees with accelerator")
            if expected_target_energy is not None and cached["target_energy"] != expected_target_energy:
                raise RuntimeError("cached target energy disagrees with accelerator")
            self._exact_payload_cache.move_to_end(cache_key)
            return {**cached, "iteration": self.epoch, "restart": self.generation,
                    "lane": lane, "elapsed": self.elapsed}
        # The verifier already computes the full integer PACF independently.
        # Reuse its result instead of computing the same profile twice.
        verification = verify_pqcp(a, b) if self.config.fast_observation else None
        profile = tuple(verification.profile if verification is not None else full_correlation_profile(a, b))
        score = pqcp_objective(profile)
        if score != expected_score:
            raise RuntimeError("MPS discrete score disagrees with full Python correlation")
        if pair_content(a, b) != (p.a_even_ones, p.a_odd_ones, p.b_even_ones, p.b_odd_ones):
            raise RuntimeError("rank projection changed required parity content")
        if verification is None:
            verification = verify_pqcp(a, b)
        if score == 0 and not verification.is_valid:
            raise RuntimeError("score zero failed independent PQCP verification")
        nonzero = list(verification.nonzero_shifts)
        factors = tuple(factor for factor in (2, 4) if self.config.L % factor == 0)
        weights = {
            factor: weight for factor, weight in ((2, 2), (4, 4))
            if factor in factors
        }
        target_energy = structured_energy_breakdown(
            profile, p.k, p.eta, factors
        ).weighted_total(weights)
        if expected_target_energy is not None and target_energy != expected_target_energy:
            raise RuntimeError("MPS target energy disagrees with full Python recomputation")
        payload = {"L": self.config.L, "method": "torch_mps" if self.device.type == "mps" else "torch_cpu_test",
                "seed": self.config.seed, "iteration": self.epoch, "restart": self.generation,
                "lane": lane, "elapsed": self.elapsed, "A": list(a), "B": list(b),
                "score": score, "profile": list(profile), "verified": verification.is_valid,
                "nonzero_shifts": nonzero, "nonzero_values": [profile[u] for u in nonzero],
                "objective_components": pqcp_objective_breakdown(profile), "k": p.k, "sign": p.eta,
                "target_energy": target_energy, "loss_mode": self.config.loss_mode,
                "loss_backend": self.config.loss_backend, "optimization_mode": self.config.optimization_mode,
                "mathematical_loss": self.config.mathematical_loss}
        # Bounded, ephemeral proof cache keyed by ALL bits and target metadata.
        # Zero candidates always go through a fresh verifier invocation.
        if self.config.fast_observation and not payload["verified"]:
            self._exact_payload_cache[cache_key] = payload
            while len(self._exact_payload_cache) > self.config.archive_size * 4:
                self._exact_payload_cache.popitem(last=False)
        return payload

    @torch.no_grad()
    def observe(self, elite_callback: Optional[Callable] = None, *, update_stagnation: bool = True,
                solution_callback: Optional[Callable] = None) -> None:
        """Inspect ALL quantized lanes; zero scores are never limited to top K."""
        c = self.config
        if c.observation_backend == "metal":
            from .torch_metal import project_and_correlate
            signs, profiles = project_and_correlate(self.theta, self.model.weights)
        else:
            signs = self.model.project(self.theta)
            profiles = self.model.correlation(signs)
        scores = self.model.discrete_scores(profiles).cpu().tolist()
        target_energies = self.model.fixed_target_energies(
            profiles, compression=c.compression
        ).to(torch.int64).cpu().tolist()
        if not all(math.isfinite(s) and s == int(s) for s in scores):
            raise RuntimeError("non-finite or non-integer discrete correlation score")
        if not bool(torch.isfinite(self.theta).all().item()):
            raise RuntimeError("non-finite PyTorch parameters")
        self.candidate_evaluations += c.batch_size
        if update_stagnation:
            for i, score in enumerate(scores):
                if self.lane_best[i] is None or score < self.lane_best[i]:
                    self.lane_best[i], self.last_improved[i] = score, self.epoch
        ranked = sorted(range(c.batch_size), key=lambda i: (scores[i], i))
        target_ranked = sorted(
            range(c.batch_size), key=lambda i: (target_energies[i], scores[i], i)
        )
        selected = sorted(
            set(ranked[:c.elite_count])
            | set(target_ranked[:c.elite_count])
            | {i for i, s in enumerate(scores) if s == 0}
        )
        bits = ((1 - signs[selected]) / 2).to(torch.int32).cpu().tolist()
        gpu_profiles = profiles[selected].cpu().tolist()
        for lane, pair, gpu_profile in zip(selected, bits, gpu_profiles):
            a, b = tuple(pair[0]), tuple(pair[1])
            payload = self._payload(a, b, lane, int(scores[lane]))
            if payload["target_energy"] != int(target_energies[lane]):
                raise RuntimeError("MPS target energy disagrees with full Python recomputation")
            if payload["profile"] != gpu_profile:
                raise RuntimeError("MPS profile disagrees with full Python correlation")
            key = tuple(sorted((a, b)))  # exact/A-B-swap only, no orbit farming
            is_new = False
            if payload["verified"]:
                self.verified_hits += 1
                is_new = append_verified_solution_if_new(c.L, a, b, self.root)
                if is_new:
                    self.new_solutions += 1
                    directory = self.root / "results/verified"
                    name = "L{}_torch_seed{}_epoch{}_lane{}.json".format(c.L, c.seed, self.epoch, lane)
                    destination = directory / name
                    suffix = 0
                    while destination.exists():
                        suffix += 1
                        destination = directory / (Path(name).stem + "_{}.json".format(suffix))
                    atomic_write_json(destination, payload)
                    print("找到新的 ({},4)-PQCP，已寫入 {}.txt".format(c.L, c.L), flush=True)
            if self.best is None or payload["score"] < self.best["score"]:
                old = None if self.best is None else self.best["score"]
                self.best = payload
                atomic_write_json(self.best_path, payload)
                history = self.root / "logs" / "torch_L{}_seed{}.jsonl".format(c.L, c.seed)
                history.parent.mkdir(parents=True, exist_ok=True)
                with history.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({**payload, "old_best_score": old}, ensure_ascii=False) + "\n")
            if key not in self.archive:
                self.archive[key] = payload
                if elite_callback is not None and payload["score"] > 0:
                    elite_callback(a, b, {"k": payload["k"], "sign": payload["sign"], "score": payload["score"]})
            if is_new and solution_callback is not None:
                solution_callback(self)
        self.archive = dict(sorted(
            self.archive.items(),
            key=lambda item: (item[1]["target_energy"], item[1]["score"]),
        )[:c.archive_size])
        self.last_observed_epoch = self.epoch

    @property
    def best_path(self) -> Path:
        return self.root / "results/best" / "L{}_torch_seed{}.json".format(self.config.L, self.config.seed)

    def checkpoint(self, path: Path) -> None:
        """Save JSON tensors and Adam moments; no pickle or executable payload."""
        state = self.optimizer.state.get(self.theta, {})
        optimizer = {key: value.detach().cpu().tolist() for key, value in state.items()}
        atomic_write_json(path, {
            "format": "pqcp-torch-search", "version": 1, "torch_version": torch.__version__,
            "config": asdict(self.config), "profiles": [asdict(p) for p in self.model.profiles],
            "theta": self.theta.detach().cpu().tolist(), "adam": optimizer,
            "generation": self.generation, "epoch": self.epoch, "round_step": self.round_step,
            "restarts": self.restarts, "elapsed": self.elapsed, "best": self.best,
            "archive": list(self.archive.values()), "lane_best": self.lane_best,
            "last_improved": self.last_improved, "last_observed_epoch": self.last_observed_epoch,
            "new_solutions": self.new_solutions, "verified_hits": self.verified_hits,
            "candidate_evaluations": self.candidate_evaluations,
        })

    @classmethod
    def resume(cls, path: Path, root: Path = Path(".")) -> "TorchSearch":
        """Restore same-device matrix continuation without reinitializing seeds."""
        try:
            with Path(path).open(encoding="utf-8") as handle:
                data = json.load(handle)
            if data["format"] != "pqcp-torch-search" or data["version"] != 1:
                raise ValueError("unsupported PyTorch checkpoint format")
            config_data = dict(data["config"])
            # Version-one checkpoints written before binary-forward search
            # must resume the exact old relaxed trajectory.
            config_data.setdefault("optimization_mode", "relaxed")
            config_data.setdefault("loss_mode", "legacy")
            config_data.setdefault("loss_backend", "profile")
            obj = cls(TorchSearchConfig(**config_data), root, initialize=False)
            profiles = tuple(TargetContentProfile(**p) for p in data["profiles"])
            if len(profiles) != obj.config.batch_size or any(p not in obj.available for p in profiles):
                raise ValueError("invalid checkpoint profiles")
            obj.model = BatchedPQCP(profiles, obj.device, continuous_kernel=obj.config.continuous_kernel)
            tensor = torch.tensor(data["theta"], dtype=torch.float32, device=obj.device)
            if tuple(tensor.shape) != (obj.config.batch_size, 2, obj.config.L) or not torch.isfinite(tensor).all().item():
                raise ValueError("invalid checkpoint parameter tensor")
            obj.theta = torch.nn.Parameter(tensor)
            obj.optimizer = torch.optim.Adam([obj.theta], lr=obj.config.learning_rate)
            if data["adam"]:
                if set(data["adam"]) != {"step", "exp_avg", "exp_avg_sq"}:
                    raise ValueError("invalid Adam state")
                for key, value in data["adam"].items():
                    item = torch.tensor(value, dtype=torch.float32, device="cpu" if key == "step" else obj.device)
                    if (key != "step" and item.shape != tensor.shape) or not torch.isfinite(item).all().item():
                        raise ValueError("invalid Adam moment tensor")
                    obj.optimizer.state[obj.theta][key] = item
            for key in ("generation", "epoch", "round_step", "restarts", "elapsed", "best",
                        "lane_best", "last_improved", "last_observed_epoch", "new_solutions",
                        "verified_hits", "candidate_evaluations"):
                setattr(obj, key, data[key])
            for key in ("generation", "epoch", "round_step", "restarts", "new_solutions",
                        "verified_hits", "candidate_evaluations"):
                value = getattr(obj, key)
                if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                    raise ValueError("invalid checkpoint counter: " + key)
            if not isinstance(obj.elapsed, (int, float)) or not math.isfinite(obj.elapsed) or obj.elapsed < 0:
                raise ValueError("invalid elapsed time")
            if obj.round_step > obj.config.steps_per_restart:
                raise ValueError("invalid restart step")
            step = obj.optimizer.state.get(obj.theta, {}).get("step")
            if obj.config.optimization_mode == "douglas_rachford":
                if data["adam"]:
                    raise ValueError("feasibility checkpoint must not contain Adam state")
            elif (step is None and obj.round_step != 0) or (step is not None and (step.ndim != 0 or step.item() != obj.round_step)):
                raise ValueError("Adam step does not match restart step")
            if len(obj.lane_best) != obj.config.batch_size or len(obj.last_improved) != obj.config.batch_size:
                raise ValueError("invalid checkpoint lane metadata")
            if any(not isinstance(t, int) or not 0 <= t <= obj.epoch for t in obj.last_improved):
                raise ValueError("invalid lane improvement epoch")
            if any(s is not None and (not isinstance(s, (int, float)) or not math.isfinite(s) or s < 0) for s in obj.lane_best):
                raise ValueError("invalid lane score")
            if obj.best is not None:
                exact = full_correlation_profile(obj.best["A"], obj.best["B"])
                if exact != obj.best["profile"] or pqcp_objective(exact) != obj.best["score"]:
                    raise ValueError("checkpoint best candidate fails correlation consistency")
            restored_archive = {}
            for payload in data["archive"]:
                if "target_energy" not in payload:
                    factors = tuple(
                        factor for factor in (2, 4) if obj.config.L % factor == 0
                    )
                    weights = {
                        factor: weight for factor, weight in ((2, 2), (4, 4))
                        if factor in factors
                    }
                    payload["target_energy"] = structured_energy_breakdown(
                        payload["profile"], payload["k"], payload["sign"], factors
                    ).weighted_total(weights)
                restored_archive[
                    tuple(sorted((tuple(payload["A"]), tuple(payload["B"]))))
                ] = payload
            obj.archive = restored_archive
            return obj
        except (KeyError, TypeError, ValueError, OSError) as error:
            raise ValueError("Invalid PyTorch checkpoint: {}".format(error)) from error

    def run(self, seconds: float = float("inf"), *, checkpoint_path: Optional[Path] = None,
            checkpoint_interval: float = 60, progress_interval: float = 10,
            progress: Optional[Callable] = None, elite_callback: Optional[Callable] = None,
            max_steps: Optional[int] = None) -> dict:
        """Run until Ctrl+C/budget, persist on exit, and continue after every solution."""
        if math.isnan(seconds) or seconds < 0 or checkpoint_interval <= 0 or progress_interval <= 0:
            raise ValueError("invalid time budget or observation interval")
        if max_steps is not None and (not isinstance(max_steps, int) or max_steps < 0):
            raise ValueError("max_steps must be non-negative")
        destination = checkpoint_path or self.root / "checkpoints" / "L{}_torch.json".format(self.config.L)
        started, prior, steps = perf_counter(), self.elapsed, 0
        last_save = last_progress = started
        interrupted = False
        try:
            if self.last_observed_epoch != self.epoch:
                self.observe(elite_callback, solution_callback=progress)
            while perf_counter() - started < seconds and (max_steps is None or steps < max_steps):
                stale = all(self.epoch - t >= self.config.stagnation_steps for t in self.last_improved)
                if self.round_step >= self.config.steps_per_restart or stale:
                    self.generation += 1
                    self._initialize_batch()
                    self.elapsed = prior + perf_counter() - started
                    self.observe(elite_callback, solution_callback=progress)
                self.step()
                steps += 1
                now = perf_counter()
                self.elapsed = prior + now - started
                if self.epoch % self.config.observation_interval == 0:
                    self.observe(elite_callback, solution_callback=progress)
                if now - last_save >= checkpoint_interval:
                    self.checkpoint(destination)
                    last_save = perf_counter()
                if now - last_progress >= progress_interval:
                    if progress is not None:
                        progress(self)
                    last_progress = perf_counter()
        except KeyboardInterrupt:
            interrupted = True
        finally:
            self.elapsed = prior + perf_counter() - started
            try:
                if self.last_observed_epoch != self.epoch:
                    # A stop between scheduled observations must not change
                    # the future stagnation/rebirth decisions on resume.
                    self.observe(elite_callback, update_stagnation=False, solution_callback=progress)
            finally:
                self.checkpoint(destination)
        return {"interrupted": interrupted, "checkpoint": str(destination),
                "device": str(self.device), "epoch": self.epoch, "restarts": self.restarts,
                "best_score": self.best["score"] if self.best else None,
                "new_solutions": self.new_solutions, "elapsed": self.elapsed,
                "candidate_evaluations": self.candidate_evaluations}
