"""Differentiable necessary-structure losses for the ORIGINAL PQCP target.

These terms guide continuous MPS optimization only.  They never prune a
binary candidate or replace exact integer correlation and the independent
verifier.  Every component is nonnegative and vanishes for an exact pair.

The PSD cap follows from Wiener--Khinchin: at a solution
``PSD_A(f)+PSD_B(f)=DFT(T)(f)`` and both individual PSDs are nonnegative.
Compression uses the exact identity that factor-m compression folds PACF
residual classes.  Liftability uses the binary compression alphabet
``{-m,-m+2,...,m}``; it is necessary but very far from sufficient.  Finally,
a binary sequence of weight w has individual periodic autocorrelations
``rho(u)=L-4h`` with ``0<=h<=min(w,L-w)``.  The last lattice constraint stops
the relaxed pair sum from hiding an individually unrealizable component.
The Bernoulli variance term treats relaxed signs as exact means of independent
random signs and supplies the missing variance in expected squared PACF error.
"""

from typing import Dict, Tuple

import torch


def proper_compression_factors(length: int) -> Tuple[int, ...]:
    """Return every nontrivial compression factor with output length >= 2."""
    if not isinstance(length, int) or isinstance(length, bool) or length < 4:
        raise ValueError("length must be an integer at least four")
    return tuple(factor for factor in range(2, length)
                 if length % factor == 0 and length // factor >= 2)


def new_compression_factors(length: int) -> Tuple[int, ...]:
    """Return mathematically valid factors absent from the old 2/4 loss."""
    return tuple(factor for factor in proper_compression_factors(length)
                 if factor not in (2, 4))


def individual_psd_cap_loss(model, signs: torch.Tensor) -> torch.Tensor:
    """Penalize violation of each sequence's necessary target PSD upper bound.

    For this Project 2 target,
    ``T_hat(f)=2L+8*eta*cos(2*pi*f*k/L)``.  At an exact pair the other
    sequence's PSD is nonnegative, hence each PSD is at most ``T_hat``.
    Squared excess is converted to correlation-energy units by Parseval,
    using frequency multiplicities and ``1/(32L)`` as in the base loss.
    """
    if model.continuous_kernel != "dft":
        raise ValueError("PSD-cap loss requires the DFT continuous kernel")
    real, imag = signs @ model.dft_real, signs @ model.dft_imag
    individual = real.square() + imag.square()
    return _individual_psd_cap_from_power(model, individual)


def _individual_psd_cap_from_power(model, individual: torch.Tensor) -> torch.Tensor:
    excess = torch.relu(individual - model.target_spectrum[:, None, :])
    return (excess.square() * model.dft_multiplicity).sum((-1, -2)) / (32 * model.L)


def additional_divisor_fold_loss(model, signs: torch.Tensor) -> torch.Tensor:
    """Mean normalized PACF-fold energy for factors not already 2 or 4.

    If ``F_m`` sums residuals in equal residue classes, ``F_m F_m^T=mI``.
    Therefore ``||F_m e||^2/m`` is the squared norm of the orthogonal
    projection onto the factor-m compression subspace.  Division by 16 keeps
    the original binary correlation units; averaging avoids making lengths
    with many divisors receive an arbitrary larger coefficient.
    """
    factors = new_compression_factors(model.L)
    if not factors:
        return signs.new_zeros(signs.shape[0])
    residual = model.continuous_correlation(signs) - model.targets
    return _additional_divisor_fold_from_residual(model, residual, factors)


def _additional_divisor_fold_from_residual(model, residual: torch.Tensor,
                                            factors: Tuple[int, ...]) -> torch.Tensor:
    energies = []
    for factor in factors:
        folded = residual.reshape(-1, factor, model.L // factor).sum(1)
        energies.append(folded.square().sum(-1) / (16 * factor))
    return torch.stack(energies).mean(0)


def compression_liftability_loss(signs: torch.Tensor) -> torch.Tensor:
    """Distance of every proper compression to its binary lift alphabet.

    A factor-m compression sums m signs, so every entry must be one of
    ``-m,-m+2,...,m``.  For the relaxed values, project each compressed entry
    to its nearest alphabet value and detach that piecewise-constant choice.
    ``||c-P(c)||^2/m`` is normalized by the compression operator norm.  The
    mean over factors prevents divisor count alone from scaling the loss.
    """
    if signs.ndim != 3 or signs.shape[1] != 2:
        raise ValueError("signs must have shape batch,2,L")
    length = signs.shape[-1]
    factors = proper_compression_factors(length)
    if not factors:
        return signs.new_zeros(signs.shape[0])
    energies = []
    for factor in factors:
        compressed = signs.reshape(-1, 2, factor, length // factor).sum(2)
        rank = ((compressed + factor) / 2).round().clamp(0, factor)
        nearest = (2 * rank - factor).detach()
        energies.append((compressed - nearest).square().sum((-1, -2)) / factor)
    return torch.stack(energies).mean(0)


def individual_pacf_lattice_loss(model, signs: torch.Tensor) -> torch.Tensor:
    """Penalize distance from each sequence's exact fixed-weight PACF lattice.

    If a binary sequence has weight ``w`` and its support overlaps its shift
    in ``q`` positions, then

    ``rho(u) = L - 4*(w-q)``.

    Here ``h=w-q`` is an integer between zero and ``min(w,L-w)``.  Thus every
    individual PACF value belongs to the finite set
    ``{L-4h : h=0..min(w,L-w)}``.  This is stronger guidance than constraining
    only the pair sum in the continuous relaxation, but it vanishes for every
    fixed-weight binary sequence and therefore cannot redefine a PQCP.

    Projection to the nearest lattice point is detached.  Division by 16
    expresses squared distances in the same four-unit correlation scale as
    the base fixed-target loss.  Only one shift from each reflection orbit is
    counted, exactly as in the base loss.
    """
    if signs.ndim != 3 or signs.shape[1:] != (2, model.L):
        raise ValueError("signs must have shape batch,2,L matching the model")
    if model.continuous_kernel == "dft":
        real, imag = signs @ model.dft_real, signs @ model.dft_imag
        individual_half = (real.square() + imag.square()) @ model.dft_inverse_half
    else:
        individual_half = (
            signs.unsqueeze(-2) * signs[..., model.shifts]
        ).sum(-1)
    weights = model.weights.sum(-1)
    maximum_h = torch.minimum(weights, model.L - weights).to(signs.dtype)
    h = (model.L - individual_half) / 4
    nearest_h = h.round().clamp_min(0)
    nearest_h = torch.minimum(nearest_h, maximum_h.unsqueeze(-1)).detach()
    nearest_rho = model.L - 4 * nearest_h
    return (individual_half[..., 1:] - nearest_rho[..., 1:]).square().sum((-1, -2)) / 16


def bernoulli_pacf_variance_loss(model, signs: torch.Tensor) -> torch.Tensor:
    """Return the exact variance part of randomized binary PACF squared loss.

    Let independent random signs ``X_i`` have means equal to the relaxed
    values ``m_i``.  For a non-half shift, write ``Y_i=X_i X_(i+u)``.
    Nonadjacent cycle edges are independent and adjacent edges share one bit,
    giving

    ``Var rho(u) = sum_i(1-m_i^2*m_(i+u)^2)``
    ``             + 2 sum_i m_i*m_(i+2u)*(1-m_(i+u)^2)``.

    At ``u=L/2`` each undirected edge occurs twice, so its variance is
    ``4 sum_(i<L/2)(1-m_i^2*m_(i+L/2)^2)``.  A and B are independent, hence
    their variances add.  Consequently adding this function to the existing
    squared mean residual gives the exact
    ``E[(C(u)-T(u))^2]/16`` over independent nonzero shifts.  It is zero for
    every binary pair, independent of whether that pair is a PQCP.
    """
    if signs.ndim != 3 or signs.shape[1:] != (2, model.L):
        raise ValueError("signs must have shape batch,2,L matching the model")
    if model.continuous_kernel == "dft":
        real, imag = signs @ model.dft_real, signs @ model.dft_imag
        return _bernoulli_variance_from_fourier(model, signs, real, imag)
    return _bernoulli_variance_direct(model, signs)


def _bernoulli_variance_direct(model, signs: torch.Tensor) -> torch.Tensor:
    """Reference implementation retaining the shift-by-shift derivation."""
    half = model.L // 2
    components = []
    if half > 1:
        shift_index = model.shifts[1:half]
        twice_index = (shift_index + shift_index[:, :1]) % model.L
        shifted = signs[..., shift_index]
        twice_shifted = signs[..., twice_index]
        edge_variance = (1 - signs.unsqueeze(-2).square() * shifted.square()).sum(-1)
        adjacent_covariance = (
            signs.unsqueeze(-2) * twice_shifted * (1 - shifted.square())
        ).sum(-1)
        components.append((edge_variance + 2 * adjacent_covariance).sum((-1, -2)))
    left, right = signs[..., :half], signs[..., half:]
    components.append((4 * (1 - left.square() * right.square())).sum((-1, -2)))
    return sum(components, signs.new_zeros(signs.shape[0])) / 16


def _bernoulli_variance_from_fourier(model, signs: torch.Tensor,
                                     real: torch.Tensor, imag: torch.Tensor) -> torch.Tensor:
    """Equivalent spectral-matrix path without a batch-by-L-by-L tape.

    The adjacent-edge covariance summed over all shifts contains the cyclic
    self-convolution ``(m*m)[2j]``.  Reconstructing that convolution from the
    already available Fourier coefficients is substantially cheaper on MPS
    than retaining every shifted product for backward propagation.
    """
    length, half = model.L, model.L // 2
    square = signs.square()
    fourth = square.square()
    term1_all = length * (length - 1) - (
        square.sum(-1).square() - fourth.sum(-1)
    )
    convolution_real = real.square() - imag.square()
    convolution_imag = 2 * real * imag
    inverse_sine = (
        model.dft_multiplicity[:, None] * model.dft_imag.transpose(0, 1) / length
    )
    inverse_cosine = (
        model.dft_multiplicity[:, None] * model.dft_real.transpose(0, 1) / length
    )
    convolution = (
        convolution_real @ inverse_cosine
        + convolution_imag @ inverse_sine
    )
    twice = (2 * torch.arange(length, device=signs.device)) % length
    covariance_sum = ((1 - square) * (convolution[..., twice] - square)).sum(-1)
    general_all = term1_all + 2 * covariance_sum

    shifted_half = square.roll(half, dims=-1)
    term1_half = (1 - square * shifted_half).sum(-1)
    covariance_half_general = (square * (1 - shifted_half)).sum(-1)
    actual_half = 2 * term1_half
    independent = (
        general_all - (term1_half + 2 * covariance_half_general) + actual_half
        + actual_half
    ) / 2
    return independent.sum(-1) / 16


def mathematical_loss_components(model, signs: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Return all terms while sharing one Fourier transform and one residual."""
    if model.continuous_kernel != "dft":
        raise ValueError("mathematical loss components require the DFT kernel")
    real, imag = signs @ model.dft_real, signs @ model.dft_imag
    individual = real.square() + imag.square()
    spectral_residual = individual.sum(-2) - model.target_spectrum
    half = spectral_residual @ model.dft_inverse_half
    residual = torch.cat((half, half[:, 1:-1].flip(-1)), dim=-1)
    factors = new_compression_factors(model.L)
    return {
        "psd_cap": _individual_psd_cap_from_power(model, individual),
        "divisor_fold": (_additional_divisor_fold_from_residual(model, residual, factors)
                         if factors else signs.new_zeros(signs.shape[0])),
        "liftability": compression_liftability_loss(signs),
        "pacf_lattice": _individual_pacf_lattice_from_power(model, individual),
        "bernoulli_variance": _bernoulli_variance_from_fourier(
            model, signs, real, imag
        ),
    }


def _individual_pacf_lattice_from_power(model, individual: torch.Tensor) -> torch.Tensor:
    half = individual @ model.dft_inverse_half
    weights = model.weights.sum(-1)
    maximum_h = torch.minimum(weights, model.L - weights).to(half.dtype)
    h = (model.L - half) / 4
    nearest_h = torch.minimum(h.round().clamp_min(0), maximum_h.unsqueeze(-1)).detach()
    nearest_rho = model.L - 4 * nearest_h
    return (half[..., 1:] - nearest_rho[..., 1:]).square().sum((-1, -2)) / 16


def mathematical_structure_loss(model, signs: torch.Tensor, mode: str) -> torch.Tensor:
    """Select one structural ablation or their combined sum."""
    if mode == "none":
        return signs.new_zeros(signs.shape[0])
    if mode not in ("psd_cap", "divisor_lift", "lattice", "variance", "combined",
                    "lattice_bootstrap"):
        raise ValueError("unknown mathematical loss mode")
    if mode == "psd_cap":
        return individual_psd_cap_loss(model, signs)
    if mode == "divisor_lift":
        return additional_divisor_fold_loss(model, signs) + compression_liftability_loss(signs)
    if mode in ("lattice", "lattice_bootstrap"):
        return individual_pacf_lattice_loss(model, signs)
    if mode == "variance":
        return bernoulli_pacf_variance_loss(model, signs)
    if mode == "combined":
        parts = mathematical_loss_components(model, signs)
        return sum(parts.values(), signs.new_zeros(signs.shape[0]))
    raise AssertionError("validated mathematical loss mode was not handled")


__all__ = (
    "additional_divisor_fold_loss", "compression_liftability_loss",
    "bernoulli_pacf_variance_loss", "individual_pacf_lattice_loss",
    "individual_psd_cap_loss", "mathematical_loss_components",
    "mathematical_structure_loss", "new_compression_factors",
    "proper_compression_factors",
)
