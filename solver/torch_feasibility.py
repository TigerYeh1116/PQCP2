"""Experimental real-MPS feasibility projections for the ORIGINAL PQCP target.

Adapted feasibility idea: Aragon Artacho et al., arXiv:1711.02502.
The formulas here are rederived for T(0)=2L, T(k)=T(L-k)=4 eta.
This is not a convergence guarantee for the nonconvex binary intersection.
No existing PQCP, floating-point solution test or different target is used.
"""

import torch


@torch.no_grad()
def project_target_spectrum(model, values: torch.Tensor) -> torch.Tensor:
    """Nearest real pair with the assigned PACF and DC/Nyquist row sums.

    Interior Fourier blocks (Re A, Im A, Re B, Im B) lie on spheres with
    radius sqrt(2L+8 eta cos(2pi fk/L)). Radial projection is Euclidean-nearest;
    Parseval's multiplicity is constant within each block. A zero block has
    multiple nearest points: deterministically use positive real A, zero B.
    DC and Nyquist instead have fixed individual signed sums, determined by
    the four contents. Their squared sums equal the required spectral power.
    Hence all frequency blocks are independent and their product projection
    is exact in real arithmetic. Small floating error is NEVER verification.

    The model stores positive-sine Fourier coordinates, so inverse synthesis
    uses PLUS sine. Multiplicity is 1 at 0,L/2 and 2 at interior frequencies.
    """
    if model.continuous_kernel != "dft":
        raise ValueError("spectral feasibility requires a DFT model")
    if values.shape != (len(model.profiles), 2, model.L):
        raise ValueError("values must match batch,2,L")
    real, imag = values @ model.dft_real, values @ model.dft_imag
    power = (real.square() + imag.square()).sum(-2)
    target = model.target_spectrum
    norm = torch.where(power > 0, power.sqrt(), torch.ones_like(power))
    scale = target.sqrt() / norm
    real, imag = real * scale[:, None, :], imag * scale[:, None, :]
    zero = power == 0
    real[:, 0, :] = torch.where(zero, target.sqrt(), real[:, 0, :])
    real[:, 1, :] = torch.where(zero, 0., real[:, 1, :])
    imag = torch.where(zero[:, None, :], 0., imag)
    real[:, :, 0] = model.L - 2 * model.weights.sum(-1)
    real[:, :, -1] = 2 * (model.weights[:, :, 1] - model.weights[:, :, 0])
    imag[:, :, 0] = 0
    imag[:, :, -1] = 0
    multiplicity = model.dft_multiplicity
    return ((real * multiplicity) @ model.dft_real.T +
            (imag * multiplicity) @ model.dft_imag.T) / model.L


@torch.no_grad()
def douglas_rachford_step(model, values: torch.Tensor, *, metal: bool = False) -> torch.Tensor:
    """x_next = x + P_spectrum(2 P_binary(x)-x) - P_binary(x).

    P_binary is the nearest four-content vertex (stable rank projection).
    At a fixed point its binary shadow belongs to both constraint sets, but
    convergence is not promised. Observe the shadow with exact correlation
    and the independent verifier, never the real iterate or a small residual.
    """
    if metal:
        from .torch_metal import project_signs
        binary = project_signs(values, model.weights)
    else:
        binary = model.project(values)
    return values + project_target_spectrum(model, 2 * binary - values) - binary
