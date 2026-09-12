"""Exact quantization/PACF observation kernels on PyTorch's MPS stream.

No optimizer, approximate correlation, CPU fallback, or solution decision is
implemented here. Kernels use torch.mps.compile_shader (PyTorch 2.8 API).
Stable ranking implements the same four-content projection as torch.argsort.
PACF sums +/-1 products using int32 and writes exact float32 integers for the
supported L<=94. The independent Python verifier remains authoritative.
"""

from functools import lru_cache

import torch


_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

kernel void content_project(device float* out,
                            device const float* values,
                            device const long* weights,
                            constant long& length,
                            uint idx [[thread_position_in_grid]]) {
    uint L = uint(length);
    uint sequence = idx / L;
    uint p = idx % L;
    uint parity = p & 1;
    float x = values[idx];
    uint rank = 0;
    for (uint j = parity; j < L; j += 2) {
        float y = values[sequence * L + j];
        rank += uint(y < x || (y == x && j < p));
    }
    out[idx] = rank < uint(weights[sequence * 2 + parity]) ? -1.0f : 1.0f;
}

kernel void pair_profile(device float* out,
                         device const float* signs,
                         constant long& length,
                         uint idx [[thread_position_in_grid]]) {
    uint L = uint(length);
    uint lane = idx / L;
    uint u = idx % L;
    if (u > L/2) return;
    uint base = lane * 2 * L;
    int total = 0;
    for (uint i = 0; i < L; ++i) {
        uint j = (i + u) % L;
        total += int(signs[base+i]) * int(signs[base+j]);
        total += int(signs[base+L+i]) * int(signs[base+L+j]);
    }
    out[lane*L+u] = float(total);
    if (u != 0 && u != L/2) out[lane*L+L-u] = float(total);
}
"""


@lru_cache(maxsize=1)
def _library():
    """Compile once, and fail explicitly on an unsupported MPS installation."""
    if not torch.backends.mps.is_available() or not hasattr(torch.mps, "compile_shader"):
        raise RuntimeError("Metal observation requires MPS and torch.mps.compile_shader")
    return torch.mps.compile_shader(_SOURCE)


@torch.no_grad()
def project_signs(logits: torch.Tensor, weights: torch.Tensor):
    """Exact stable four-content projection, without computing correlation."""
    if logits.device.type != "mps" or weights.device != logits.device:
        raise ValueError("Metal observation requires tensors on the same MPS device")
    if logits.dtype != torch.float32 or weights.dtype != torch.int64:
        raise ValueError("Metal observation requires float32 logits and int64 weights")
    if logits.ndim != 3 or logits.shape[1] != 2 or weights.shape != (logits.shape[0], 2, 2):
        raise ValueError("expected batch,2,L logits and batch,2,2 weights")
    length = logits.shape[-1]
    if not 4 <= length <= 94 or length % 2:
        raise ValueError("Metal observation supports even 4<=L<=94")
    library = _library()
    signs = torch.empty_like(logits, memory_format=torch.contiguous_format)
    library.content_project(signs, logits.contiguous(), weights.contiguous(), length)
    return signs


@torch.no_grad()
def project_and_correlate(logits: torch.Tensor, weights: torch.Tensor):
    """Return exact (binary signs, full periodic pair profile) for every lane."""
    signs = project_signs(logits, weights)
    length = logits.shape[-1]
    library = _library()
    profile = torch.empty((logits.shape[0], length), device=logits.device, dtype=torch.float32)
    library.pair_profile(profile, signs, length)
    return signs, profile
