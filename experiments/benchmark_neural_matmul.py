"""Probe DFT matrix layouts that MPS may map to GPU Neural Accelerators.

Timing cannot establish which hardware unit ran a kernel: use a Metal trace.
FP16 is experimental and is never used by production search or verification.
"""

import argparse
import gc
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.nn import functional as F
from solver.target_profiles import target_content_profiles
from solver.torch_search import BatchedPQCP


class PackedDFT(torch.nn.Module):
    """The same real DFT, packing cosine/sine into one padded GEMM.

    Only the forward DFT GEMM optionally uses FP16. Squaring, summing the two
    sequence powers and the inverse DFT remain FP32. Autograd remains enabled.
    """

    def __init__(self, model, dtype=torch.float32, alignment=32):
        super().__init__()
        if model.continuous_kernel != 'dft':
            raise ValueError('a DFT model is required')
        self.length = model.L
        self.frequencies = model.L // 2 + 1
        self.padded_length = (model.L + alignment-1) // alignment * alignment
        width = (2 * self.frequencies + alignment-1) // alignment * alignment
        packed = torch.cat((model.dft_real, model.dft_imag), dim=1)
        self.register_buffer('matrix', F.pad(packed, (0, width-packed.shape[1],
                                                     0, self.padded_length-model.L)).to(dtype))
        self.register_buffer('inverse', model.dft_inverse_half.clone())

    def forward(self, signs):
        if signs.ndim != 3 or signs.shape[1:] != (2, self.length):
            raise ValueError('expected batch,2,L signs')
        flat = F.pad(signs.reshape(-1, self.length), (0, self.padded_length-self.length))
        spectrum = (flat.to(self.matrix.dtype) @ self.matrix).float()
        spectrum = spectrum[:, :2*self.frequencies].reshape(-1, 2, 2*self.frequencies)
        real, imag = spectrum.split(self.frequencies, dim=-1)
        power = (real.square()+imag.square()).sum(-2)
        half = power @ self.inverse
        return torch.cat((half, half[:, 1:-1].flip(-1)), dim=-1)


class SeparateHalfDFT(torch.nn.Module):
    """Isolate dtype from packing/padding cost; never used in production."""

    def __init__(self, model):
        super().__init__()
        self.register_buffer('real', model.dft_real.half())
        self.register_buffer('imag', model.dft_imag.half())
        self.register_buffer('inverse', model.dft_inverse_half.clone())

    def forward(self, signs):
        low = signs.half()
        power = ((low @ self.real).float().square() +
                 (low @ self.imag).float().square()).sum(-2)
        half = power @ self.inverse
        return torch.cat((half, half[:, 1:-1].flip(-1)), dim=-1)


def error_stats(reference, actual):
    delta = (actual-reference).abs()
    return dict(max_abs=delta.max().item(),
                relative_l2=(delta.norm()/reference.norm().clamp_min(1e-12)).item(),
                finite=bool(torch.isfinite(actual).all().item()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--lengths', type=int, nargs='+', default=[44, 68])
    parser.add_argument('--batches', type=int, nargs='+', default=[4096, 8192, 16384])
    parser.add_argument('--steps', type=int, default=100)
    args = parser.parse_args()
    if args.steps <= 0 or any(batch <= 0 for batch in args.batches):
        parser.error('steps and batches must be positive')
    if not torch.backends.mps.is_available():
        raise RuntimeError('MPS is required; this benchmark does not fall back to CPU')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = dict(torch_version=torch.__version__, rows=[],
                  scope='DFT forward/backward only; not hardware utilization or search throughput',
                  neural_accelerator_use='unconfirmed without Metal hardware trace')
    for length in args.lengths:
        profiles = target_content_profiles(length)
        if not profiles:
            raise ValueError('L={} has no feasible target/content profile'.format(length))
        profile = profiles[0]
        model = BatchedPQCP((profile,), torch.device('mps'), continuous_kernel='dft')
        variants = [('baseline_fp32', model.continuous_correlation),
                    ('packed_fp32', PackedDFT(model)),
                    ('packed_fp16_experimental', PackedDFT(model, torch.float16)),
                    ('unpadded_fp32', PackedDFT(model, alignment=1)),
                    ('unpadded_fp16_experimental', PackedDFT(model, torch.float16, alignment=1)),
                    ('separate_fp16_experimental', SeparateHalfDFT(model))]
        for batch in args.batches:
            generator = torch.Generator().manual_seed(123)
            x = (torch.rand(batch, 2, length, generator=generator)*2-1).to('mps').requires_grad_()
            ref = model.continuous_correlation(x)
            grad_ref = torch.autograd.grad((ref-model.targets).square().mean(), x)[0]
            ref = ref.detach()
            for name, evaluate in variants:
                value = evaluate(x)
                gradient = torch.autograd.grad((value-model.targets).square().mean(), x)[0]
                errors = dict(profile=error_stats(ref, value), gradient=error_stats(grad_ref, gradient))
                if not all(item['finite'] for item in errors.values()):
                    raise RuntimeError('non-finite profile/gradient: '+name)
                if name in ('packed_fp32', 'unpadded_fp32'):
                    torch.testing.assert_close(value, ref, rtol=1e-4, atol=1e-3)
                    torch.testing.assert_close(gradient, grad_ref, rtol=1e-3, atol=1e-5)
                for _ in range(5):
                    torch.autograd.grad((evaluate(x)-model.targets).square().mean(), x)
                torch.mps.synchronize()
                started = perf_counter()
                for _ in range(args.steps):
                    torch.autograd.grad((evaluate(x)-model.targets).square().mean(), x)
                torch.mps.synchronize()
                elapsed = perf_counter()-started
                row = dict(L=length, batch=batch, variant=name, seconds=elapsed,
                           lane_steps_per_second=batch*args.steps/elapsed, errors=errors,
                           driver_bytes=torch.mps.driver_allocated_memory())
                report['rows'].append(row)
                print(json.dumps(row), flush=True)
                (output/'report.json').write_text(json.dumps(report, indent=2))
            del x, ref, grad_ref, value, gradient
            gc.collect()
            torch.mps.empty_cache()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
