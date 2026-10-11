"""The experimental packed FP32 GEMM must preserve the DFT and its gradient."""

import pytest
import torch

from experiments.benchmark_neural_matmul import PackedDFT, error_stats
from solver.target_profiles import TargetContentProfile, target_content_profiles
from solver.torch_search import BatchedPQCP


@pytest.mark.parametrize('length', [4, 44, 46, 58, 68, 86, 90, 94])
@pytest.mark.parametrize('alignment', [1, 32])
def test_packed_fp32_profile_and_gradient(length, alignment):
    # Kernel identity is defined even at lengths with no feasible PQCP target.
    # This synthetic target is only a gradient-test fixture, never a search seed.
    profile = TargetContentProfile(length, 1, 1, 0, 0, 0, 0)
    model = BatchedPQCP((profile,), torch.device('cpu'), continuous_kernel='dft')
    packed = PackedDFT(model, alignment=alignment)
    generator = torch.Generator().manual_seed(781)
    x = (torch.rand(8, 2, length, generator=generator)*2-1).requires_grad_()
    ref = model.continuous_correlation(x)
    actual = packed(x)
    grad_ref = torch.autograd.grad((ref-model.targets).square().mean(), x)[0]
    grad_actual = torch.autograd.grad((actual-model.targets).square().mean(), x)[0]
    torch.testing.assert_close(actual, ref, rtol=1e-4, atol=1e-3)
    torch.testing.assert_close(grad_actual, grad_ref, rtol=1e-4, atol=1e-3)
    assert error_stats(ref, actual)['finite']


def test_packed_rejects_wrong_sequence_shape():
    model = BatchedPQCP((target_content_profiles(44)[0],), torch.device('cpu'), continuous_kernel='dft')
    with pytest.raises(ValueError, match='batch,2,L'):
        PackedDFT(model)(torch.zeros(2, 44))
