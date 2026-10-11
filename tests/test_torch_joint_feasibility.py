import torch

from solver.target_profiles import canonical_target_content_profiles
from solver.torch_joint_feasibility import search_joint_feasibility, spectral_joint_projection
from solver.verifier import verify_pqcp


def test_radial_joint_projection_matches_direct_dft_sphere_and_real_inverse():
    d = 7
    values = torch.randn(3, 2, 2, d, dtype=torch.float64, generator=torch.Generator().manual_seed(4314))
    frequency = torch.arange(d//2+1, dtype=values.dtype)
    angle = 2*torch.pi*frequency[:, None]*torch.arange(d, dtype=values.dtype)[None]/d
    target = torch.full((3, 2, d//2+1), 7., dtype=values.dtype)
    out = spectral_joint_projection(values, angle.cos(), angle.sin(), target)
    f = torch.fft.rfft(values)
    expected = torch.fft.irfft(f*(target/f.abs().square().sum(-2)).sqrt()[..., None, :], n=d)
    assert torch.allclose(out, expected, atol=1e-12)
    assert torch.allclose(torch.fft.rfft(out).abs().square().sum(-2), target, atol=1e-11)


def test_joint_feasibility_audits_original_length_and_retains_all_zeros():
    targets = canonical_target_content_profiles(6)
    profiles = tuple(targets[i % len(targets)] for i in range(16))
    out = search_joint_feasibility(profiles, device=torch.device('cpu'), seed=1436,
                                  iterations=12, observation_interval=3, export_limit=0)
    assert out.iterations == 12 and out.native_certificates+out.exact_integer_fallbacks == 384
    assert out.candidates and all(row['verified'] and verify_pqcp(row['A'], row['B']).is_valid for row in out.candidates)


def test_observation_interval_cannot_skip_transient_discrete_candidates():
    profiles = canonical_target_content_profiles(6)*4
    first = search_joint_feasibility(profiles, device=torch.device('cpu'), seed=512, iterations=11,
                                    observation_interval=1, export_limit=16)
    second = search_joint_feasibility(profiles, device=torch.device('cpu'), seed=512, iterations=11,
                                     observation_interval=1000, export_limit=16)
    assert first == second
