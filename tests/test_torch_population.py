"""Exactness tests for fixed-content PyTorch population search."""

import pytest

torch = pytest.importorskip("torch")

from solver.correlation import full_correlation_profile
from solver.objective import pqcp_objective
from solver.structured_energy import structured_energy_breakdown
from solver.target_profiles import pair_content
from solver.torch_population import PopulationSearchConfig, TorchPopulationSearch


def test_population_sampling_preserves_every_profile_content():
    search = TorchPopulationSearch(PopulationSearchConfig(
        8, device="cpu", islands_per_profile=2, population_size=5, elite_count=2,
    ))
    bits = search.sample().to(torch.int32).tolist()
    for profile, population in zip(search.profiles, bits):
        expected = (
            profile.a_even_ones, profile.a_odd_ones,
            profile.b_even_ones, profile.b_odd_ones,
        )
        assert all(pair_content(a, b) == expected for a, b in population)


def test_population_profile_score_and_energy_match_python_exactly():
    search = TorchPopulationSearch(PopulationSearchConfig(
        8, device="cpu", islands_per_profile=1, population_size=4, elite_count=2,
    ))
    result = search.evaluate(search.sample())
    for index, target in enumerate(search.profiles):
        for member in range(search.config.population_size):
            a, b = result.bits[index, member].to(torch.int32).tolist()
            exact = full_correlation_profile(a, b)
            assert result.profiles[index, member].tolist() == exact
            assert result.scores[index, member].item() == pqcp_objective(exact)
            factors = (2, 4) if search.config.L % 4 == 0 else (2,)
            expected = structured_energy_breakdown(exact, target.k, target.eta, factors)
            weights = {2: 2, **({4: 4} if search.config.L % 4 == 0 else {})}
            assert result.energy[index, member].item() == expected.weighted_total(weights)


def test_population_step_updates_counters_and_keeps_finite_logits():
    search = TorchPopulationSearch(PopulationSearchConfig(
        8, device="cpu", islands_per_profile=1, population_size=8, elite_count=2,
    ))
    search.step()
    assert search.generation == 1
    assert search.candidate_evaluations == len(search.profiles) * 8
    assert search.best is not None
    assert bool(torch.isfinite(search.logits).all().item())


@pytest.mark.parametrize("kwargs", [
    {"population_size": 2, "elite_count": 3},
    {"learning_rate": 0},
    {"probability_floor": 0.5},
])
def test_population_config_rejects_invalid_controls(kwargs):
    with pytest.raises(ValueError):
        PopulationSearchConfig(8, device="cpu", **kwargs)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("length", [4, 8, 28, 44, 46, 68])
def test_refinement_preserves_content_and_exact_pacf_through_multiple_swaps(length, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("real CUDA unavailable")
    search = TorchPopulationSearch(PopulationSearchConfig(
        length, device=device, population_size=8, elite_count=2, islands_per_profile=1,
    ))
    for generation in range(34):
        batch = search.step()
        if generation not in (0, 1, 8, 32, 33):
            continue
        # Include a freshly sampled immigrant and a mutated elite in every
        # target family, including the midpoint shared-edge case in swaps.
        bits = batch.bits[:, [0, 7]].to(torch.int32).cpu().tolist()
        profiles = batch.profiles[:, [0, 7]].cpu().tolist()
        scores = batch.scores[:, [0, 7]].cpu().tolist()
        energies = batch.energy[:, [0, 7]].cpu().tolist()
        for island, target in enumerate(search.profiles):
            for member, (a, b) in enumerate(bits[island]):
                exact = full_correlation_profile(a, b)
                assert profiles[island][member] == exact
                assert scores[island][member] == pqcp_objective(exact)
                assert pair_content(a, b) == (target.a_even_ones, target.a_odd_ones,
                                              target.b_even_ones, target.b_odd_ones)
                weights = {2: 2, **({4: 4} if length % 4 == 0 else {})}
                expected = structured_energy_breakdown(exact, target.k, target.eta, tuple(weights))
                assert energies[island][member] == expected.weighted_total(weights)


def test_same_seed_produces_same_unseeded_learning_trajectory():
    config = PopulationSearchConfig(28, device="cpu", population_size=16,
                                    elite_count=4, islands_per_profile=1)
    left, right = TorchPopulationSearch(config), TorchPopulationSearch(config)
    for _ in range(12):
        a, b = left.step(), right.step()
        assert torch.equal(a.bits, b.bits)
        assert torch.equal(a.profiles, b.profiles)
    assert torch.equal(left.logits, right.logits)


def test_solutions_are_not_mutated_into_offspring():
    """Even an all-solved tiny island may not breed solution neighborhoods."""
    search = TorchPopulationSearch(PopulationSearchConfig(
        4, device="cpu", population_size=16, elite_count=4, islands_per_profile=1,
    ))
    search.step()
    # Force all archive entries to be considered solved to audit the guard.
    from solver.torch_population import PopulationBatch
    old = search.elites
    search.elites = PopulationBatch(old.bits, old.profiles, old.energy, torch.zeros_like(old.scores))
    offspring = search._offspring(8)
    assert torch.equal(offspring.bits, old.bits[:, :1].expand(-1, 8, -1, -1))
