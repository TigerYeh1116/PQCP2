"""Brute-force proof checks for the quarter-frequency initialization filter."""

from dataclasses import replace
from itertools import product

import pytest

from solver.correlation import periodic_autocorrelation
from solver.frequency_constraints import _quarter_squares, quarter_frequency_feasible
from solver.target_profiles import TargetContentProfile, target_content_profiles


@pytest.mark.parametrize("length", [4, 8, 12, 16])
def test_one_parity_group_attainable_squares_match_all_binary_assignments(length):
    actual = {weight: set() for weight in range(length // 2 + 1)}
    for bits in product((0, 1), repeat=length // 2):
        real = sum((-1 if j % 2 else 1) * (1 - 2 * bit)
                   for j, bit in enumerate(bits))
        assert real % 2 == 0
        actual[sum(bits)].add((real // 2) ** 2)
    for weight, squares in actual.items():
        assert _quarter_squares(length, weight) == squares


@pytest.mark.parametrize("length", [4, 8])
def test_no_actual_small_pqcp_profile_is_pruned(length):
    sequences = list(product((0, 1), repeat=length))
    correlations = {a: tuple(periodic_autocorrelation(a, u) for u in range(length))
                    for a in sequences}
    valid = 0
    for a in sequences:
        for b in sequences:
            profile = tuple(x + y for x, y in zip(correlations[a], correlations[b]))
            nonzero = [u for u in range(1, length) if profile[u]]
            if len(nonzero) != 2 or any(abs(profile[u]) != 4 for u in nonzero):
                continue
            k = min(nonzero)
            target = TargetContentProfile(length, k, profile[k] // 4,
                                          sum(a[::2]), sum(a[1::2]),
                                          sum(b[::2]), sum(b[1::2]))
            assert quarter_frequency_feasible(target)
            valid += 1
    assert valid > 0  # The safety check must not pass vacuously.


def test_reported_impossible_l44_profiles_have_an_integer_obstruction():
    profiles = target_content_profiles(44, decimation_reduced=True)
    removed = [p for p in profiles if not quarter_frequency_feasible(p)]
    assert len(profiles) == 128 and len(removed) == 32
    for p in removed:
        assert p.k == 2 and p.eta == -1
        assert all(w % 2 for w in (p.a_even_ones, p.a_odd_ones,
                                   p.b_even_ones, p.b_odd_ones))
        # Four odd squares total 4 mod 8; required PSD/4 is 24 = 0 mod 8.
        assert (44 // 2 - 2 * p.eta) % 8 == 0


def test_lengths_without_quarter_frequency_are_not_pruned():
    for length in (6, 46, 86, 94):
        assert all(quarter_frequency_feasible(p) for p in target_content_profiles(length))


def test_invalid_counts_raise_instead_of_becoming_a_pruning_verdict():
    p = target_content_profiles(8)[0]
    with pytest.raises(ValueError, match="invalid"):
        quarter_frequency_feasible(replace(p, a_even_ones=5))
    p = target_content_profiles(4)[0]
    quarter_frequency_feasible(p)  # Prime the cache with the integer key.
    with pytest.raises(ValueError, match="invalid"):
        quarter_frequency_feasible(replace(p, k=True))


def test_filtered_torch_initialization_and_resume_use_only_possible_profiles(tmp_path):
    torch = pytest.importorskip("torch")
    from solver.torch_search import TorchSearch, TorchSearchConfig
    config = TorchSearchConfig(44, batch_size=8, device="cpu", frequency_pruning=True)
    search = TorchSearch(config, tmp_path)
    assert len(search.available) == 96
    assert all(quarter_frequency_feasible(p) for p in search.model.profiles)
    search.step()
    path = tmp_path / "frequency.json"
    search.checkpoint(path)
    restored = TorchSearch.resume(path, tmp_path)
    assert restored.config.frequency_pruning
    search.step(); restored.step()
    assert torch.equal(search.theta, restored.theta)
