"""Exact construction, correlation, lift, and candidate-start tests for GCPs."""

import pytest

from solver.checkpoint import SearchParameters
from solver.correlation import full_correlation_profile
from solver.golay import (
    complete_one_flip_each,
    is_golay_complementary_pair,
    is_periodic_golay_pair,
    project_length_golay_seed,
    rudin_shapiro_golay_pair,
)
from solver.objective import pqcp_objective
from solver.search_runner import SearchRunner
from solver.verifier import verify_pqcp


@pytest.mark.parametrize("length", (1, 2, 4, 8, 16, 32))
def test_rudin_shapiro_pair_is_golay_and_has_score_two(length):
    a, b = rudin_shapiro_golay_pair(length)
    assert len(a) == len(b) == length
    assert is_golay_complementary_pair(a, b)
    profile = full_correlation_profile(a, b)
    assert profile == [2 * length] + [0] * (length - 1)
    assert pqcp_objective(profile) == 2


@pytest.mark.parametrize("length", (0, 3, 6, True))
def test_rudin_shapiro_rejects_unsupported_lengths(length):
    with pytest.raises(ValueError):
        rudin_shapiro_golay_pair(length)


@pytest.mark.parametrize("length", (4, 8))
def test_one_flip_each_completion_is_independently_verified(length):
    solution = complete_one_flip_each(length, seed=3)
    assert solution is not None
    assert verify_pqcp(*solution).is_valid


def test_finite_golay_neighborhood_miss_is_not_reported_as_solution():
    assert complete_one_flip_each(16, seed=0) is None


def test_candidate_runner_starts_exactly_from_golay_pair_and_resumes_deterministically(tmp_path):
    a, b = rudin_shapiro_golay_pair(8)
    parameters = SearchParameters(stagnation_iterations=None, acceptance_mode="objective_plus_target_pair")
    runner = SearchRunner.from_candidate(a, b, 17, parameters)
    assert runner.state.current_a == a and runner.state.current_b == b
    assert runner.state.current_score == runner.state.best_score == 2

    twin = SearchRunner.from_candidate(a, b, 17, parameters)
    for _ in range(20):
        runner.step()
        twin.step()
    assert runner.state.to_dict() == twin.state.to_dict()


@pytest.mark.parametrize("length", (58, 68, 90))
def test_published_project_seed_is_exact_periodic_golay(length):
    seed = project_length_golay_seed(length, seed=123)
    assert seed.kind == "periodic"
    assert len(seed.a) == len(seed.b) == length
    assert is_periodic_golay_pair(seed.a, seed.b)
    profile = full_correlation_profile(seed.a, seed.b)
    assert profile == [2 * length] + [0] * (length - 1)
    assert pqcp_objective(profile) == 2


@pytest.mark.parametrize(
    "length,expected_score,expected_value,expected_count",
    (
        (44, 48, -8, 10),
        (46, 20, -4, 22),
        (86, 40, -4, 42),
        (94, 44, -4, 46),
    ),
)
def test_turyn_gcp_adaptation_has_exact_expected_profile(
    length, expected_score, expected_value, expected_count
):
    seed = project_length_golay_seed(length, seed=123)
    assert seed.kind == "turyn-near-periodic"
    assert len(seed.a) == len(seed.b) == length
    assert not is_periodic_golay_pair(seed.a, seed.b)
    profile = full_correlation_profile(seed.a, seed.b)
    nonzero_values = [value for value in profile[1:] if value]
    assert nonzero_values == [expected_value] * expected_count
    assert pqcp_objective(profile) == expected_score


@pytest.mark.parametrize("length", (44, 46, 86, 94))
def test_exact_periodic_golay_is_arithmetically_impossible_at_adapted_lengths(length):
    assert not any(x * x + y * y == 2 * length for x in range(length + 1) for y in range(length + 1))


@pytest.mark.parametrize("length", (44, 46, 58, 68, 86, 90, 94))
def test_project_seed_equivalence_is_reproducible_and_preserves_profile(length):
    first = project_length_golay_seed(length, seed=7)
    twin = project_length_golay_seed(length, seed=7)
    other = project_length_golay_seed(length, seed=8)
    assert first == twin
    assert (first.a, first.b) != (other.a, other.b)
    assert full_correlation_profile(first.a, first.b) == full_correlation_profile(other.a, other.b)


def test_unregistered_non_power_of_two_golay_length_is_rejected():
    with pytest.raises(ValueError, match="no verified low-cost Golay construction"):
        project_length_golay_seed(6)
