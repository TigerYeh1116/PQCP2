"""Tests for the isolated reference-project search preset."""

import pytest

from solver.checkpoint import SearchParameters, save_checkpoint
from solver.correlation import full_correlation_profile
from solver.reference_search import (
    encoded_target_profiles,
    legacy_search_parameters,
    reference_search_parameters,
)
from solver.search_runner import SearchRunner, _target_profile_index
from solver.target_profiles import pair_content


def test_reference_preset_encodes_ported_controls_without_changing_target():
    parameters = reference_search_parameters(44)
    assert parameters.acceptance_mode == "fixed_target_multiscale"
    assert parameters.fkm_seed_policy == "compressed_a_random_b_top_q"
    assert parameters.preserve_alternating_content
    assert not parameters.randomize_target_profiles
    assert parameters.target_profile_offset == 0
    assert parameters.proposal_samples == 2
    assert parameters.temperature_schedule == "linear_restart"
    assert parameters.metropolis_scale == 4.0
    assert parameters.kick_stagnation_iterations == 1_000
    assert parameters.kick_swaps == 3
    assert parameters.max_iterations_per_restart == 64_000
    assert parameters.target_content_profiles == encoded_target_profiles(44)


def test_reference_runner_preserves_profile_and_content_through_kicks():
    base = reference_search_parameters(8)
    parameters = SearchParameters(**{
        **base.to_dict(),
        "kick_stagnation_iterations": 1,
        "max_iterations_per_restart": 100,
    })
    runner = SearchRunner.new(8, 123, parameters)
    kicked = False
    for _ in range(30):
        outcome = runner.step()
        kicked = kicked or outcome.kick_performed
        target = runner._current_target_content_profile()
        expected_content = (
            target.a_even_ones, target.a_odd_ones,
            target.b_even_ones, target.b_odd_ones,
        )
        assert pair_content(runner.state.current_a, runner.state.current_b) == expected_content
        assert runner._correlation.profile == tuple(full_correlation_profile(
            runner.state.current_a, runner.state.current_b
        ))
    assert kicked


def test_linear_schedule_and_kick_resume_are_deterministic(tmp_path):
    base = reference_search_parameters(44)
    parameters = SearchParameters(**{
        **base.to_dict(),
        "kick_stagnation_iterations": 2,
        "max_iterations_per_restart": 50,
    })
    continuous = SearchRunner.new(44, 991, parameters)
    split = SearchRunner.new(44, 991, parameters)
    continuous.step()
    assert continuous.state.temperature == pytest.approx(
        6.15 - (6.15 - 0.15) / 50
    )
    for _ in range(39):
        continuous.step()
    for _ in range(17):
        split.step()
    path = tmp_path / "reference.json"
    save_checkpoint(path, split.state)
    resumed = SearchRunner.resume(path)
    for _ in range(23):
        resumed.step()
    assert resumed.state.to_dict() == continuous.state.to_dict()


def test_legacy_preset_remains_an_explicit_rollback_path():
    parameters = legacy_search_parameters(44)
    assert parameters.acceptance_mode == "fixed_target_full"
    assert parameters.fkm_seed_policy == "legacy"
    assert parameters.temperature_schedule == "geometric"
    assert parameters.kick_stagnation_iterations is None


def test_seed_specific_profile_order_is_a_complete_deterministic_permutation():
    count = len(encoded_target_profiles(44))
    first = [_target_profile_index(123, restart, count, True) for restart in range(count)]
    repeated = [_target_profile_index(123, restart, count, True) for restart in range(count)]
    second_seed = [_target_profile_index(456, restart, count, True) for restart in range(count)]
    assert first == repeated
    assert sorted(first) == list(range(count))
    assert first != second_seed
    assert [_target_profile_index(123, restart, count, False) for restart in range(count)] == list(range(count))


def test_worker_offset_starts_from_distinct_profiles_without_randomness():
    first = SearchRunner.new(44, 123, reference_search_parameters(44, 0))
    second = SearchRunner.new(44, 123, reference_search_parameters(44, 1))
    assert first._current_target_content_profile() != second._current_target_content_profile()
    assert first.state.algorithm_parameters.target_profile_offset == 0
    assert second.state.algorithm_parameters.target_profile_offset == 1


@pytest.mark.parametrize(
    "kwargs",
    (
        {"temperature_schedule": "bad"},
        {"metropolis_scale": 0},
        {"kick_stagnation_iterations": 0},
        {"kick_swaps": 0},
        {"temperature_schedule": "linear_restart"},
    ),
)
def test_new_search_controls_are_strictly_validated(kwargs):
    with pytest.raises(ValueError):
        SearchParameters(**kwargs)
