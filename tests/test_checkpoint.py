"""JSON checkpoint safety and deterministic state-restoration tests."""

import json

import pytest

from solver.checkpoint import CheckpointError, SearchParameters, load_checkpoint, save_checkpoint
from solver.search_runner import SearchRunner
from solver.target_profiles import canonical_target_content_profiles


def test_save_load_round_trip_preserves_json_state(tmp_path):
    runner = SearchRunner.new(6, 123, SearchParameters(stagnation_iterations=None))
    for _ in range(7):
        runner.step()
    path = tmp_path / "L6.json"
    save_checkpoint(path, runner.state)
    restored = load_checkpoint(path)
    assert restored.to_dict() == runner.state.to_dict()
    assert json.loads(path.read_text(encoding="utf-8"))["schema_version"] == 1


def test_continuous_and_save_resume_have_identical_trajectory(tmp_path):
    parameters = SearchParameters(stagnation_iterations=None)
    continuous = SearchRunner.new(6, 77, parameters)
    split = SearchRunner.new(6, 77, parameters)
    for _ in range(30):
        continuous.step()
    for _ in range(11):
        split.step()
    path = tmp_path / "state.json"
    save_checkpoint(path, split.state)
    resumed = SearchRunner.resume(path)
    for _ in range(19):
        resumed.step()
    assert resumed.state.to_dict() == continuous.state.to_dict()


def test_guided_energy_continuous_and_resume_trajectory_match(tmp_path):
    parameters = SearchParameters(
        stagnation_iterations=None,
        acceptance_mode="objective_plus_target_pair",
        initial_temperature=8.0,
    )
    continuous = SearchRunner.new(8, 91, parameters)
    split = SearchRunner.new(8, 91, parameters)
    for _ in range(40):
        continuous.step()
    for _ in range(17):
        split.step()
    path = tmp_path / "guided.json"
    save_checkpoint(path, split.state)
    resumed = SearchRunner.resume(path)
    for _ in range(23):
        resumed.step()
    assert resumed.state.to_dict() == continuous.state.to_dict()


def test_multisample_continuous_and_resume_trajectory_match(tmp_path):
    parameters = SearchParameters(
        stagnation_iterations=None,
        acceptance_mode="objective_plus_target_pair",
        proposal_samples=8,
    )
    continuous = SearchRunner.new(8, 319, parameters)
    split = SearchRunner.new(8, 319, parameters)
    for _ in range(40):
        continuous.step()
    for _ in range(17):
        split.step()
    path = tmp_path / "multisample.json"
    save_checkpoint(path, split.state)
    resumed = SearchRunner.resume(path)
    for _ in range(23):
        resumed.step()
    assert resumed.state.to_dict() == continuous.state.to_dict()


def test_compressed_seed_and_tiebreak_resume_trajectory_match(tmp_path):
    profile = canonical_target_content_profiles(8)[0]
    encoded = ((profile.k, profile.eta, profile.a_even_ones,
                profile.a_odd_ones, profile.b_even_ones, profile.b_odd_ones),)
    parameters = SearchParameters(
        stagnation_iterations=None,
        target_content_profiles=encoded,
        preserve_alternating_content=True,
        acceptance_mode="fixed_target_full_compressed_tiebreak",
        proposal_samples=5,
        fkm_seed_policy="phase_top_q",
        fkm_candidate_count=8,
        fkm_elite_count=3,
    )
    continuous = SearchRunner.new(8, 8128, parameters)
    split = SearchRunner.new(8, 8128, parameters)
    for _ in range(50):
        continuous.step()
    for _ in range(21):
        split.step()
    path = tmp_path / "compressed.json"
    save_checkpoint(path, split.state)
    resumed = SearchRunner.resume(path)
    for _ in range(29):
        resumed.step()
    assert resumed.state.to_dict() == continuous.state.to_dict()


def test_corrupted_checkpoint_is_rejected(tmp_path):
    path = tmp_path / "corrupt.json"
    path.write_text("not json", encoding="utf-8")
    with pytest.raises(CheckpointError):
        load_checkpoint(path)


def test_atomic_rewrite_keeps_latest_complete_checkpoint(tmp_path):
    runner = SearchRunner.new(4, 6, SearchParameters(stagnation_iterations=None))
    path = tmp_path / "state.json"
    save_checkpoint(path, runner.state)
    runner.step()
    save_checkpoint(path, runner.state)
    assert load_checkpoint(path).iteration == 1
    assert not list(tmp_path.glob("*.tmp"))


def test_legacy_parameters_without_proposal_samples_keep_baseline_behavior():
    payload = SearchParameters().to_dict()
    del payload["proposal_samples"]
    del payload["objective_energy_weight"]
    del payload["fkm_seed_policy"]
    del payload["fkm_candidate_count"]
    del payload["fkm_elite_count"]
    assert SearchParameters.from_dict(payload).proposal_samples == 1
    assert SearchParameters.from_dict(payload).objective_energy_weight == 2
    assert SearchParameters.from_dict(payload).fkm_seed_policy == "legacy"


def test_invalid_fkm_seed_selection_parameters_are_rejected():
    with pytest.raises(ValueError):
        SearchParameters(fkm_seed_policy="unknown")
    with pytest.raises(ValueError):
        SearchParameters(fkm_candidate_count=0)
    with pytest.raises(ValueError):
        SearchParameters(fkm_elite_count=0)
    with pytest.raises(ValueError):
        SearchParameters(
            fkm_seed_policy="phase_top_q",
            fkm_candidate_count=3,
            fkm_elite_count=4,
        )
