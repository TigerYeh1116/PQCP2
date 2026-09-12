"""Fast integration tests for CLI parsing and the orchestration layer."""

import json
from types import SimpleNamespace

import pytest

import main
import solver.pipeline as pipeline_module
from solver.checkpoint import SearchParameters, save_checkpoint
from solver.compression import CorrelationState
from solver.enhanced_search import EnhancedParameters, EnhancedSearch
from solver.objective import pqcp_objective
from solver.pipeline import (
    DEFAULT_REPAIR_BEAM_WIDTH,
    DEFAULT_REPAIR_MAX_DEPTH,
    DEFAULT_Z3_COMPLETION_RADIUS,
    DEFAULT_Z3_TRIGGER_SCORE,
    DEFAULT_Z3_TIMEOUT_SECONDS,
    GCP_INITIAL_TEMPERATURE,
    _restart_after_enhanced_solution,
    _run_optional_z3,
    _select_z3_result,
)
from solver.hybrid import SAElite
from solver.pipeline import PipelineConfig, run_pipeline, validate_length
from solver.search_runner import (
    DEFAULT_GUIDED_OBJECTIVE_ENERGY_WEIGHT,
    DEFAULT_GUIDED_PROPOSAL_SAMPLES,
    SearchRunner,
)


def test_time_budget_defaults_and_rejects_multiple_units():
    assert main.time_budget(main.parse_args(["--L", "44"])) == float("inf")
    assert main.time_budget(main.parse_args(["--L", "44", "--minutes", "2"])) == 120.0
    with pytest.raises(ValueError):
        main.time_budget(main.parse_args(["--L", "44", "--seconds", "1", "--hours", "1"]))


def test_cuda_batch_default_scales_for_wider_lengths_and_allows_override():
    assert main._effective_torch_batch_size(44, None) == 2048
    assert main._effective_torch_batch_size(68, None) == 1024
    assert main._effective_torch_batch_size(94, 4096) == 4096
    with pytest.raises(ValueError):
        main._effective_torch_batch_size(44, 0)


def test_z3_timeout_default_is_sixty_seconds_and_remains_overridable():
    assert DEFAULT_Z3_TIMEOUT_SECONDS == 60.0
    assert main.parse_args(["--L", "44"]).z3_timeout == 60.0
    assert main.parse_args(["--L", "44", "--z3-timeout", "90"]).z3_timeout == 90.0
    assert PipelineConfig().z3_timeout == 60.0


def test_low_score_completion_trigger_default_is_16():
    assert DEFAULT_Z3_TRIGGER_SCORE == 16
    assert main.parse_args(["--L", "44"]).z3_trigger_score == 16
    assert PipelineConfig().z3_trigger_score == 16
    assert not main.parse_args(["--L", "44"]).z3_broad
    assert main.parse_args(["--L", "44", "--z3-broad"]).z3_broad
    assert PipelineConfig().repair_beam_width == DEFAULT_REPAIR_BEAM_WIDTH == 30
    assert PipelineConfig().repair_max_depth == DEFAULT_REPAIR_MAX_DEPTH == 4
    assert PipelineConfig().z3_completion_radius == DEFAULT_Z3_COMPLETION_RADIUS == 4


def test_cli_requires_length_except_for_resume():
    assert main.main(["--seconds", "1"]) == 2
    assert validate_length(44)
    assert not validate_length(4)
    with pytest.raises(ValueError):
        validate_length(0)


def test_parallel_cli_rejects_synchronous_z3_or_repair():
    assert main.main([
        "--L", "44", "--seconds", "0.1", "--workers", "2", "--z3",
        "--python-backend",
    ]) == 2
    assert main.main([
        "--L", "44", "--seconds", "0.1", "--workers", "2", "--repair",
        "--python-backend",
    ]) == 2


def test_no_argument_interactive_mode_accepts_length(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    answers = iter(("4", "77", "", ""))
    prompts = []
    def answer(prompt):
        prompts.append(prompt)
        return next(answers)
    monkeypatch.setattr("builtins.input", answer)
    monkeypatch.setattr("sys.argv", ["main.py"])
    monkeypatch.setattr(main, "time_budget", lambda _args: 1.0)
    captured = []
    monkeypatch.setattr(
        main, "_run_torch_backend",
        lambda args, budget: captured.append((args, budget)) or 0,
    )
    assert main.main() == 0
    assert captured[0][0].seed == 77
    assert captured[0][0].L == 4
    assert captured[0][1] == 1.0
    assert captured[0][0].torch_mathematical_loss == "none"
    assert all("搜尋模式" not in prompt for prompt in prompts)


def test_interactive_mode_can_enable_z3_and_new_mathematical_loss(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    answers = iter(("4", "77", "y", "y"))
    prompts = []

    def answer(prompt):
        prompts.append(prompt)
        return next(answers)

    monkeypatch.setattr("builtins.input", answer)
    monkeypatch.setattr("sys.argv", ["main.py"])
    monkeypatch.setattr(main, "time_budget", lambda _args: 1.0)
    captured = []
    monkeypatch.setattr(
        main, "_run_torch_backend",
        lambda args, budget: captured.append((args, budget)) or 0,
    )
    assert main.main() == 0
    args = captured[0][0]
    assert args.z3 and args.repair
    assert args.torch_mathematical_loss == "lattice"
    assert any("數學 loss" in prompt for prompt in prompts)


def test_cli_default_is_cuda_seed_formation_with_exact_swap_completion():
    args = main.parse_args(["--L", "44"])
    assert args.reference_port
    assert args.backend == "cuda"
    assert args.torch_kernel == "dft"
    assert main.parse_args(["--torch-kernel", "dft"]).torch_kernel == "dft"
    assert main.parse_args(["--backend", "c"]).backend == "c"
    assert main.parse_args(["--backend", "cuda"]).backend == "cuda"
    assert not args.gcp and not args.enhanced
    assert not args.python_backend and args.workers == 8
    assert not args.legacy_sampler
    assert main.parse_args(["--L", "44", "--legacy-sampler"]).legacy_sampler


def test_explicit_cuda_and_old_torch_checkpoint_route_to_cuda(tmp_path, monkeypatch, capsys):
    captured = []
    monkeypatch.setattr(main, "_run_torch_backend", lambda args, budget: captured.append((args, budget)) or 0)
    assert main.main(["--L", "44", "--backend", "cuda", "--seconds", "1"]) == 0
    path = tmp_path / "torch.json"
    path.write_text(json.dumps({"format": "pqcp-torch-search"}))
    assert main.main(["--resume", str(path), "--seconds", "1"]) == 0
    assert len(captured) == 2
    assert "載入至 CUDA" in capsys.readouterr().out
    assert main.main(["--L", "44", "--torch-kernel", "dft"]) == 0
    assert len(captured) == 3


def test_baseline_pipeline_smoke_persists_rich_best_snapshot(tmp_path):
    result = run_pipeline(PipelineConfig(L=4, seconds=0, root=tmp_path))
    assert result.best_path.exists() and result.checkpoint_path.exists()
    payload = json.loads(result.best_path.read_text(encoding="utf-8"))
    assert payload["score"] == payload["objective_components"]["total"]
    assert len(payload["A"]) == len(payload["B"]) == 4
    assert result.log_path.exists()


def test_resume_uses_checkpoint_state_without_reinitialization(tmp_path):
    runner = SearchRunner.new(4, 77, SearchParameters(stagnation_iterations=None))
    runner.step()
    checkpoint = tmp_path / "resume.json"
    save_checkpoint(checkpoint, runner.state)
    result = run_pipeline(PipelineConfig(resume=checkpoint, seconds=0, root=tmp_path))
    assert result.state.iteration == runner.state.iteration
    assert result.L == 4


def test_enhanced_flag_routes_to_existing_enhanced_checkpoint_format(tmp_path):
    result = run_pipeline(PipelineConfig(L=4, seconds=0, enhanced=True, root=tmp_path))
    assert result.method == "enhanced"
    assert result.checkpoint_path.name == "L4_enhanced.json"


def test_z3_flag_routes_near_global_best_to_existing_portfolio(tmp_path, monkeypatch):
    calls = []

    def fake_portfolio(config, length, elites, root):
        calls.append((length, tuple(elites), root))
        return {"status": "UNAVAILABLE", "reason": "test"}

    monkeypatch.setattr(pipeline_module, "_run_optional_z3", fake_portfolio)
    result = run_pipeline(PipelineConfig(L=8, seconds=0, z3=True, z3_trigger_score=9999, root=tmp_path))
    assert calls and calls[0][0] == 8
    assert result.z3_result["status"] == "UNAVAILABLE"
    assert result.z3_trigger_count == 1


def test_initial_bounded_z3_miss_immediately_starts_next_restart(tmp_path, monkeypatch):
    monkeypatch.setattr(
        pipeline_module, "_run_optional_z3",
        lambda *_args, **_kwargs: {
            "status": "Z3_UNKNOWN", "solved": False,
            "z3_executed": True, "reason": "timeout",
        },
    )
    result = run_pipeline(PipelineConfig(
        L=8, seconds=0, z3=True, z3_guided=False,
        z3_trigger_score=9999, root=tmp_path,
    ))
    assert result.z3_trigger_count == 1
    assert result.state.restart_index == 1


def test_z3_sat_does_not_force_completion_miss_restart(tmp_path, monkeypatch):
    monkeypatch.setattr(
        pipeline_module, "_run_optional_z3",
        lambda *_args, **_kwargs: {
            "status": "Z3_SAT", "solved": True,
            "z3_executed": True, "reason": None,
        },
    )
    result = run_pipeline(PipelineConfig(
        L=8, seconds=0, z3=True, z3_guided=False,
        z3_trigger_score=9999, root=tmp_path,
    ))
    assert result.state.restart_index == 0


def test_fixed_z3_trigger_skips_initial_state_above_configured_threshold(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        pipeline_module, "_run_optional_z3",
        lambda *args: calls.append(args) or {"status": "UNAVAILABLE"},
    )
    run_pipeline(PipelineConfig(L=5, seconds=0, z3=True, z3_trigger_score=8, root=tmp_path))
    assert not calls


def test_z3_summary_does_not_hide_earlier_solved_result():
    class Result:
        def __init__(self, solved):
            self.solved = solved

    solved = Result(True)
    assert _select_z3_result((Result(False), solved, Result(False))) is solved
    assert _select_z3_result(()) is None


def test_guidance_solution_is_saved_without_redundant_z3(tmp_path, monkeypatch):
    a, b = (0, 0, 0, 0), (0, 0, 1, 1)
    monkeypatch.setattr(
        pipeline_module, "guided_completion",
        lambda *_args, **_kwargs: SimpleNamespace(
            solved=True, a=a, b=b, first_moves_examined=3, second_moves_examined=4,
        ),
    )
    monkeypatch.setattr(
        pipeline_module, "beam_repair",
        lambda *_args, **_kwargs: pytest.fail("beam must not run after guidance solved"),
    )
    result = _run_optional_z3(
        PipelineConfig(L=4, z3=True), 4, (SAElite(a, b, 1, 1, 0),), tmp_path
    )
    assert result["status"] == "GUIDANCE_SAT"
    assert result["solved"] and not result["z3_executed"]
    assert (tmp_path / "4.txt").exists()


def test_beam_solution_is_saved_without_z3_after_guidance_miss(tmp_path, monkeypatch):
    a, b = (0, 0, 0, 0), (0, 0, 1, 1)
    monkeypatch.setattr(
        pipeline_module, "guided_completion",
        lambda *_args, **_kwargs: SimpleNamespace(
            solved=False, first_moves_examined=10, second_moves_examined=20,
        ),
    )
    monkeypatch.setattr(
        pipeline_module, "beam_repair",
        lambda *_args, **_kwargs: SimpleNamespace(
            solved=True, a=a, b=b, depth=1, states_examined=50, elapsed_time=0.1,
        ),
    )
    result = _run_optional_z3(
        PipelineConfig(L=4, z3=True), 4, (SAElite(a, b, 1, 1, 0),), tmp_path
    )
    assert result["status"] == "BEAM_SAT"
    assert result["solved"] and not result["z3_executed"]


def test_improved_beam_miss_calls_positive_radius_z3(tmp_path, monkeypatch):
    import solver.z3_solver as z3_solver_module
    a = b = (0, 0, 1, 1)
    monkeypatch.setattr(
        pipeline_module, "guided_completion",
        lambda *_args, **_kwargs: SimpleNamespace(
            solved=False, first_moves_examined=10, second_moves_examined=20,
        ),
    )
    monkeypatch.setattr(
        pipeline_module, "beam_repair",
        lambda *_args, **_kwargs: SimpleNamespace(
            solved=False, a=a, b=b, initial_score=20, best_score=8,
            states_examined=100,
        ),
    )
    calls = []
    monkeypatch.setattr(
        z3_solver_module, "solve_with_z3",
        lambda L, center_a, center_b, radius, timeout_ms: calls.append(
            (L, center_a, center_b, radius, timeout_ms)
        ) or SimpleNamespace(
            status="UNKNOWN", verified=False, a=None, b=None, profile=None,
            radius=radius, elapsed_time=0.5, reason="timeout",
        ),
    )
    result = _run_optional_z3(
        PipelineConfig(L=4, z3=True, z3_timeout=7, z3_completion_radius=4),
        4, (SAElite(a, b, 20, 1, 0),), tmp_path,
    )
    assert calls == [(4, a, b, 4, 7000)]
    assert result["status"] == "Z3_UNKNOWN" and result["z3_executed"]


def test_structured_handoff_passes_target_content_cases_to_z3(tmp_path, monkeypatch):
    import solver.z3_solver as z3_solver_module
    a = b = (0, 0, 1, 1)
    monkeypatch.setattr(
        pipeline_module, "guided_completion",
        lambda *_args, **_kwargs: SimpleNamespace(
            solved=False, first_moves_examined=1, second_moves_examined=1,
        ),
    )
    monkeypatch.setattr(
        pipeline_module, "beam_repair",
        lambda *_args, **_kwargs: SimpleNamespace(
            solved=False, a=a, b=b, initial_score=20, best_score=8,
            states_examined=10,
        ),
    )
    calls = []
    monkeypatch.setattr(
        z3_solver_module, "solve_with_z3",
        lambda L, center_a, center_b, **kwargs: calls.append(kwargs) or SimpleNamespace(
            status="UNKNOWN", verified=False, a=None, b=None, profile=None,
            radius=kwargs["radius"], elapsed_time=0.1, reason="timeout",
        ),
    )
    encoded = [[1, 1, 0, 0, 1, 1]]
    _run_optional_z3(
        PipelineConfig(L=4, z3=True), 4, (SAElite(a, b, 20, 1, 0),),
        tmp_path, encoded,
    )
    profiles = calls[0]["allowed_target_content_profiles"]
    assert len(profiles) == 1
    assert (profiles[0].k, profiles[0].eta) == (1, 1)


def test_unimproved_beam_miss_still_executes_bounded_z3(tmp_path, monkeypatch):
    import solver.z3_solver as z3_solver_module
    a = b = (0, 0, 1, 1)
    monkeypatch.setattr(
        pipeline_module, "guided_completion",
        lambda *_args, **_kwargs: SimpleNamespace(
            solved=False, first_moves_examined=10, second_moves_examined=20,
        ),
    )
    monkeypatch.setattr(
        pipeline_module, "beam_repair",
        lambda *_args, **_kwargs: SimpleNamespace(
            solved=False, a=a, b=b, initial_score=8, best_score=8,
            states_examined=100,
        ),
    )
    calls = []
    monkeypatch.setattr(
        z3_solver_module, "solve_with_z3",
        lambda L, center_a, center_b, radius, timeout_ms: calls.append(
            (L, center_a, center_b, radius, timeout_ms)
        ) or SimpleNamespace(
            status="UNKNOWN", verified=False, a=None, b=None, profile=None,
            radius=radius, elapsed_time=0.5, reason="timeout",
        ),
    )
    result = _run_optional_z3(
        PipelineConfig(L=4, z3=True), 4, (SAElite(a, b, 8, 1, 0),), tmp_path
    )
    assert calls == [(4, a, b, 4, 60000)]
    assert result["status"] == "Z3_UNKNOWN"
    assert not result["solved"] and result["z3_executed"]
    assert result["restart_recommended"]


def test_main_formats_normal_non_solution_result(capsys, tmp_path, monkeypatch):
    # Route generated files into a temporary working directory without changing solver behaviour.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(main, "run_c_search", lambda config, **_kwargs: SimpleNamespace(
        L=config.L, elapsed=0.01, restarts=1, moves=10,
        swap_evaluations=20, valid_results=0, verified_candidates=0,
        new_solutions=0, fkm_seeds_applied=1, fkm_seed_misses=0,
        seed_bank_path=tmp_path / "seeds.txt",
        log_path=tmp_path / "search.log", moves_per_second=1000.0,
        swap_evaluations_per_second=2000.0,
    ))
    assert main.main(["--L", "4", "--seconds", "0.001", "--backend", "c"]) == 0
    output = capsys.readouterr().out
    assert "[pipeline] 搜尋開始前既有結果：0 組" in output
    assert "========== 最終驗證摘要 ==========" in output
    assert "========== 搜尋效率統計 ==========" in output
    assert "0 組通過 Python 獨立 verifier" in output
    assert "Z3 觸發使用次數" not in output


def test_c_z3_mode_prints_execution_count_with_search_statistics(
    capsys, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)

    def fake_c(config, line_callback=None, **_kwargs):
        assert line_callback is not None
        line_callback(
            "搜尋統計：restart=3, moves=10, swap evaluations=20, "
            "valid results=0"
        )
        return SimpleNamespace(
            L=config.L, elapsed=0.01, restarts=3, moves=10,
            swap_evaluations=20, valid_results=0, verified_candidates=0,
            new_solutions=0, fkm_seeds_applied=1, fkm_seed_misses=0,
            seed_bank_path=tmp_path / "seeds.txt",
            log_path=tmp_path / "search.log", moves_per_second=1000.0,
            swap_evaluations_per_second=2000.0,
        )

    monkeypatch.setattr(main, "run_c_search", fake_c)
    assert main.main([
        "--L", "4", "--seconds", "0.001", "--z3", "--z3-timeout", "0",
        "--backend", "c",
    ]) == 0
    output = capsys.readouterr().out
    # One report follows the periodic C statistics and one is in the final
    # summary after the completion worker has been joined.
    assert output.count("Z3 觸發使用次數：0") == 2


@pytest.mark.parametrize("completion_duplicate", [False, True])
def test_c_and_completion_display_new_persisted_count_even_during_close(
    tmp_path, monkeypatch, capsys, completion_duplicate
):
    from solver.search_runner import append_verified_solution_if_new

    monkeypatch.chdir(tmp_path)
    a, b = (0, 0, 0, 0), (0, 0, 1, 1)
    # Pre-existing results must not contribute to this run's counter.
    assert append_verified_solution_if_new(4, a, (1, 1, 0, 0), tmp_path)

    def fake_completion(*_args):
        other = (b, a) if completion_duplicate else (a, (0, 1, 1, 0))
        written = append_verified_solution_if_new(4, *other, tmp_path)
        return {"solved": True, "status": "Z3_SAT", "z3_executed": True,
                "new_solution_written": written}

    def fake_c(config, line_callback=None, elite_callback=None):
        assert not config.echo
        assert append_verified_solution_if_new(4, a, b, tmp_path)
        line_callback("搜尋統計：restart=3, moves=10, swap evaluations=20, valid results=99")
        elite_callback(a, b, {"k": 1, "sign": 1})
        return SimpleNamespace(
            elapsed=0.01, restarts=3, moves=10, valid_results=99,
            verified_candidates=99, new_solutions=1, fkm_seeds_applied=1,
            fkm_seed_misses=0, seed_bank_path=tmp_path / "seeds.txt",
            log_path=tmp_path / "search.log", moves_per_second=1000.0,
            swap_evaluations_per_second=2000.0,
        )

    monkeypatch.setattr(main, "_run_optional_z3", fake_completion)
    monkeypatch.setattr(main, "run_c_search", fake_c)
    assert main.main(["--L", "4", "--seconds", "0.001", "--z3", "--backend", "c"]) == 0
    output = capsys.readouterr().out
    expected = 1 if completion_duplicate else 2
    counts = [int(line.rsplit("=", 1)[1]) for line in output.splitlines()
              if line.startswith("搜尋統計：")]
    assert counts == [1, expected]
    assert "最終 valid results（本輪新增，C + completion）：{}".format(expected) in output
    assert "Z3 觸發使用次數：1" in output


def test_final_restart_average_uses_only_new_pqcp_count(capsys):
    main._print_restarts_per_new_pqcp(12, 3)
    assert (
        "平均每找到 1 組新 PQCP 所需 restart：4.000 次"
        in capsys.readouterr().out
    )


def test_final_restart_average_is_na_when_no_new_pqcp(capsys):
    main._print_restarts_per_new_pqcp(12, 0)
    assert "N/A（本輪新增 0 組）" in capsys.readouterr().out


def test_persisted_new_count_replaces_raw_c_statistics():
    line = (
        "搜尋統計：restart=100, moves=200, swap evaluations=400, "
        "valid results=2"
    )
    assert main._with_new_valid_results(line, 3).endswith(
        "valid results=3"
    )
    assert main._with_new_valid_results("ordinary output", 3) == (
        "ordinary output"
    )


def test_c_completion_bridge_counts_verified_completion_results(tmp_path):
    args = SimpleNamespace(repair=False, z3=False)
    bridge = main._CCompletionBridge(args, 4, tmp_path)
    bridge._results.extend((
        {"status": "GUIDANCE_SAT", "solved": True, "z3_executed": False},
        {"status": "Z3_UNKNOWN", "solved": False, "z3_executed": True},
        SimpleNamespace(solved=True),
    ))
    assert bridge.valid_result_count == 2


def test_c_elite_metadata_skips_only_existing_score_rejections(tmp_path, monkeypatch):
    bridge = main._CCompletionBridge(SimpleNamespace(repair=False, z3=False), 44, tmp_path)
    bridge.enabled = True
    bridge.args.z3_trigger_score = 16
    monkeypatch.setattr(main, "full_correlation_profile", lambda *_args: pytest.fail(
        "ineligible elite must not invoke the expensive full profile"))
    bridge.consider((), (), {"score": 20})
    bridge._last_score = 8
    bridge.consider((), (), {"score": 8})
    bridge.consider((), (), {"score": 12})


def test_c_elite_metadata_is_rechecked_before_completion_handoff(tmp_path):
    bridge = main._CCompletionBridge(SimpleNamespace(repair=False, z3=False), 4, tmp_path)
    bridge.enabled = True
    bridge.args.z3_trigger_score = 16
    with pytest.raises(RuntimeError, match="consistency check"):
        bridge.consider((0, 0, 0, 0), (0, 0, 1, 1), {"score": 1})


def test_c_completion_bridge_counts_only_actual_z3_executions(tmp_path):
    args = SimpleNamespace(repair=False, z3=False)
    bridge = main._CCompletionBridge(args, 4, tmp_path)
    bridge._results.extend((
        {"status": "GUIDANCE_MISS", "z3_executed": False},
        {"status": "Z3_UNKNOWN", "z3_executed": True},
        SimpleNamespace(status_counts={"SAT": 0, "UNSAT": 2, "UNKNOWN": 1}),
    ))
    assert bridge.trigger_count == 3
    assert bridge.z3_execution_count == 4


def test_pipeline_keyboard_interrupt_still_saves_checkpoint_and_rich_best(tmp_path, monkeypatch):
    def interrupt(_self):
        raise KeyboardInterrupt

    monkeypatch.setattr(SearchRunner, "step", interrupt)
    result = run_pipeline(PipelineConfig(L=4, seconds=1, root=tmp_path))
    assert result.interrupted
    assert result.checkpoint_path.exists() and result.best_path.exists()


def test_enhanced_verified_solution_restarts_instead_of_terminating():
    search = EnhancedSearch.new(4, 9, EnhancedParameters(stagnation_iterations=1000, two_bit_samples=1))
    a, b = (0, 0, 0, 0), (0, 0, 1, 1)
    correlation = CorrelationState(a, b)
    search._correlation = correlation
    search.state.current_a, search.state.current_b = a, b
    search.state.current_score = pqcp_objective(correlation.profile)
    search.state.best_a, search.state.best_b, search.state.best_score = a, b, 0
    search.state.finished = True
    prior_restart = search.state.restart_index
    _restart_after_enhanced_solution(search)
    assert not search.state.finished
    assert search.state.restart_index == prior_restart + 1
    assert search.state.rng_state == search._rng.getstate()


def test_project_pipeline_initializes_from_safe_l44_weight_schedule(tmp_path):
    result = run_pipeline(PipelineConfig(L=44, seconds=0, root=tmp_path))
    assert (sum(result.state.current_a), sum(result.state.current_b)) == (18, 20)
    assert result.state.algorithm_parameters.acceptance_mode == "fixed_target_full"
    assert result.state.algorithm_parameters.preserve_alternating_content
    assert result.state.algorithm_parameters.target_content_profiles
    assert result.state.algorithm_parameters.initial_temperature == 8.0
    assert result.state.algorithm_parameters.proposal_samples == DEFAULT_GUIDED_PROPOSAL_SAMPLES
    assert result.state.algorithm_parameters.objective_energy_weight == DEFAULT_GUIDED_OBJECTIVE_ENERGY_WEIGHT


def test_pipeline_threads_reversible_fkm_seed_selection_parameters(tmp_path):
    result = run_pipeline(PipelineConfig(
        L=44,
        seed=8080,
        seconds=0,
        root=tmp_path,
        fkm_seed_policy="phase_top_q",
        fkm_candidate_count=8,
        fkm_elite_count=3,
    ))
    parameters = result.state.algorithm_parameters
    assert parameters.fkm_seed_policy == "phase_top_q"
    assert parameters.fkm_candidate_count == 8
    assert parameters.fkm_elite_count == 3


def test_reference_port_is_isolated_and_uses_its_own_checkpoint(tmp_path):
    result = run_pipeline(PipelineConfig(
        L=44, seed=8080, seconds=0, root=tmp_path, reference_port=True,
    ))
    parameters = result.state.algorithm_parameters
    assert result.method == "reference"
    assert result.checkpoint_path.name == "L44_reference.json"
    assert parameters.fkm_seed_policy == "compressed_a_random_b_top_q"
    assert parameters.acceptance_mode == "fixed_target_multiscale"
    assert parameters.target_profile_offset == 0


def test_reference_port_reuses_existing_z3_completion_interface(tmp_path, monkeypatch):
    calls = []

    def fake_completion(config, length, elites, root, encoded_profiles=None):
        calls.append((length, elites[0].score, encoded_profiles))
        return {
            "status": "Z3_UNKNOWN", "solved": False,
            "z3_executed": True, "restart_recommended": True,
        }

    monkeypatch.setattr("solver.pipeline._run_optional_z3", fake_completion)
    result = run_pipeline(PipelineConfig(
        L=44, seed=123, seconds=0, root=tmp_path,
        reference_port=True, z3=True, z3_trigger_score=10_000,
        z3_profile_radius=10_000,
    ))
    assert calls and calls[0][0] == 44
    assert calls[0][2]
    assert result.z3_trigger_count == 1


def test_reference_checkpoint_resume_keeps_method_identity(tmp_path):
    first = run_pipeline(PipelineConfig(
        L=44, seed=91, seconds=0, root=tmp_path, reference_port=True,
    ))
    resumed = run_pipeline(PipelineConfig(
        seconds=0, root=tmp_path, resume=first.checkpoint_path,
    ))
    assert resumed.method == "reference"
    assert resumed.checkpoint_path == first.checkpoint_path
    assert resumed.state.iteration == first.state.iteration
    assert resumed.state.current_a == first.state.current_a
    assert resumed.state.current_b == first.state.current_b
    assert resumed.state.rng_state == first.state.rng_state


def test_reference_port_rejects_conflicting_initializers(tmp_path):
    with pytest.raises(ValueError, match="cannot be combined"):
        run_pipeline(PipelineConfig(
            L=44, seconds=0, root=tmp_path,
            reference_port=True, gcp=True,
        ))


def test_pipeline_rejects_invalid_fkm_seed_selection_controls(tmp_path):
    with pytest.raises(ValueError, match="unsupported FKM seed policy"):
        run_pipeline(PipelineConfig(L=44, seconds=0, root=tmp_path, fkm_seed_policy="bad"))
    with pytest.raises(ValueError, match="must not exceed"):
        run_pipeline(PipelineConfig(
            L=44, seconds=0, root=tmp_path,
            fkm_seed_policy="phase_top_q",
            fkm_candidate_count=2,
            fkm_elite_count=3,
        ))


def test_pipeline_rejects_project_length_without_any_admissible_weight_pair(tmp_path):
    with pytest.raises(ValueError, match="no ordinary/alternating target-content profile"):
        run_pipeline(PipelineConfig(L=58, seconds=0, root=tmp_path))


def test_gcp_pipeline_lift_is_verified_and_persisted(tmp_path):
    result = run_pipeline(PipelineConfig(L=4, seed=3, seconds=0, gcp=True, root=tmp_path))
    assert result.verification.is_valid
    assert result.verified_paths
    assert (tmp_path / "4.txt").exists()
    assert result.state.algorithm_parameters.acceptance_mode == "objective_plus_target_pair"
    assert result.state.algorithm_parameters.proposal_samples == DEFAULT_GUIDED_PROPOSAL_SAMPLES
    assert result.state.algorithm_parameters.objective_energy_weight == DEFAULT_GUIDED_OBJECTIVE_ENERGY_WEIGHT


def test_gcp_pipeline_rejects_unregistered_length(tmp_path):
    with pytest.raises(ValueError, match="no verified low-cost Golay construction"):
        run_pipeline(PipelineConfig(L=6, seconds=0, gcp=True, root=tmp_path))


@pytest.mark.parametrize(
    "length",
    (44, 46, 68, 86, 94),
)
def test_gcp_pipeline_projects_to_and_keeps_an_admissible_weight_pair(length, tmp_path):
    result = run_pipeline(PipelineConfig(L=length, seed=123, seconds=0, gcp=True, root=tmp_path))
    assert len(result.state.current_a) == len(result.state.current_b) == length
    from solver.weight_constraints import is_admissible_weight_pair
    assert is_admissible_weight_pair(length, sum(result.state.current_a), sum(result.state.current_b))
    assert pqcp_objective(result.verification.profile) == result.state.best_score
    assert result.state.temperature == GCP_INITIAL_TEMPERATURE
    assert result.state.algorithm_parameters.initial_temperature == 8.0


@pytest.mark.parametrize("length", (58, 90))
def test_fixed_weight_pipeline_rejects_arithmetically_impossible_project_lengths(length, tmp_path):
    with pytest.raises(ValueError, match="no integer Hamming-weight pair"):
        run_pipeline(PipelineConfig(L=length, seed=123, seconds=0, gcp=True, root=tmp_path))
