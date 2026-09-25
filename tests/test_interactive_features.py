"""One-click settings only: tests never launch a GPU, C process, or Z3."""

import math

import pytest

import main


def run_interactive(monkeypatch, tmp_path, answers):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["main.py"])
    remaining = iter(answers)
    prompts, captured = [], {}

    def answer(prompt):
        prompts.append(prompt)
        return next(remaining)

    def backend(args, budget):
        captured.update(args=args, budget=budget)
        return 0

    monkeypatch.setattr("builtins.input", answer)
    monkeypatch.setattr(main, "_run_torch_backend", backend)
    assert main.main() == 0
    assert list(remaining) == []
    assert math.isinf(captured["budget"])
    return captured["args"], prompts


@pytest.mark.parametrize("length", [46, 86, 94])
def test_one_click_can_enable_all_new_features(monkeypatch, tmp_path, capsys, length):
    args, prompts = run_interactive(monkeypatch, tmp_path,
                                   [str(length), "321", "", "", "y", " Y ", "yes", "y"])
    assert args.seed == 321 and args.z3 and args.repair
    assert args.torch_mathematical_loss == "lattice"
    assert args.torch_polish_mode == "alternating"
    assert args.torch_direct_compression and args.z3_orientation_search
    assert args.z3_orientation_preprocessing and args.z3_compressed_lift
    assert args.z3_lift_enumeration
    assert len(prompts) == 8
    assert "直接三值壓縮=開啟" in capsys.readouterr().out


def test_enter_preserves_previous_defaults(monkeypatch, tmp_path):
    args, _ = run_interactive(monkeypatch, tmp_path, ["86", "", "", "", "", "", "", ""])
    assert args.seed == 123 and args.z3
    assert args.torch_polish_mode == "baseline"
    assert not args.torch_direct_compression and not args.z3_orientation_search
    assert not args.z3_lift_enumeration


@pytest.mark.parametrize("direct,direction", [("y", "n"), ("n", "y")])
def test_two_new_features_are_independent(monkeypatch, tmp_path, direct, direction):
    args, _ = run_interactive(monkeypatch, tmp_path, ["86", "", "", "", "n", direct, direction, "n"])
    assert args.torch_direct_compression == (direct == "y")
    assert args.z3_orientation_search == (direction == "y")
    assert args.torch_polish_mode == "baseline"


def test_z3_declined_is_not_silently_reenabled(monkeypatch, tmp_path, capsys):
    args, prompts = run_interactive(monkeypatch, tmp_path, ["86", "", "n", "", "y"])
    assert not args.z3 and not args.repair
    assert args.torch_polish_mode == "alternating"
    assert not args.torch_direct_compression and not args.z3_orientation_search
    assert len(prompts) == 5
    assert "不會自動開啟 Z3" in capsys.readouterr().out


@pytest.mark.parametrize("length", [4, 44, 68, 100])
def test_unsupported_lengths_keep_existing_four_questions(monkeypatch, tmp_path, capsys, length):
    args, prompts = run_interactive(monkeypatch, tmp_path, [str(length), "", "", ""])
    assert len(prompts) == 4
    assert not args.torch_direct_compression and not args.z3_orientation_search
    assert args.torch_polish_mode == "baseline"
    assert "略過此選單" in capsys.readouterr().out


def test_invalid_choice_reprompts_without_starting_search(monkeypatch, tmp_path, capsys):
    args, prompts = run_interactive(monkeypatch, tmp_path, ["86", "", "", "", "maybe", "y", "n", "n", "n"])
    assert args.torch_polish_mode == "alternating"
    assert len(prompts) == 9
    assert "請輸入 y 或 n" in capsys.readouterr().out


def test_explicit_cli_does_not_prompt(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: pytest.fail("unexpected input prompt"))
    seen = []
    monkeypatch.setattr(main, "_run_torch_backend", lambda args, budget: seen.append(args) or 0)
    assert main.main(["--L", "86", "--torch-direct-compression", "--z3-orientation-search"]) == 0
    assert seen[0].torch_direct_compression and seen[0].z3_orientation_search
