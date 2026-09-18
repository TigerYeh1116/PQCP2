"""Tests for the audited PyTorch-elite to discrete-SA bridge."""

from itertools import product
import json

import pytest

from solver.correlation import full_correlation_profile
from solver.objective import pqcp_objective
from solver.target_profiles import pair_content, target_content_profiles
from solver.torch_hybrid import elite_pair_key, write_elite_pair_seed_bank
from solver.verifier import verify_pqcp


def _non_solution_elite(length=8):
    """Return one exact, target-content-compatible synthetic archive item."""
    for target in target_content_profiles(length, decimation_reduced=False):
        for a in product((0, 1), repeat=length):
            if (sum(a[0::2]), sum(a[1::2])) != (
                target.a_even_ones, target.a_odd_ones
            ):
                continue
            for b in product((0, 1), repeat=length):
                if pair_content(a, b) != (
                    target.a_even_ones, target.a_odd_ones,
                    target.b_even_ones, target.b_odd_ones,
                ):
                    continue
                if not verify_pqcp(a, b).is_valid:
                    profile = full_correlation_profile(a, b)
                    return {
                        "A": list(a), "B": list(b), "profile": list(profile),
                        "score": pqcp_objective(profile), "iteration": 7,
                        "k": target.k, "sign": target.eta,
                    }
    raise AssertionError("test fixture could not find a non-solution")


def test_elite_pair_key_identifies_a_b_exchange_only():
    a, b = (0, 0, 1, 1), (0, 1, 0, 1)
    assert elite_pair_key(a, b) == elite_pair_key(b, a)
    assert elite_pair_key(a, b) != elite_pair_key(a[::-1], b)


def test_elite_seed_bank_is_exact_deduplicated_and_traceable(tmp_path):
    elite = _non_solution_elite()
    exchanged = {
        **elite, "A": elite["B"], "B": elite["A"], "iteration": 8,
    }
    exchanged["profile"] = full_correlation_profile(exchanged["A"], exchanged["B"])
    destination = tmp_path / "elite_pairs.txt"
    records = write_elite_pair_seed_bank(
        (exchanged, elite), destination, length=8,
    )
    assert len(records) == 1
    assert destination.read_text(encoding="ascii").count("\na=") == 1
    sidecar = json.loads(destination.with_suffix(".txt.json").read_text())
    assert sidecar["format"] == "pqcp-torch-elite-seeds"
    saved = sidecar["records"][0]
    assert saved["profile"] == full_correlation_profile(saved["A"], saved["B"])
    assert saved["score"] == pqcp_objective(saved["profile"])
    assert pair_content(saved["A"], saved["B"]) == pair_content(elite["A"], elite["B"])
    assert 1 <= saved["k"] < 4 and saved["sign"] in (-1, 1)


def test_elite_seed_bank_rejects_stale_accelerator_score(tmp_path):
    elite = _non_solution_elite()
    elite["score"] += 1
    with pytest.raises(ValueError, match="full recomputation"):
        write_elite_pair_seed_bank((elite,), tmp_path / "bad.txt", length=8)


def test_target_aligned_bank_preserves_source_case(tmp_path):
    elite = _non_solution_elite()
    destination = tmp_path / "aligned.txt"
    record = write_elite_pair_seed_bank(
        (elite,), destination, length=8, retarget=False,
    )[0]
    assert (record["k"], record["sign"]) == (elite["k"], elite["sign"])
    elite["k"] = 999
    with pytest.raises(ValueError, match="incompatible"):
        write_elite_pair_seed_bank(
            (elite,), tmp_path / "wrong.txt", length=8, retarget=False,
        )
