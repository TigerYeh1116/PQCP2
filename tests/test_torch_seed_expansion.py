"""Tests for verified-solution MPS seed expansion."""

import pytest

pytest.importorskip("torch")

from solver.correlation import full_correlation_profile
from solver.objective import pqcp_objective
from solver.target_profiles import pair_content
from solver.torch_seed_expansion import (
    SeedExpansionConfig, expand_solution_neighborhoods, load_verified_solution_seeds,
)
from solver.verifier import verify_pqcp


KNOWN = (
    tuple(map(int, "10100001100011100100011010001011000010011111")),
    tuple(map(int, "01010100011010100100111110011010010000000100")),
)


def test_solution_loader_verifies_and_deduplicates_exchange(tmp_path):
    path = tmp_path / "44.txt"
    path.write_text(
        "a={}\nb={}\n\na={}\nb={}\n".format(
            "".join(map(str, KNOWN[0])), "".join(map(str, KNOWN[1])),
            "".join(map(str, KNOWN[1])), "".join(map(str, KNOWN[0])),
        )
    )
    assert load_verified_solution_seeds(path, 44) == (KNOWN,)


def test_expansion_is_deterministic_exact_and_content_preserving():
    config = SeedExpansionConfig(
        44, device="cpu", clones_per_solution=4, max_swap_radius=4, limit=4,
    )
    first = expand_solution_neighborhoods((KNOWN,), config)
    second = expand_solution_neighborhoods((KNOWN,), config)
    assert first == second and len(first) == 4
    required = pair_content(*KNOWN)
    for record in first:
        assert record["swap_radius"] >= 1
        assert pair_content(record["A"], record["B"]) == required
        assert record["profile"] == full_correlation_profile(record["A"], record["B"])
        assert record["score"] == pqcp_objective(record["profile"])


def test_expansion_rejects_non_solution_seed():
    with pytest.raises(ValueError, match="verified"):
        expand_solution_neighborhoods(
            (((0,) * 44, (0,) * 44),),
            SeedExpansionConfig(44, device="cpu", clones_per_solution=1, limit=1),
        )
