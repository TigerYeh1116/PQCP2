"""Bounded beam repair correctness without long-running searches."""

import pytest

from solver.beam_repair import beam_repair
from solver.objective import pqcp_objective
from solver.verifier import verify_pqcp


KNOWN_A = (1, 0, 0, 0, 0, 0)
KNOWN_B = (1, 1, 0, 0, 0, 0)


def test_beam_repair_accepts_and_verifies_exact_center():
    result = beam_repair(KNOWN_A, KNOWN_B)
    assert result.solved and result.depth == 0
    assert result.best_score == 0
    assert verify_pqcp(result.a, result.b).is_valid


def test_beam_repair_recovers_one_fixed_weight_swap():
    perturbed_b = (1, 0, 0, 1, 0, 0)
    result = beam_repair(KNOWN_A, perturbed_b, max_depth=2, beam_width=3)
    assert result.solved and result.depth == 1
    assert result.a is not None and result.b is not None
    assert verify_pqcp(result.a, result.b).is_valid


def test_beam_miss_is_not_reported_as_a_solution_or_unsat():
    result = beam_repair((0, 0, 0, 1, 1, 1), (0, 0, 0, 1, 1, 1), max_depth=0)
    assert not result.solved
    assert result.depth is None
    assert len(result.a) == len(result.b) == len(result.profile) == 6
    assert result.best_score <= result.initial_score
    assert result.best_score == pqcp_objective(result.profile)


@pytest.mark.parametrize("kwargs", (
    {"max_depth": -1}, {"beam_width": 0}, {"objective_energy_weight": 0},
    {"ranking_mode": "unknown"},
))
def test_beam_repair_rejects_invalid_controls(kwargs):
    with pytest.raises(ValueError):
        beam_repair(KNOWN_A, KNOWN_B, **kwargs)
