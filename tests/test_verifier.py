"""Tests for invalid PQCP candidates; no legal PQCP is generated here."""

from solver.problem import PQCPInstance
from solver.verifier import verify_pqcp


def test_verifier_rejects_nonbinary_sequence():
    result = verify_pqcp((0, 1, 2, 0), (0, 1, 0, 1))
    assert not result.is_valid
    assert any("A is not a binary" in error for error in result.errors)


def test_verifier_rejects_unequal_lengths():
    result = verify_pqcp((0, 1, 0), (0, 1, 0, 1))
    assert not result.is_valid
    assert "A and B have unequal lengths" in result.errors


def test_verifier_rejects_obviously_noncomplementary_pair():
    result = verify_pqcp((0, 0, 0, 0), (0, 0, 0, 0))
    assert not result.is_valid
    assert result.profile == (8, 8, 8, 8)
    assert result.nonzero_shifts == (1, 2, 3)
    assert any("magnitude 4" in error for error in result.errors)


def test_verifier_checks_instance_length_and_reports_profile():
    result = verify_pqcp((0, 1, 0, 1), (0, 0, 1, 1), PQCPInstance(L=6))
    assert not result.is_valid
    assert result.profile == (8, -4, 0, -4)
    assert any("does not match instance L=6" in error for error in result.errors)
