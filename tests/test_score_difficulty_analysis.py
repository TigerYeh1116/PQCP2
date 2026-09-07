"""Lightweight statistical helper tests; exhaustive analysis is not run by pytest."""

import pytest

from experiments.analyze_score_difficulty import spearman_rank_correlation


def test_spearman_handles_ties_and_direction():
    assert spearman_rank_correlation((1, 2, 3), (10, 20, 30)) == pytest.approx(1.0)
    assert spearman_rank_correlation((1, 2, 3), (30, 20, 10)) == pytest.approx(-1.0)
    assert 0 < spearman_rank_correlation((1, 1, 2, 3), (0, 1, 2, 3)) < 1


def test_spearman_rejects_mismatched_or_tiny_inputs():
    with pytest.raises(ValueError):
        spearman_rank_correlation((1,), (1,))
    with pytest.raises(ValueError):
        spearman_rank_correlation((1, 2), (1,))
