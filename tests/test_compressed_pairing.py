"""Exact and exhaustive tests for compressed PACF signature hash pairing."""

from itertools import product
from types import SimpleNamespace

import pytest

from solver.compressed_pairing import (
    compressed_pair_matches_target,
    compressed_pacf_signature,
    pair_by_compressed_signature,
    required_partner_signature,
)
from solver.target_profiles import (
    TargetContentProfile,
    canonical_target_content_profiles,
    pair_content,
)
from solver.verifier import verify_pqcp


KNOWN_L44_A = tuple(map(int, "10001001110010011010011110101000011000010000"))
KNOWN_L44_B = tuple(map(int, "11000000111101111000001011001010101101000100"))


def _known_target(a, b):
    check = verify_pqcp(a, b)
    assert check.is_valid
    shift = min(check.nonzero_shifts)
    sign = check.profile[shift] // 4
    return TargetContentProfile(len(a), shift, sign, *pair_content(a, b))


def test_known_l44_solution_matches_factor_two_and_factor_four_exactly():
    target = _known_target(KNOWN_L44_A, KNOWN_L44_B)
    a, b = KNOWN_L44_A, KNOWN_L44_B
    assert compressed_pair_matches_target(a, b, target, (2, 4))
    assert compressed_pacf_signature(b, (2, 4)) == required_partner_signature(
        a, target, (2, 4)
    )


@pytest.mark.parametrize("length", (4, 8))
def test_hash_pairing_equals_complete_brute_force_cross_product(length):
    target = canonical_target_content_profiles(length)[0]
    words = tuple(product((0, 1), repeat=length))
    a_pool = tuple(
        word for word in words
        if pair_content(word, (0,) * length)[:2] == (
            target.a_even_ones, target.a_odd_ones
        )
    )
    b_pool = tuple(
        word for word in words
        if pair_content((0,) * length, word)[2:] == (
            target.b_even_ones, target.b_odd_ones
        )
    )
    factors = (2, 4) if length % 4 == 0 else (2,)
    expected = {
        (a, b) for a in a_pool for b in b_pool
        if compressed_pair_matches_target(a, b, target, factors)
    }
    actual = pair_by_compressed_signature(
        a_pool + a_pool[:2], b_pool + b_pool[:2], target, factors=factors
    )
    assert {(match.a, match.b) for match in actual} == expected
    assert len(actual) == len(expected)


def test_empty_bounded_pool_is_not_reported_as_global_impossibility():
    target = canonical_target_content_profiles(8)[0]
    a = (0,) * 8
    b = (0,) * 8
    assert pair_by_compressed_signature((a,), (b,), target) == ()


@pytest.mark.parametrize(
    "kwargs",
    (
        {"factors": ()},
        {"factors": (3,)},
        {"limit": -1},
    ),
)
def test_invalid_pairing_controls_raise(kwargs):
    target = canonical_target_content_profiles(8)[0]
    with pytest.raises(ValueError):
        pair_by_compressed_signature(((0,) * 8,), ((0,) * 8,), target, **kwargs)


def test_zero_limit_returns_no_matches_without_special_cases():
    target = canonical_target_content_profiles(8)[0]
    words = tuple(product((0, 1), repeat=8))
    assert pair_by_compressed_signature(words, words, target, limit=0) == ()


def test_torch_fkm_initialization_prefers_available_exact_matches(tmp_path):
    torch = pytest.importorskip("torch")
    from solver.torch_search import TorchSearch, TorchSearchConfig

    search = TorchSearch(TorchSearchConfig(
        8, device="cpu", batch_size=4, fkm_pool_size=8,
        symmetry_pruning=True, frequency_pruning=True,
        compression_signature_pairing=True,
    ), tmp_path)
    assert search._compression_pair_cache
    assert all(search._compression_pair_cache.values())
    pairs = search._initial_pairs(search.model.profiles, generation=11).to(
        dtype=torch.int64
    )
    for index, target in enumerate(search.model.profiles):
        a = tuple(pairs[index, 0].tolist())
        b = tuple(pairs[index, 1].tolist())
        assert compressed_pair_matches_target(a, b, target, (2, 4))


def test_archive_sides_are_recombined_and_fully_recomputed(tmp_path):
    from solver.objective import pqcp_objective
    from solver.torch_hybrid_runner import _compression_signature_records

    target = canonical_target_content_profiles(8)[0]
    words = tuple(product((0, 1), repeat=8))
    a_pool = tuple(word for word in words if pair_content(word, (0,) * 8)[:2] == (
        target.a_even_ones, target.a_odd_ones
    ))
    b_pool = tuple(word for word in words if pair_content((0,) * 8, word)[2:] == (
        target.b_even_ones, target.b_odd_ones
    ))
    exact_matches = pair_by_compressed_signature(a_pool, b_pool, target)
    selected = next(
        match for match in exact_matches
        if not verify_pqcp(match.a, match.b).is_valid
    )
    other_a = next(word for word in a_pool if word != selected.a)
    other_b = next(word for word in b_pool if word != selected.b)
    records = [
        {"A": list(selected.a), "B": list(other_b),
         "k": target.k, "sign": target.eta},
        {"A": list(other_a), "B": list(selected.b),
         "k": target.k, "sign": target.eta},
    ]
    search = SimpleNamespace(
        config=SimpleNamespace(
            L=8, seed=77, compression_signature_pairing=True,
        ),
        root=tmp_path,
        epoch=12,
        generation=3,
        verified_hits=0,
        new_solutions=0,
        available=(target,),
    )
    output = _compression_signature_records(search, records, limit=16)
    combined = next(
        record for record in output
        if tuple(record["A"]) == selected.a and tuple(record["B"]) == selected.b
    )
    check = verify_pqcp(combined["A"], combined["B"])
    assert combined["method"] == "exact_compressed_signature_pairing"
    assert combined["compression_factors"] == [2, 4]
    assert combined["profile"] == list(check.profile)
    assert combined["score"] == pqcp_objective(check.profile)
