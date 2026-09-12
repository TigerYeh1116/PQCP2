"""Exact factor-compressed PACF signature pairing for Project 2 seeds.

For ``L=m*d`` let ``x^(m)[r] = sum_q x[r+q*d]`` be genuine factor-``m``
compression.  The periodic autocorrelation aliasing identity is

``PAF(x^(m))[r] = sum_q PAF(x)[r+q*d]``.

Consequently every pair realizing a Project target ``T`` must obey

``PAF(a^(m)) + PAF(b^(m)) = fold_m(T)``.

This module hashes the complete integer vectors on both sides.  A returned
match is therefore an exact necessary-condition match, not a floating-point
similarity score.  Failure to find a match in a bounded candidate pool is
*not* an impossibility proof for the full problem; callers must retain a
fallback or exhaust the underlying candidate family.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Optional, Sequence, Tuple

from .correlation import normalize_binary_sequence
from .psd_pruning import certified_binary_psd_pruning
from .structured_energy import (
    compress_binary_sequence,
    fold_profile,
    integer_periodic_autocorrelation_profile,
    target_profile,
)
from .target_profiles import TargetContentProfile


BinaryWord = Tuple[int, ...]
Signature = Tuple[Tuple[int, Tuple[int, ...]], ...]


@dataclass(frozen=True)
class CompressedPairMatch:
    """One pair whose complete compressed PACFs match every named factor."""

    a: BinaryWord
    b: BinaryWord
    factors: Tuple[int, ...]
    a_signature: Signature
    b_signature: Signature


def compression_factors(length: int) -> Tuple[int, ...]:
    """Return production compression factors available at ``length``."""
    if not isinstance(length, int) or isinstance(length, bool) or length < 4:
        raise ValueError("length must be an integer at least four")
    factors = [2] if length % 2 == 0 else []
    if length % 4 == 0:
        factors.append(4)
    return tuple(factors)


def compressed_pacf_signature(
    sequence: Sequence[int], factors: Sequence[int],
) -> Signature:
    """Return complete exact integer PACF vectors after each compression."""
    bits = normalize_binary_sequence(sequence)
    normalized = _normalize_factors(len(bits), factors)
    return _cached_compressed_pacf_signature(bits, normalized)


@lru_cache(maxsize=4096)
def _cached_compressed_pacf_signature(
    bits: BinaryWord, normalized: Tuple[int, ...],
) -> Signature:
    """Bound repeated archive scans without retaining an unbounded history."""
    return tuple(
        (factor, integer_periodic_autocorrelation_profile(
            compress_binary_sequence(bits, factor)
        ))
        for factor in normalized
    )


def required_partner_signature(
    a: Sequence[int],
    target: TargetContentProfile,
    factors: Optional[Sequence[int]] = None,
) -> Signature:
    """Return the unique compressed PACF signature required from partner B."""
    a_bits = normalize_binary_sequence(a)
    _validate_target(target, len(a_bits))
    normalized = _normalize_factors(
        target.L,
        compression_factors(target.L) if factors is None else factors,
    )
    desired = target_profile(target.L, target.k, target.eta)
    observed = dict(compressed_pacf_signature(a_bits, normalized))
    return tuple(
        (
            factor,
            tuple(
                wanted - actual
                for wanted, actual in zip(
                    fold_profile(desired, factor), observed[factor]
                )
            ),
        )
        for factor in normalized
    )


def compressed_pair_matches_target(
    a: Sequence[int],
    b: Sequence[int],
    target: TargetContentProfile,
    factors: Optional[Sequence[int]] = None,
) -> bool:
    """Test the exact compression identity for a complete binary pair."""
    a_bits = normalize_binary_sequence(a)
    b_bits = normalize_binary_sequence(b)
    if len(a_bits) != len(b_bits):
        raise ValueError("a and b must have equal lengths")
    _validate_target(target, len(a_bits))
    normalized = _normalize_factors(
        target.L,
        compression_factors(target.L) if factors is None else factors,
    )
    return compressed_pacf_signature(b_bits, normalized) == required_partner_signature(
        a_bits, target, normalized
    )


def pair_by_compressed_signature(
    a_candidates: Iterable[Sequence[int]],
    b_candidates: Iterable[Sequence[int]],
    target: TargetContentProfile,
    *,
    factors: Optional[Sequence[int]] = None,
    limit: Optional[int] = None,
    general_psd_pruning: bool = False,
) -> Tuple[CompressedPairMatch, ...]:
    """Hash-join bounded A/B pools by complete exact compressed signatures.

    The function deduplicates complete words while preserving encounter order.
    ``limit`` caps returned pairs, not examined candidates.  When
    ``general_psd_pruning`` is enabled, an individual word is omitted only if
    the certified all-frequency PSD cap proves that it cannot participate in
    this exact target.  An empty result means only that these supplied pools
    do not contain a match.
    """
    if limit is not None and (
        not isinstance(limit, int) or isinstance(limit, bool) or limit < 0
    ):
        raise ValueError("limit must be a non-negative integer or None")
    if limit == 0:
        return ()
    if not isinstance(general_psd_pruning, bool):
        raise ValueError("general_psd_pruning must be boolean")
    normalized = _normalize_factors(
        target.L,
        compression_factors(target.L) if factors is None else factors,
    )
    a_words = _unique_words(a_candidates, target.L)
    b_words = _unique_words(b_candidates, target.L)
    buckets = {}
    b_signatures = {}
    for b in b_words:
        if general_psd_pruning and certified_binary_psd_pruning(
            b, target, stop_at_first_violation=True,
        ).should_prune:
            continue
        signature = compressed_pacf_signature(b, normalized)
        b_signatures[b] = signature
        buckets.setdefault(signature, []).append(b)

    matches = []
    seen_pairs = set()
    for a in a_words:
        if general_psd_pruning and certified_binary_psd_pruning(
            a, target, stop_at_first_violation=True,
        ).should_prune:
            continue
        a_signature = compressed_pacf_signature(a, normalized)
        needed = required_partner_signature(a, target, normalized)
        for b in buckets.get(needed, ()):
            key = (a, b)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            matches.append(CompressedPairMatch(
                a=a,
                b=b,
                factors=normalized,
                a_signature=a_signature,
                b_signature=b_signatures[b],
            ))
            if limit is not None and len(matches) >= limit:
                return tuple(matches)
    return tuple(matches)


def _unique_words(candidates: Iterable[Sequence[int]], length: int) -> Tuple[BinaryWord, ...]:
    words = []
    seen = set()
    for candidate in candidates:
        word = normalize_binary_sequence(candidate)
        if len(word) != length:
            raise ValueError("candidate length does not match target")
        if word not in seen:
            seen.add(word)
            words.append(word)
    return tuple(words)


def _normalize_factors(length: int, factors: Sequence[int]) -> Tuple[int, ...]:
    try:
        normalized = tuple(sorted(set(factors)))
    except TypeError as error:
        raise ValueError("factors must be an iterable of integers") from error
    if not normalized:
        raise ValueError("at least one compression factor is required")
    if any(
        not isinstance(factor, int) or isinstance(factor, bool)
        or factor < 2 or length % factor
        for factor in normalized
    ):
        raise ValueError("every factor must be an integer divisor of L at least two")
    return normalized


def _validate_target(target: TargetContentProfile, length: int) -> None:
    if not isinstance(target, TargetContentProfile) or target.L != length:
        raise ValueError("target must be a matching TargetContentProfile")
    # Reuse the exact target constructor's range/sign validation.
    target_profile(target.L, target.k, target.eta)


__all__ = (
    "CompressedPairMatch",
    "compressed_pacf_signature",
    "compressed_pair_matches_target",
    "compression_factors",
    "pair_by_compressed_signature",
    "required_partner_signature",
)
