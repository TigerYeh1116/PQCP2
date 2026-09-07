"""Deterministic fixed-content FKM seed selection strategies.

The FKM generator deliberately yields one lexicographically minimal word per
cyclic orbit.  Interleaving four such representatives with their zero phases
aligned is a representation choice, not a Project 2 condition.  This module
keeps the four exact target-content counts fixed while optionally varying the
relative cyclic phase of the odd subsequence in A and B.

``legacy`` reproduces :func:`solver.annealing.initialize_from_fkm_content_profile`
exactly and is the rollback path.  New policies use stable, domain-separated
RNG streams, so changing phase enumeration or elite selection never consumes
the RNG stream later used by simulated annealing.

``compressed_top_q`` uses ternary fixed-content FKM necklaces for both
sequences.  ``compressed_a_random_b_top_q`` follows the reference project more
closely: A comes from that structured queue while B is independently sampled
under the same exact even/odd content constraints.  Both lift every compressed
symbol to positions ``i`` and ``i+L/2``.

For ``phase_top_q``, candidates are ordered first by the exact full
fixed-target energy.  Exact factor-two compression, and factor four whenever
L is divisible by four, are used only as deterministic tie-breaks.  These are
seed-ranking heuristics; they do not alter the objective, verifier, or any
mathematical feasibility condition.
"""

from dataclasses import dataclass
import hashlib
from itertools import islice
import random
from typing import Sequence, Tuple

from .correlation import full_correlation_profile
from .compressed_fkm import compressed_fkm_lift_candidates
from .fkm import generate_fkm_sequences
from .objective import pqcp_objective
from .structured_energy import structured_energy_breakdown
from .target_profiles import TargetContentProfile, pair_content


BinaryWord = Tuple[int, ...]
BinaryPair = Tuple[BinaryWord, BinaryWord]
EnergyKey = Tuple[int, ...]
Content = Tuple[int, int, int, int]


SEED_POLICIES = (
    "legacy", "phase_random", "phase_top_q", "compressed_top_q",
    "compressed_a_random_b_top_q",
)
_RNG_NAMESPACE = "pqcp2-fixed-content-seed-selection-v1"


@dataclass(frozen=True)
class SeedSelectionDiagnostics:
    """Immutable information needed to audit one initialization decision.

    ``selected_rank`` is zero-based within ``ranked_energy_keys``.
    ``ranked_phases`` and ``ranked_energy_keys`` use the same order.  An
    energy key is ``(E_full, E_2)`` or ``(E_full, E_2, E_4)``; compressed
    components never precede or replace the complete fixed-target energy.
    """

    policy: str
    seed: int
    pool_size: int
    requested_candidates: int
    candidates_examined: int
    elite_count: int
    selected_rank: int
    selected_phase_a: int
    selected_phase_b: int
    selected_energy_key: EnergyKey
    selected_objective: int
    compression_factors: Tuple[int, ...]
    target_content: Content
    pool_indices: Tuple[int, int, int, int]
    unique_profiles: int
    ranked_phases: Tuple[Tuple[int, int], ...]
    ranked_energy_keys: Tuple[EnergyKey, ...]


@dataclass(frozen=True)
class SeedSelectionResult:
    """One complete fixed-content A/B seed and its selection diagnostics."""

    a: BinaryWord
    b: BinaryWord
    diagnostics: SeedSelectionDiagnostics

    @property
    def pair(self) -> BinaryPair:
        """Return ``(a, b)`` for callers that need the existing pair shape."""
        return self.a, self.b


@dataclass(frozen=True)
class _Candidate:
    """Internal exact seed candidate and its deterministic ranking values."""

    a: BinaryWord
    b: BinaryWord
    phase_a: int
    phase_b: int
    profile: Tuple[int, ...]
    energy_key: EnergyKey
    objective: int


def select_fkm_content_seed(
    profile: TargetContentProfile,
    seed: int,
    pool_size: int = 128,
    policy: str = "legacy",
    candidate_count: int = 16,
    elite_count: int = 4,
) -> SeedSelectionResult:
    """Select a deterministic FKM-derived seed with exact target content.

    ``phase_random`` chooses one independent relative phase for A and B.
    ``phase_top_q`` evaluates up to ``candidate_count`` distinct phase pairs,
    ranks them by :func:`seed_energy_key`, and uses an independent RNG stream
    to choose among the best ``elite_count`` entries.  Requesting more phase
    pairs than exist examines every pair exactly once.

    The compressed policies apply the same rank/elite rule to complete pairs.
    The sentinel phase and pool indices ``-1`` in their diagnostics indicate
    that binary parity-necklace phases were not used.

    The legacy policy intentionally ignores ``candidate_count`` and
    ``elite_count`` and performs the historical four calls to ``rng.choice``.
    This preserves an exact, low-risk rollback path.
    """
    _validate_inputs(profile, seed, pool_size, policy, candidate_count, elite_count)
    content = _profile_content(profile)

    if policy in ("compressed_top_q", "compressed_a_random_b_top_q"):
        a_pool = compressed_fkm_lift_candidates(
            profile.L, profile.a_even_ones, profile.a_odd_ones,
            seed=_domain_rng(seed, profile, "compressed-a").getrandbits(128),
            limit=max(candidate_count, elite_count),
            max_necklaces=max(pool_size, 16 * candidate_count),
        )
        if policy == "compressed_top_q":
            b_pool = compressed_fkm_lift_candidates(
                profile.L, profile.b_even_ones, profile.b_odd_ones,
                seed=_domain_rng(seed, profile, "compressed-b").getrandbits(128),
                limit=max(candidate_count, elite_count),
                max_necklaces=max(pool_size, 16 * candidate_count),
            )
        else:
            b_pool = tuple(
                _random_exact_content_word(
                    profile.L, profile.b_even_ones, profile.b_odd_ones,
                    _domain_rng(seed, profile, "random-b-{}".format(index)),
                )
                for index in range(max(candidate_count, elite_count))
            )
        if not a_pool or not b_pool:
            raise ValueError("compressed FKM produced no exact-content lifts")
        if policy == "compressed_top_q":
            available = [
                (left, right) for left in range(len(a_pool))
                for right in range(len(b_pool))
            ]
        else:
            # One independently constrained B for each queued compressed A
            # mirrors the reference initialization without a Cartesian bias.
            available = [
                (index % len(a_pool), index % len(b_pool))
                for index in range(max(len(a_pool), len(b_pool)))
            ]
        pair_rng = _domain_rng(seed, profile, "compressed-pair-order")
        pair_rng.shuffle(available)
        selected_pairs = available[:min(candidate_count, len(available))]
        candidates = tuple(sorted(
            (_score_candidate(a_pool[left], b_pool[right], -1, -1, profile)
             for left, right in selected_pairs),
            key=_candidate_order,
        ))
        effective_elites = min(elite_count, len(candidates))
        choice_rng = _domain_rng(seed, profile, "compressed-top-q-choice")
        selected_index = choice_rng.randrange(effective_elites)
        selected = candidates[selected_index]
        indices = (-1, -1, -1, -1)
        requested = candidate_count
        factors = _compression_factors(profile.L)
        diagnostics = SeedSelectionDiagnostics(
            policy=policy,
            seed=seed,
            pool_size=pool_size,
            requested_candidates=requested,
            candidates_examined=len(candidates),
            elite_count=effective_elites,
            selected_rank=selected_index,
            selected_phase_a=-1,
            selected_phase_b=-1,
            selected_energy_key=selected.energy_key,
            selected_objective=selected.objective,
            compression_factors=factors,
            target_content=content,
            pool_indices=indices,
            unique_profiles=len({candidate.profile for candidate in candidates}),
            ranked_phases=tuple((-1, -1) for _ in candidates),
            ranked_energy_keys=tuple(candidate.energy_key for candidate in candidates),
        )
        return SeedSelectionResult(selected.a, selected.b, diagnostics)

    pools = _fixed_content_pools(profile, pool_size)

    # All policies deliberately begin from the same four historical FKM pool
    # selections.  This makes a policy comparison isolate relative-phase
    # selection instead of silently changing both the necklaces and phases.
    # The SA move RNG is a separate ``random.Random(restart_seed)`` owned by
    # SearchRunner, so preserving this exact legacy draw order consumes none of
    # the later trajectory stream.
    legacy_rng = random.Random(seed)
    indices = tuple(legacy_rng.randrange(len(pool)) for pool in pools)
    words = tuple(pool[index] for pool, index in zip(pools, indices))

    if policy == "legacy":
        candidates = (_build_candidate(words, 0, 0, profile),)
        selected_index = 0
        requested = 1
        effective_elites = 1
    else:
        half = profile.L // 2
        if policy == "phase_random":
            phase_rng = _domain_rng(seed, profile, "relative-phase")
            phases = ((phase_rng.randrange(half), phase_rng.randrange(half)),)
            requested = 1
            effective_elites = 1
        else:
            phase_rng = _domain_rng(seed, profile, "phase-candidate-order")
            available = [(left, right) for left in range(half) for right in range(half)]
            phase_rng.shuffle(available)
            phases = tuple(available[:min(candidate_count, len(available))])
            requested = candidate_count
            effective_elites = min(elite_count, len(phases))
        candidates = tuple(
            _build_candidate(words, phase_a, phase_b, profile)
            for phase_a, phase_b in phases
        )
        candidates = tuple(sorted(candidates, key=_candidate_order))
        if policy == "phase_top_q":
            choice_rng = _domain_rng(seed, profile, "top-q-choice")
            selected_index = choice_rng.randrange(effective_elites)
        else:
            selected_index = 0

    selected = candidates[selected_index]
    _assert_exact_content(selected.a, selected.b, content)
    factors = _compression_factors(profile.L)
    diagnostics = SeedSelectionDiagnostics(
        policy=policy,
        seed=seed,
        pool_size=pool_size,
        requested_candidates=requested,
        candidates_examined=len(candidates),
        elite_count=effective_elites,
        selected_rank=selected_index,
        selected_phase_a=selected.phase_a,
        selected_phase_b=selected.phase_b,
        selected_energy_key=selected.energy_key,
        selected_objective=selected.objective,
        compression_factors=factors,
        target_content=content,
        pool_indices=indices,
        unique_profiles=len({candidate.profile for candidate in candidates}),
        ranked_phases=tuple((candidate.phase_a, candidate.phase_b) for candidate in candidates),
        ranked_energy_keys=tuple(candidate.energy_key for candidate in candidates),
    )
    return SeedSelectionResult(selected.a, selected.b, diagnostics)


def seed_energy_key(
    a: Sequence[int],
    b: Sequence[int],
    profile: TargetContentProfile,
) -> EnergyKey:
    """Return the exact full-energy rank followed by compression tie-breaks.

    Factor two is always available for the even Project lengths represented by
    ``TargetContentProfile``.  Factor four is appended only when it divides L.
    No arbitrary weights combine the components, so compressed energy cannot
    outrank a lower complete fixed-target energy.
    """
    observed = tuple(full_correlation_profile(a, b))
    factors = _compression_factors(profile.L)
    breakdown = structured_energy_breakdown(
        observed, profile.k, profile.eta, factors
    )
    return (breakdown.full,) + tuple(
        breakdown.component(factor) for factor in factors
    )


def _fixed_content_pools(
    profile: TargetContentProfile,
    pool_size: int,
) -> Tuple[Tuple[BinaryWord, ...], ...]:
    """Materialize the same bounded four FKM pools as the legacy initializer."""
    counts = _profile_content(profile)
    pools = tuple(
        tuple(islice(
            generate_fkm_sequences(profile.L // 2, weight=count), pool_size
        ))
        for count in counts
    )
    if any(not pool for pool in pools):
        raise ValueError("FKM produced no parity-content candidates")
    return pools


def _build_candidate(
    words: Sequence[BinaryWord],
    phase_a: int,
    phase_b: int,
    target: TargetContentProfile,
) -> _Candidate:
    """Interleave two parity pairs after changing only their relative phases."""
    a = _interleave(words[0], _rotate(words[1], phase_a))
    b = _interleave(words[2], _rotate(words[3], phase_b))
    _assert_exact_content(a, b, _profile_content(target))
    return _score_candidate(a, b, phase_a, phase_b, target)


def _score_candidate(
    a: BinaryWord,
    b: BinaryWord,
    phase_a: int,
    phase_b: int,
    target: TargetContentProfile,
) -> _Candidate:
    """Recompute and rank one complete exact-content pair."""
    _assert_exact_content(a, b, _profile_content(target))
    observed = tuple(full_correlation_profile(a, b))
    factors = _compression_factors(target.L)
    breakdown = structured_energy_breakdown(
        observed, target.k, target.eta, factors
    )
    energy_key = (breakdown.full,) + tuple(
        breakdown.component(factor) for factor in factors
    )
    return _Candidate(
        a=tuple(a),
        b=tuple(b),
        phase_a=phase_a,
        phase_b=phase_b,
        profile=observed,
        energy_key=energy_key,
        objective=pqcp_objective(observed),
    )


def _candidate_order(candidate: _Candidate) -> Tuple[object, ...]:
    """Order by exact energies, then bits and phases for deterministic ties."""
    return (
        candidate.energy_key,
        candidate.a,
        candidate.b,
        candidate.phase_a,
        candidate.phase_b,
    )


def _domain_rng(seed: int, profile: TargetContentProfile, domain: str) -> random.Random:
    """Return a stable RNG whose state is independent of every other domain."""
    fields = (
        _RNG_NAMESPACE, domain, str(seed), str(profile.L), str(profile.k),
        str(profile.eta), str(profile.a_even_ones), str(profile.a_odd_ones),
        str(profile.b_even_ones), str(profile.b_odd_ones),
    )
    digest = hashlib.sha256("|".join(fields).encode("ascii")).digest()
    return random.Random(int.from_bytes(digest, "big"))


def _interleave(even: Sequence[int], odd: Sequence[int]) -> BinaryWord:
    """Return a full sequence from equal-length even/odd subsequences."""
    if len(even) != len(odd) or not even:
        raise ValueError("even and odd subsequences must have equal positive length")
    result = [0] * (2 * len(even))
    result[0::2] = even
    result[1::2] = odd
    return tuple(result)


def _random_exact_content_word(
    length: int,
    even_ones: int,
    odd_ones: int,
    rng: random.Random,
) -> BinaryWord:
    """Sample one binary word while fixing even and odd one counts exactly."""
    even_positions = tuple(range(0, length, 2))
    odd_positions = tuple(range(1, length, 2))
    ones = set(rng.sample(even_positions, even_ones))
    ones.update(rng.sample(odd_positions, odd_ones))
    return tuple(int(index in ones) for index in range(length))


def _rotate(word: BinaryWord, offset: int) -> BinaryWord:
    """Rotate one immutable parity word without changing its content."""
    normalized = offset % len(word)
    return word[normalized:] + word[:normalized]


def _compression_factors(length: int) -> Tuple[int, ...]:
    """Return the required exact compression tie-break factors."""
    return (2, 4) if length % 4 == 0 else (2,)


def _profile_content(profile: TargetContentProfile) -> Content:
    return (
        profile.a_even_ones,
        profile.a_odd_ones,
        profile.b_even_ones,
        profile.b_odd_ones,
    )


def _assert_exact_content(a: BinaryWord, b: BinaryWord, expected: Content) -> None:
    actual = pair_content(a, b)
    if actual != expected:
        raise RuntimeError(
            "FKM seed selection changed target content: {} != {}".format(
                actual, expected
            )
        )


def _validate_inputs(
    profile: TargetContentProfile,
    seed: int,
    pool_size: int,
    policy: str,
    candidate_count: int,
    elite_count: int,
) -> None:
    if not isinstance(profile, TargetContentProfile):
        raise TypeError("profile must be a TargetContentProfile")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    if not isinstance(pool_size, int) or isinstance(pool_size, bool) or pool_size <= 0:
        raise ValueError("pool_size must be a positive integer")
    if policy not in SEED_POLICIES:
        raise ValueError("policy must be one of {}".format(", ".join(SEED_POLICIES)))
    for name, value in (
        ("candidate_count", candidate_count), ("elite_count", elite_count),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError("{} must be a positive integer".format(name))
    if policy in (
        "phase_top_q", "compressed_top_q", "compressed_a_random_b_top_q",
    ) and elite_count > candidate_count:
        raise ValueError("elite_count must not exceed candidate_count")
    # Reject a malformed target before potentially expensive FKM generation.
    # ``structured_energy_breakdown`` independently validates the same target
    # again when candidates are scored.
    half = profile.L // 2 if isinstance(profile.L, int) else -1
    if (
        not isinstance(profile.L, int) or isinstance(profile.L, bool)
        or profile.L < 4 or profile.L % 2
        or not isinstance(profile.k, int) or isinstance(profile.k, bool)
        or not 1 <= profile.k < profile.L // 2
        or profile.eta not in (-1, 1) or isinstance(profile.eta, bool)
    ):
        raise ValueError("profile has an invalid Project target")
    counts = _profile_content(profile)
    if any(
        not isinstance(count, int) or isinstance(count, bool)
        or not 0 <= count <= half
        for count in counts
    ):
        raise ValueError("profile content counts must lie between zero and L/2")
