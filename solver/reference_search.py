"""Auditable search preset adapted from the faster reference PQCP project.

Only methods compatible with this project's independently verified target are
represented here.  The preset combines:

* exact ordinary-sum and alternating-sum target-content profiles;
* genuine factor-two compressed FKM seeds, lifted at ``i`` and ``i+L/2``;
* same-parity zero/one exchanges, preserving all four content counts;
* exact full PACF error plus factor-two/factor-four folded error;
* two sampled legal exchanges per move;
* a restart-local linear temperature schedule and bounded stagnation kicks.

The compressed energy, proposal sampling, temperature schedule, and kick are
navigation heuristics.  They never prune candidates and never replace the
original objective or independent verifier.  :func:`legacy_search_parameters`
is an explicit rollback configuration for controlled comparisons.
"""

from .checkpoint import SearchParameters
from .target_profiles import canonical_target_content_profiles


def encoded_target_profiles(length: int):
    """Return every canonical exact target-content profile in JSON-safe order."""
    profiles = canonical_target_content_profiles(length)
    if not profiles:
        raise ValueError(
            "no target-content profile satisfies the current necessary conditions "
            "for L={}".format(length)
        )
    return tuple(
        (
            profile.k,
            profile.eta,
            profile.a_even_ones,
            profile.a_odd_ones,
            profile.b_even_ones,
            profile.b_odd_ones,
        )
        for profile in profiles
    )


def reference_search_parameters(
    length: int,
    target_profile_offset: int = 0,
) -> SearchParameters:
    """Build the conservative Python port of the reference search controls.

    The reference C implementation uses raw squared energy and a Metropolis
    denominator ``64*T``.  This project stores the same exact energy divided
    by 16, so ``metropolis_scale=4`` preserves that acceptance ratio exactly.
    """
    return SearchParameters(
        initial_temperature=6.15,
        min_temperature=0.15,
        temperature_schedule="linear_restart",
        metropolis_scale=4.0,
        fkm_seed_policy="compressed_a_random_b_top_q",
        fkm_candidate_count=16,
        fkm_elite_count=4,
        stagnation_iterations=None,
        kick_stagnation_iterations=1_000,
        kick_swaps=3,
        max_iterations_per_restart=20_000 + 1_000 * length,
        acceptance_mode="fixed_target_multiscale",
        proposal_samples=2,
        target_content_profiles=encoded_target_profiles(length),
        preserve_alternating_content=True,
        target_profile_offset=target_profile_offset,
    )


def legacy_search_parameters(length: int) -> SearchParameters:
    """Return the previous fixed-target production controls for rollback."""
    return SearchParameters(
        initial_temperature=8.0,
        min_temperature=0.05,
        cooling_rate=0.999,
        fkm_seed_policy="legacy",
        stagnation_iterations=100_000,
        acceptance_mode="fixed_target_full",
        proposal_samples=10,
        target_content_profiles=encoded_target_profiles(length),
        preserve_alternating_content=True,
    )


__all__ = (
    "encoded_target_profiles", "legacy_search_parameters",
    "reference_search_parameters",
)
