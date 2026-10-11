"""Search-target orbits under simultaneous alternating negation.

For even L, x'_i=(-1)^i x_i gives T'(u)=(-1)^u T(u). Thus odd-k
positive/negative PQCP targets are equivalent for new-class discovery.
Only eliminate a target when its counterpart is present in the supplied
canonical content pool; single prescribed target/sign requests stay intact.
"""
from functools import lru_cache

from .target_profiles import TargetContentProfile


@lru_cache(maxsize=256)
def canonical_counts(length, counts):
    half = length//2
    if (type(length) is not int or length < 4 or length % 2 or len(counts) != 4
            or any(type(v) is not int or not 0 <= v <= half for v in counts)):
        raise ValueError('valid even length and four parity counts required')
    variants = []
    ae, ao, be, bo = counts
    for ca in (False, True):
        aa = (half-ae, half-ao) if ca else (ae, ao)
        for cb in (False, True):
            bb = (half-be, half-bo) if cb else (be, bo)
            for ra in (False, True):
                a = aa[::-1] if ra else aa
                for rb in (False, True):
                    b = bb[::-1] if rb else bb
                    variants.extend((a+b, b+a))
    return min(variants)


def alternate_canonical_profile(profile):
    if not isinstance(profile, TargetContentProfile):
        raise ValueError('canonical target profile required')
    if (type(profile.k) is not int or not 1 <= profile.k < profile.L//2
            or type(profile.eta) is not int or profile.eta not in (-1, 1)):
        raise ValueError('non-midpoint Project shift and target sign required')
    counts = (profile.a_even_ones, profile.a_odd_ones, profile.b_even_ones, profile.b_odd_ones)
    if canonical_counts(profile.L, counts) != counts:
        raise ValueError('content gauge must be canonical before alternation reduction')
    half = profile.L//2
    new_counts = canonical_counts(profile.L, (counts[0], half-counts[1], counts[2], half-counts[3]))
    return TargetContentProfile(profile.L, profile.k, profile.eta*(-1 if profile.k % 2 else 1), *new_counts)


def reduce_alternation_profile_orbits(profiles):
    profiles = tuple(sorted(set(profiles)))
    supplied = set(profiles)
    representatives = set()
    mapping = {}
    for profile in profiles:
        partner = alternate_canonical_profile(profile)
        if alternate_canonical_profile(partner) != profile:
            raise RuntimeError('alternation on canonical contents is not an involution')
        representative = min(profile, partner) if partner in supplied else profile
        representatives.add(representative)
        mapping[profile] = representative
    return tuple(sorted(representatives)), mapping
