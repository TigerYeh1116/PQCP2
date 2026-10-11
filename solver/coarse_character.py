"""Exact character restrictions before constructing fixed-C_A norm profiles.

For CRT rows C=(A_0+A_1)/2 and G=(A_0-A_1)/2, supports are
complementary. Summing the two target norm equations gives two integer
sum-of-squares equations. These are necessary conditions, not solutions.
"""
from functools import lru_cache
from math import isqrt

from .coarse_row_completion import validate_coarse_row
from .full_row_completion import signed_norm_psd_obstruction
from .structured_energy import integer_periodic_autocorrelation_profile as pacf


@lru_cache(maxsize=8192)
def _characters(d, weight, sigma):
    other_weight = d-weight
    allowed = []
    for eta in (-1, 1):
        square = d+2*eta-sigma*sigma
        if square < 0:
            continue
        b = isqrt(square)
        if b*b != square or b > other_weight or (other_weight-b) % 2:
            continue
        for parity in (0, 1):
            total = d+2*eta*(-1 if parity else 1)
            # G_A has other_weight nonzero entries; G_B has weight.
            for a in range(other_weight % 2, min(other_weight, isqrt(total))+1, 2):
                rest = total-a*a
                b = isqrt(rest)
                if b*b == rest and b <= weight and (weight-b) % 2 == 0:
                    allowed.append((eta, parity))
                    break
    return tuple(allowed)


def admissible_coarse_characters(row):
    """Return every (eta, original k mod 2) passing scalar restrictions.

    Empty means all strict targets are impossible at this literal point.
    Nonempty provides no assertion about any nonzero-shift equation.
    Independent global signs justify using nonnegative unknown row sums.
    """
    row = validate_coarse_row(row)
    return _characters(len(row), sum(v != 0 for v in row), sum(row))


def coarse_all_target_psd_obstruction(row):
    """Certify PSD(C_A,f)>d+2, impossible for EVERY strict k/eta.

    PSD(C_A,f)+PSD(C_B,f)=d+2*eta*cos(2*pi*tau*f/d)<=d+2.
    The synthetic symmetric vector below has Fourier value d+2-PSD(C_A).
    Existing outward integer intervals certify strict negativity; a failed
    certificate simply retains the point. DC is handled by characters.
    """
    proof = _all_target_psd(validate_coarse_row(row))
    return dict(proof) if proof is not None else None


@lru_cache(maxsize=64)
def _all_target_psd(row):
    profile = tuple(pacf(row))
    d, weight = len(row), profile[0]
    if weight*weight <= d+2:
        return None  # Triangle inequality already precludes the violation.
    envelope = (d+2-weight,)+tuple(-v for v in profile[1:])
    proof = signed_norm_psd_obstruction(envelope)
    if proof is None:
        return None
    return dict(proof, reason='COARSE_PSD_EXCEEDS_ALL_TARGET_ENVELOPE',
                maximum_pair_psd=d+2, coarse_profile=profile)
