"""Independent, search-free verification for Project 2 PQCP candidates."""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

from .correlation import full_correlation_profile, normalize_binary_sequence
from .problem import PQCPInstance


@dataclass(frozen=True)
class VerificationResult:
    """The recomputed profile and all reasons a candidate did not verify."""

    is_valid: bool
    errors: Tuple[str, ...]
    profile: Tuple[int, ...]
    nonzero_shifts: Tuple[int, ...]


def verify_pqcp(
    a: Sequence[Union[int, str]],
    b: Sequence[Union[int, str]],
    instance: Optional[PQCPInstance] = None,
) -> VerificationResult:
    """Recompute and verify whether ``a, b`` satisfy the Project 2 PQCP rules.

    If an ``instance`` is provided, its ``L`` is also checked.  Otherwise L
    is the common input length.  This verifier only uses the mathematical
    correlation routines; it has no dependency on search techniques or Z3.
    """
    errors = []
    try:
        a_bits = normalize_binary_sequence(a)
    except ValueError as error:
        a_bits = None
        errors.append("A is not a binary sequence: {}".format(error))
    try:
        b_bits = normalize_binary_sequence(b)
    except ValueError as error:
        b_bits = None
        errors.append("B is not a binary sequence: {}".format(error))

    if a_bits is None or b_bits is None:
        return VerificationResult(False, tuple(errors), (), ())

    if len(a_bits) != len(b_bits):
        errors.append("A and B have unequal lengths")
        return VerificationResult(False, tuple(errors), (), ())

    length = len(a_bits)
    if instance is not None and length != instance.L:
        errors.append("sequence length {} does not match instance L={}".format(length, instance.L))

    profile = tuple(full_correlation_profile(a_bits, b_bits))
    nonzero_shifts = tuple(shift for shift in range(1, length) if profile[shift] != 0)

    if profile[0] != 2 * length:
        errors.append("S[0] must equal 2L")
    if len(nonzero_shifts) != 2:
        errors.append("exactly two nonzero sums are required for shifts u != 0")
    if any(abs(profile[shift]) != 4 for shift in nonzero_shifts):
        errors.append("every nonzero sum for u != 0 must have magnitude 4")
    if any(profile[shift] != profile[(-shift) % length] for shift in range(length)):
        errors.append("periodic correlation symmetry S[u] = S[L-u] is violated")

    return VerificationResult(not errors, tuple(errors), profile, nonzero_shifts)
