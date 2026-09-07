"""Exact Z3 completion in a bounded Hamming neighborhood of an SA candidate.

This module deliberately requires a complete center pair and a finite Hamming
radius; it does not expose a blank full-space PQCP search.  Shifts u and L-u
are represented by one symbolic correlation because periodic symmetry makes
them equal, while their contribution to the Project 2 *actual shift* count is
still two (except the self-symmetric shift L/2 for even L).
"""

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Optional, Sequence, Tuple, Union

import z3

from .correlation import normalize_binary_sequence
from .verifier import verify_pqcp
from .target_profiles import TargetContentProfile, pair_content, target_content_profiles
from .weight_constraints import admissible_weight_pairs


BinaryInput = Sequence[Union[int, str]]


@dataclass(frozen=True)
class Z3CompletionResult:
    """Classification and optional independently verified Z3 completion."""

    status: str
    L: int
    radius: int
    elapsed_time: float
    a: Optional[Tuple[int, ...]]
    b: Optional[Tuple[int, ...]]
    profile: Optional[Tuple[int, ...]]
    verified: bool
    reason: Optional[str]


def solve_with_z3(
    L: int,
    initial_a: BinaryInput,
    initial_b: BinaryInput,
    radius: int = 3,
    timeout_ms: Optional[int] = 5_000,
    use_target_content_profiles: bool = True,
    allowed_target_content_profiles: Optional[Sequence[TargetContentProfile]] = None,
) -> Z3CompletionResult:
    """Find an exact PQCP within Hamming distance ``radius`` of a center pair.

    Radius is the combined distance
    ``distance(A, initial_a) + distance(B, initial_b) <= radius``.  Thus
    radius zero is exact verification of the supplied center, while positive
    radii perform progressive neighborhood completion with symbolic bits.
    ``UNKNOWN`` is returned unchanged (including timeout reasons); it is never
    interpreted as ``UNSAT``.
    """
    center_a, center_b = _validate_inputs(L, initial_a, initial_b, radius, timeout_ms)
    started = perf_counter()
    solver = _make_solver()
    if timeout_ms is not None:
        solver.set(timeout=timeout_ms)

    # A symbolic variable means "flip this center bit".  Bool variables need
    # no separate binary-domain constraints and the Hamming bound is exactly
    # the number of true flip variables.
    a_flips = [z3.Bool("flip_a_{}".format(index)) for index in range(L)]
    b_flips = [z3.Bool("flip_b_{}".format(index)) for index in range(L)]
    effective_radius = radius
    reachable_weight_pairs = None
    if L % 2 == 0:
        reachable_weight_pairs = _reachable_admissible_weight_pairs(
            L, center_a, center_b, radius
        )
        parities = {
            (target_a - sum(center_a) + target_b - sum(center_b)) % 2
            for target_a, target_b in reachable_weight_pairs
        }
        # If every reachable legal weight pair requires the same flip-count
        # parity, an odd outer radius may contain no additional legal point.
        # Tightening 3 to 2 in that case is exact, not heuristic pruning.
        if len(parities) == 1:
            parity = next(iter(parities))
            if effective_radius % 2 != parity:
                effective_radius -= 1
    solver.add(z3.PbLe(
        [(variable, 1) for variable in a_flips + b_flips],
        max(0, effective_radius),
    ))

    # For a pair of new bits, mismatch is center_mismatch XOR flip_i XOR
    # flip_j.  If m(u) counts all A/B mismatches at shift u, then the exact
    # pair correlation is S(u)=2L-2m(u).  Hence S(u) in {0,-4,+4} iff
    # m(u) is in {L,L+2,L-2}; no integer correlation expression is needed.
    nonzero_representatives = []
    target_value_conditions = {}
    for shift in range(1, L // 2 + 1):
        mismatches = (
            _shift_mismatches(center_a, a_flips, shift)
            + _shift_mismatches(center_b, b_flips, shift)
        )
        if L <= 20:
            # Native PB cardinalities solve small bounded neighborhoods faster.
            zero = _pb_equal(mismatches, L)
            plus_four = _pb_equal(mismatches, L - 2)
            minus_four = _pb_equal(mismatches, L + 2)
        else:
            # For project lengths, share one sum AST between all three target
            # values; this avoids tripling construction cost before timeout.
            mismatch_count = z3.Sum([z3.If(mismatch, 1, 0) for mismatch in mismatches])
            zero = mismatch_count == L
            plus_four = mismatch_count == L - 2
            minus_four = mismatch_count == L + 2
        if L % 2 == 0 and shift == L // 2:
            # The self-symmetric shift represents only one actual shift.  It
            # cannot participate when exactly two nonzero actual shifts are
            # required, so it is necessarily zero.
            solver.add(zero)
        else:
            solver.add(z3.Or(zero, plus_four, minus_four))
            nonzero_representatives.append(z3.Or(plus_four, minus_four))
            target_value_conditions[shift] = (minus_four, plus_four)

    # Every non-half representative accounts for the actual pair u,L-u, so
    # exactly one representative must be nonzero.
    if nonzero_representatives:
        solver.add(z3.PbEq([(indicator, 1) for indicator in nonzero_representatives], 1))
    else:
        solver.add(z3.BoolVal(False))

    # The total-autocorrelation identity is a necessary condition for every
    # even-length solution.  Use all ordered weight pairs here (not canonical
    # representatives), because a bounded neighborhood is not invariant under
    # complementing a sequence while leaving its center fixed.
    if L % 2 == 0:
        weight_pairs = reachable_weight_pairs or ()
        weight_a_bits = [_new_bit(center, flip) for center, flip in zip(center_a, a_flips)]
        weight_b_bits = [_new_bit(center, flip) for center, flip in zip(center_b, b_flips)]
        if weight_pairs:
            if L <= 20:
                solver.add(z3.Or(*(
                    z3.And(_pb_equal(weight_a_bits, pair[0]), _pb_equal(weight_b_bits, pair[1]))
                    for pair in weight_pairs
                )))
            else:
                weight_a = z3.Sum([z3.If(bit, 1, 0) for bit in weight_a_bits])
                weight_b = z3.Sum([z3.If(bit, 1, 0) for bit in weight_b_bits])
                solver.add(z3.Or(*(
                    z3.And(weight_a == pair[0], weight_b == pair[1])
                    for pair in weight_pairs
                )))
        else:
            solver.add(z3.BoolVal(False))

        if use_target_content_profiles and L >= 4:
            reachable_profiles = _reachable_target_content_profiles(
                L, center_a, center_b, radius, allowed_target_content_profiles
            )
            if reachable_profiles:
                a_even = [weight_a_bits[index] for index in range(0, L, 2)]
                a_odd = [weight_a_bits[index] for index in range(1, L, 2)]
                b_even = [weight_b_bits[index] for index in range(0, L, 2)]
                b_odd = [weight_b_bits[index] for index in range(1, L, 2)]
                if L <= 20:
                    content_conditions = {
                        profile: z3.And(
                            _pb_equal(a_even, profile.a_even_ones),
                            _pb_equal(a_odd, profile.a_odd_ones),
                            _pb_equal(b_even, profile.b_even_ones),
                            _pb_equal(b_odd, profile.b_odd_ones),
                        )
                        for profile in reachable_profiles
                    }
                else:
                    content_sums = tuple(
                        z3.Sum([z3.If(bit, 1, 0) for bit in bits])
                        for bits in (a_even, a_odd, b_even, b_odd)
                    )
                    content_conditions = {
                        profile: z3.And(
                            content_sums[0] == profile.a_even_ones,
                            content_sums[1] == profile.a_odd_ones,
                            content_sums[2] == profile.b_even_ones,
                            content_sums[3] == profile.b_odd_ones,
                        )
                        for profile in reachable_profiles
                    }
                solver.add(z3.Or(*(
                    z3.And(
                        content_conditions[profile],
                        target_value_conditions[profile.k][0 if profile.eta == -1 else 1],
                    )
                    for profile in reachable_profiles
                )))
            else:
                solver.add(z3.BoolVal(False))

    outcome = solver.check()
    elapsed = perf_counter() - started
    if outcome == z3.sat:
        model = solver.model()
        solution_a = tuple(
            bit ^ int(z3.is_true(model.evaluate(flip, model_completion=True)))
            for bit, flip in zip(center_a, a_flips)
        )
        solution_b = tuple(
            bit ^ int(z3.is_true(model.evaluate(flip, model_completion=True)))
            for bit, flip in zip(center_b, b_flips)
        )
        verification = verify_pqcp(solution_a, solution_b)
        return Z3CompletionResult(
            status="SAT",
            L=L,
            radius=radius,
            elapsed_time=elapsed,
            a=solution_a,
            b=solution_b,
            profile=verification.profile,
            verified=verification.is_valid,
            reason=None,
        )
    if outcome == z3.unsat:
        return Z3CompletionResult("UNSAT", L, radius, elapsed, None, None, None, False, None)
    return Z3CompletionResult(
        "UNKNOWN", L, radius, elapsed, None, None, None, False, solver.reason_unknown()
    )


def save_verified_solution(result: Z3CompletionResult, directory: Union[str, Path] = "results") -> Path:
    """Write a verified SAT result as ``results/L.txt`` without overwriting files."""
    if result.status != "SAT" or not result.verified or result.a is None or result.b is None or result.profile is None:
        raise ValueError("only an independently verified SAT result can be saved")
    destination_directory = Path(directory)
    destination_directory.mkdir(parents=True, exist_ok=True)
    destination = destination_directory / "{}.txt".format(result.L)
    if destination.exists():
        raise FileExistsError("refusing to overwrite existing result: {}".format(destination))
    nonzero_shifts = tuple(shift for shift in range(1, result.L) if result.profile[shift] != 0)
    nonzero_values = tuple(result.profile[shift] for shift in nonzero_shifts)
    destination.write_text(
        "L={}\n"
        "nonzero shifts={}\n"
        "nonzero PACS={}\n"
        "a={}\n"
        "b={}\n"
        "verification status=verified\n".format(
            result.L,
            ",".join(str(shift) for shift in nonzero_shifts),
            ",".join(str(value) for value in nonzero_values),
            "".join(str(bit) for bit in result.a),
            "".join(str(bit) for bit in result.b),
        ),
        encoding="utf-8",
    )
    return destination


def _pair_correlation(a_vars, b_vars, shift: int, length: int):
    """Build an exact Z3 expression for rho(A;u) + rho(B;u)."""
    terms = []
    for index in range(length):
        terms.append(z3.If(a_vars[index] == a_vars[(index + shift) % length], 1, -1))
        terms.append(z3.If(b_vars[index] == b_vars[(index + shift) % length], 1, -1))
    return z3.Sum(terms)


def _shift_mismatches(center: Sequence[int], flips, shift: int):
    """Return exact Bool mismatch expressions after center-relative flips."""
    length = len(center)
    return [
        flips[index] != flips[(index + shift) % length]
        if center[index] == center[(index + shift) % length]
        else flips[index] == flips[(index + shift) % length]
        for index in range(length)
    ]


def _new_bit(center_bit: int, flip):
    """Return the Bool value of one center bit after its symbolic flip."""
    return flip if center_bit == 0 else z3.Not(flip)


def _pb_equal(booleans, target: int):
    """Return a pseudo-Boolean equality, or false for an impossible target."""
    if not 0 <= target <= len(booleans):
        return z3.BoolVal(False)
    return z3.PbEq([(boolean, 1) for boolean in booleans], target)


def _make_solver():
    """Return Z3's finite-domain solver for the Bool/PB completion model."""
    return z3.SolverFor("QF_FD")


def _reachable_admissible_weight_pairs(
    L: int,
    center_a: Sequence[int],
    center_b: Sequence[int],
    radius: int,
) -> Tuple[Tuple[int, int], ...]:
    """Return exactly the legal weight pairs reachable inside a Hamming ball.

    Changing a sequence weight from ``w`` to ``t`` needs at least ``|t-w|``
    bit flips.  Therefore a pair whose two absolute differences exceed the
    combined radius cannot occur in the bounded completion.  Removing it is
    a mathematically safe reduction of the existing weight disjunction.
    """
    weight_a, weight_b = sum(center_a), sum(center_b)
    return tuple(
        pair
        for pair in admissible_weight_pairs(L)
        if abs(pair[0] - weight_a) + abs(pair[1] - weight_b) <= radius
    )


def _reachable_target_content_profiles(
    L: int,
    center_a: Sequence[int],
    center_b: Sequence[int],
    radius: int,
    allowed_profiles: Optional[Sequence[TargetContentProfile]] = None,
):
    """Return exact target/content cases whose parity counts are reachable.

    Reaching a requested number of ones in one parity class needs at least
    the absolute count difference in that class.  Summing the four disjoint
    classes is therefore an exact lower bound on combined Hamming distance.
    The retained profile also binds its target shift and sign, preventing the
    ordinary- and alternating-character identities from choosing mutually
    incompatible cases.
    """
    current = pair_content(center_a, center_b)
    universe = (
        tuple(allowed_profiles) if allowed_profiles is not None
        else target_content_profiles(L)
    )
    if any(profile.L != L for profile in universe):
        raise ValueError("allowed target content profile has the wrong length")
    return tuple(
        profile for profile in universe
        if sum(abs(left - right) for left, right in zip(
            current,
            (profile.a_even_ones, profile.a_odd_ones,
             profile.b_even_ones, profile.b_odd_ones),
        )) <= radius
    )


def _validate_inputs(
    L: int,
    initial_a: BinaryInput,
    initial_b: BinaryInput,
    radius: int,
    timeout_ms: Optional[int],
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Validate the bounded-neighborhood API and normalize binary centers."""
    if not isinstance(L, int) or isinstance(L, bool) or L <= 0:
        raise ValueError("L must be a positive integer")
    if not isinstance(radius, int) or isinstance(radius, bool) or not 0 <= radius <= 2 * L:
        raise ValueError("radius must be an integer in the range 0 through 2L")
    if timeout_ms is not None and (not isinstance(timeout_ms, int) or isinstance(timeout_ms, bool) or timeout_ms < 0):
        raise ValueError("timeout_ms must be a non-negative integer or None")
    a_bits = normalize_binary_sequence(initial_a)
    b_bits = normalize_binary_sequence(initial_b)
    if len(a_bits) != L or len(b_bits) != L:
        raise ValueError("initial_a and initial_b must both have length L")
    return a_bits, b_bits
