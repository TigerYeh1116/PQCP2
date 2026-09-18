"""Exact factor-2 lift completion, not a Hamming-ball repair.

For L=2d, c[i]=x[i]+x[i+d]. Nonzero c entries fix both signs;
each zero has one orientation bit, placed at i and i+d (NOT adjacent
positions). The compressed PACF pair must equal fold_2(T). Since T(d)=0,
the two words then have exactly d zero entries in total: rho_x(d)=L-4z_x.
Z3 searches every orientation of this *fixed* compressed pair. UNSAT is
local to that pair and optional content profile; UNKNOWN proves nothing.
"""

from dataclasses import dataclass
from time import perf_counter
from typing import Optional, Sequence, Tuple

from .structured_energy import (
    fold_profile, integer_periodic_autocorrelation_profile, target_profile,
)
from .target_profiles import TargetContentProfile, pair_content
from .verifier import verify_pqcp


@dataclass(frozen=True)
class CompressedLiftResult:
    """An auditable exact result; timings distinguish construction and check."""

    status: str
    L: int
    k: int
    eta: int
    orientation_variables: int
    correlation_constraints: int
    content_constraints: int
    construction_time: float
    solve_time: float
    z3_executed: bool
    a: Optional[Tuple[int, ...]] = None
    b: Optional[Tuple[int, ...]] = None
    profile: Optional[Tuple[int, ...]] = None
    verified: bool = False
    reason: Optional[str] = None
    cross_pattern_count: int = 0
    preferred_values: int = 0


def _compressed_word(values: Sequence[int]) -> Tuple[int, ...]:
    word = tuple(values)
    if len(word) < 2 or any(
        not isinstance(x, int) or isinstance(x, bool) or x not in (-2, 0, 2)
        for x in word
    ):
        raise ValueError("factor-2 words must have length >= 2 and entries -2, 0, 2")
    return word


def _validate_content(content: TargetContentProfile, L: int, k: int, eta: int) -> None:
    if not isinstance(content, TargetContentProfile) or (
        content.L, content.k, content.eta
    ) != (L, k, eta):
        raise ValueError("content profile must match L, k and eta")
    counts = (content.a_even_ones, content.a_odd_ones,
              content.b_even_ones, content.b_odd_ones)
    if any(not isinstance(x, int) or isinstance(x, bool) or not 0 <= x <= L // 2
           for x in counts):
        raise ValueError("invalid parity content counts")


def solve_compressed_lift(
    compressed_a: Sequence[int], compressed_b: Sequence[int],
    k: int, eta: int, timeout_ms: Optional[int] = 5_000,
    content_profile: Optional[TargetContentProfile] = None,
    cross_compressions: Optional[Sequence[Tuple[Tuple[int, ...], Tuple[int, ...]]]] = None,
    preferred_pair: Optional[Tuple[Sequence[int], Sequence[int]]] = None,
) -> CompressedLiftResult:
    """Solve all binary lifts of one exactly matched factor-2 pair.

    Inputs are compressed BIPOLAR sums; outputs are Project bits (0=+1).
    k is an actual representative shift 1..L/2-1 and eta is +/-1. There is
    no center or radius constraint. All independent full-length shifts are
    enforced exactly with integer pseudo-Boolean mismatch counts. At L=2d,
    S(u)=2L-2*mismatch_count(u); the desired count is L-T(u)/2.

    An optional content profile restricts even/odd one counts. This is needed
    when d is odd, since zero orientations can change alternating sums. With
    d even these counts are already fixed by c, but we still check them.
    SAT is independently verified and checked against this exact target.
    A SAT query returns one model, not an enumeration of every distinct lift.
    timeout_ms applies to Z3.check(), not Python model construction; None
    or zero means no solver timeout, matching the existing Z3 convention.
    Optional cross_compressions is an exact disjunction of factor-(L/4)
    patterns. It restricts this subproblem only; the multilevel caller must
    supply ALL compatible patterns to preserve its completeness claim.
    preferred_pair supplies initial Boolean preferences only, never equality
    or Hamming constraints. A preference can violate the fixed compression;
    the exact equations still take precedence. It need not be a solution.
    """
    started = perf_counter()
    ca, cb = _compressed_word(compressed_a), _compressed_word(compressed_b)
    if len(ca) != len(cb):
        raise ValueError("compressed words must have equal lengths")
    L, d = 2 * len(ca), len(ca)
    if not isinstance(k, int) or isinstance(k, bool) or not 0 < k < d:
        raise ValueError("k must be an integer with 0 < k < L/2")
    if not isinstance(eta, int) or isinstance(eta, bool) or eta not in (-1, 1):
        raise ValueError("eta must be -1 or +1")
    if timeout_ms is not None and (not isinstance(timeout_ms, int)
            or isinstance(timeout_ms, bool) or timeout_ms < 0):
        raise ValueError("timeout_ms must be a nonnegative integer or None")
    if content_profile is not None:
        _validate_content(content_profile, L, k, eta)
    if cross_compressions is not None:
        from .cross_compression import normalize_cross_pairs
        cross_compressions = normalize_cross_pairs(cross_compressions, L, k, eta)
    if preferred_pair is not None:
        from .correlation import normalize_binary_sequence
        if len(preferred_pair) != 2:
            raise ValueError("preferred_pair must contain A and B")
        preferred_pair = tuple(normalize_binary_sequence(w) for w in preferred_pair)
        if any(len(w) != L for w in preferred_pair):
            raise ValueError("preferred_pair length mismatch")
    orientations = ca.count(0) + cb.count(0)
    desired = target_profile(L, k, eta)
    compressed_pair = tuple(x + y for x, y in zip(
        integer_periodic_autocorrelation_profile(ca),
        integer_periodic_autocorrelation_profile(cb),
    ))
    if compressed_pair != fold_profile(desired, 2):
        return CompressedLiftResult(
            "UNSAT", L, k, eta, orientations, 0, 0,
            perf_counter() - started, 0.0, False,
            reason="fixed compressed PACF pair does not equal fold_2(T)",
        )
    if orientations != d:  # an implementation invariant, not a heuristic
        raise RuntimeError("matched compression violates the half-shift identity")

    import z3  # optional dependency; do not load it for non-Z3 searches

    def lift_bits(word, name):
        bits = [None] * L
        for i, value in enumerate(word):
            if value == 0:
                first = z3.Bool("lift_{}_{}".format(name, i))
                bits[i], bits[i + d] = first, z3.Not(first)
            else:
                bits[i] = bits[i + d] = z3.BoolVal(value == -2)
        return bits

    def count_equals(expressions, wanted):
        # Remove constants and combine identical terms before PB encoding.
        # This preserves multiplicity, including repeated cyclic edges.
        fixed, weighted = 0, {}
        for expression in expressions:
            expression = z3.simplify(expression)
            if z3.is_true(expression):
                fixed += 1
            elif not z3.is_false(expression):
                key = expression.get_id()
                old = weighted.get(key)
                weighted[key] = (expression, 1 if old is None else old[1] + 1)
        wanted -= fixed
        terms = list(weighted.values())
        if wanted < 0 or wanted > sum(weight for _, weight in terms):
            return z3.BoolVal(False)
        return z3.PbEq(terms, wanted) if terms else z3.BoolVal(wanted == 0)

    a_bits, b_bits = lift_bits(ca, "a"), lift_bits(cb, "b")
    solver = z3.SolverFor("QF_FD")
    if timeout_ms is not None:
        solver.set(timeout=timeout_ms)
    for shift in range(1, d):
        mismatches = [z3.Xor(bits[i], bits[(i + shift) % L])
                      for bits in (a_bits, b_bits) for i in range(L)]
        solver.add(count_equals(mismatches, L - desired[shift] // 2))
    # S(d)=0 and S(0)=2L are guaranteed by the compressed match and binary
    # construction. S(L-u)=S(u), so enforcing 1..d-1 covers all other shifts.
    content_constraints = 0
    if content_profile is not None:
        for bits, even, odd in (
            (a_bits, content_profile.a_even_ones, content_profile.a_odd_ones),
            (b_bits, content_profile.b_even_ones, content_profile.b_odd_ones),
        ):
            solver.add(count_equals(bits[0::2], even))
            solver.add(count_equals(bits[1::2], odd))
        content_constraints = 4
    if cross_compressions is not None:
        # Each residue mod 4 contains n=L/4 bits. Its sign sum r means
        # exactly (n-r)/2 ones. Enumerated patterns remain coupled A/B pairs.
        clauses = []
        for cross_a, cross_b in cross_compressions:
            clauses.append(z3.And([
                count_equals(bits[r::4], (L//4 - pattern[r])//2)
                for bits, pattern in ((a_bits,cross_a),(b_bits,cross_b)) for r in range(4)]))
        solver.add(z3.Or(clauses))
    preferred_values = 0
    if preferred_pair is not None and hasattr(solver, "set_initial_value"):
        for bits, pattern, preferred in zip((a_bits,b_bits),(ca,cb),preferred_pair):
            for i,value in enumerate(pattern):
                if value == 0:
                    solver.set_initial_value(bits[i], bool(preferred[i]))
                    preferred_values += 1
    construction_time = perf_counter() - started
    solve_started = perf_counter()
    outcome = solver.check()
    solve_time = perf_counter() - solve_started
    common = dict(L=L, k=k, eta=eta, orientation_variables=orientations,
                  correlation_constraints=d - 1, content_constraints=content_constraints,
                  construction_time=construction_time, solve_time=solve_time,
                  z3_executed=True, cross_pattern_count=len(cross_compressions or ()),
                  preferred_values=preferred_values)
    if outcome == z3.unsat:
        return CompressedLiftResult(status="UNSAT", **common)
    if outcome != z3.sat:
        return CompressedLiftResult(status="UNKNOWN", reason=solver.reason_unknown(), **common)
    model = solver.model()
    a, b = (tuple(int(z3.is_true(model.evaluate(bit, model_completion=True)))
                  for bit in bits) for bits in (a_bits, b_bits))
    verification = verify_pqcp(a, b)
    if not verification.is_valid or verification.profile != desired:
        raise RuntimeError("compressed Z3 lift SAT failed independent exact-target verification")
    if content_profile is not None and pair_content(a, b) != (
        content_profile.a_even_ones, content_profile.a_odd_ones,
        content_profile.b_even_ones, content_profile.b_odd_ones,
    ):
        raise RuntimeError("compressed Z3 lift SAT changed the requested content")
    if cross_compressions is not None:
        from .structured_energy import compress_binary_sequence
        cross_pair = tuple(tuple(1-2*x for x in bits) if L == 4
                           else compress_binary_sequence(bits,L//4) for bits in (a,b))
        if cross_pair not in cross_compressions:
            raise RuntimeError("compressed Z3 lift SAT changed the cross-compression constraints")
    return CompressedLiftResult(status="SAT", a=a, b=b,
                                profile=verification.profile, verified=True, **common)
