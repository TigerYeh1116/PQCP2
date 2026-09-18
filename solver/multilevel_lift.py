"""Exact factor-4 -> factor-2 -> binary completion of a fixed PQCP target.

For L=4n, c4[i]=c2[i]+c2[i+n]; c2 entries are -2,0,2.
Enumerate every factor-2 pair satisfying PAF(c2a)+PAF(c2b)=fold_2(T),
then use the existing exact binary lift. fold_4(T) is only a necessary
condition. Intermediate values are placed at i and i+n, never adjacent.

A model/time cap is a search budget, NOT a proof of impossibility. UNSAT
requires exhaustion of all middle models with every binary child UNSAT.
An UNKNOWN child remains unresolved even after its middle model is blocked
to try another branch. No known solutions are used to initialize this search.
"""

from dataclasses import dataclass
from math import ceil
from time import perf_counter
from typing import Optional, Sequence, Tuple

from .compressed_lift import CompressedLiftResult, _validate_content, solve_compressed_lift
from .structured_energy import fold_profile, integer_periodic_autocorrelation_profile, target_profile
from .target_profiles import TargetContentProfile


@dataclass(frozen=True)
class MultilevelLiftResult(CompressedLiftResult):
    """Aggregate timings/check counts, including unresolved middle branches."""

    middle_models: int = 0
    middle_checks: int = 0
    binary_checks: int = 0
    unknown_children: int = 0
    middle_a: Optional[Tuple[int, ...]] = None
    middle_b: Optional[Tuple[int, ...]] = None
    cross_pruning_applied: bool = False
    compatible_cross_pairs: int = 0
    middle_cross_pruned: int = 0
    cross_a: Optional[Tuple[int, ...]] = None
    cross_b: Optional[Tuple[int, ...]] = None


def _word(values: Sequence[int]) -> Tuple[int, ...]:
    word = tuple(values)
    if not word or any(not isinstance(x, int) or isinstance(x, bool)
                       or x not in (-4, -2, 0, 2, 4) for x in word):
        raise ValueError("factor-4 words need entries -4,-2,0,2,4 and positive length")
    return word


def build_middle_solver(ca, cb, desired, content_profile=None):
    """Build the existing exact middle encoding; also used by resumable jobs.

    No Z3 check is performed here. Returned selectors uniquely describe each
    middle pattern, allowing an audited pattern to be blocked after restart.
    """
    import z3
    n, L = len(ca), len(desired)
    solver = z3.SolverFor("QF_FD")
    selectors = []

    def middle(word, name):
        first, second = [], []
        for i, value in enumerate(word):
            choices = tuple(x for x in (-2,0,2) if value-x in (-2,0,2))
            flags = tuple(z3.Bool("middle_{}_{}_{}".format(name,i,j)) for j in range(len(choices)))
            solver.add(z3.PbEq([(flag,1) for flag in flags],1))
            selectors.extend(flags)
            first.append(tuple(zip(flags,choices)))
            second.append(tuple((flag,value-x) for flag,x in zip(flags,choices)))
        return first+second

    rows = (middle(ca,"a"),middle(cb,"b"))
    middle_target = fold_profile(desired,2)
    for u in range(n+1):
        terms = [z3.If(z3.And(f,g),x*y,0)
                 for row in rows for i in range(2*n)
                 for f,x in row[i] for g,y in row[(i+u)%(2*n)] if x*y]
        solver.add(z3.Sum(terms)==middle_target[u])
    if content_profile is not None:
        for row,even,odd in zip(rows,
                (content_profile.a_even_ones,content_profile.b_even_ones),
                (content_profile.a_odd_ones,content_profile.b_odd_ones)):
            for parity,weight in ((0,even),(1,odd)):
                solver.add(z3.Sum([z3.If(f,x,0) for entry in row[parity::2]
                                   for f,x in entry]) == L//2-2*weight)
    return solver,rows,selectors


def solve_multilevel_lift(
    compressed_a: Sequence[int], compressed_b: Sequence[int], k: int, eta: int,
    timeout_ms: Optional[int] = 5_000,
    content_profile: Optional[TargetContentProfile] = None,
    max_middle_models: Optional[int] = 16,
    cross_compression_pruning: bool = True,
) -> MultilevelLiftResult:
    """Search middle lifts using the remaining shared wall-clock budget.

    None/zero timeout is unlimited; None model cap requests full enumeration.
    Positive caps return UNKNOWN when exhausted, never UNSAT. Optional parity
    content is enforced at the middle level: with d=L/2 even,
    sum(c2[even])=L/2-2*w_even and likewise for odd positions.
    Each middle product is an exact table over finite one-hot domains, not
    a floating-point correlation or a symbolic nonlinear approximation.
    The existing final lift applies its timeout to check(), so its construction
    and verification can add a small overrun to the shared deadline.
    When n is odd, CRT/Gale--Ryser checks factor-4/factor-n compatibility.
    All compatible cross patterns are retained and enforced in binary lift.
    For even n this optional test is inapplicable and the old path is used.
    """
    started = perf_counter()
    if not isinstance(cross_compression_pruning, bool):
        raise ValueError("cross_compression_pruning must be boolean")
    ca, cb = _word(compressed_a), _word(compressed_b)
    if len(ca) != len(cb):
        raise ValueError("compressed words must have equal lengths")
    n, L = len(ca), 4 * len(ca)
    if (not isinstance(k, int) or isinstance(k, bool) or not 0 < k < L // 2
            or not isinstance(eta, int) or isinstance(eta, bool) or eta not in (-1, 1)):
        raise ValueError("invalid target shift/sign")
    if timeout_ms is not None and (not isinstance(timeout_ms, int)
            or isinstance(timeout_ms, bool) or timeout_ms < 0):
        raise ValueError("timeout must be a nonnegative integer or None")
    if max_middle_models is not None and (not isinstance(max_middle_models, int)
            or isinstance(max_middle_models, bool) or max_middle_models <= 0):
        raise ValueError("middle model cap must be positive or None")
    if content_profile is not None:
        _validate_content(content_profile, L, k, eta)
    desired = target_profile(L, k, eta)
    folded = fold_profile(desired, 4)
    actual = tuple(x + y for x, y in zip(
        integer_periodic_autocorrelation_profile(ca),
        integer_periodic_autocorrelation_profile(cb)))
    if actual != folded:
        return MultilevelLiftResult("UNSAT", L, k, eta, 0, 0, 0,
                                    perf_counter() - started, 0.0, False,
                                    reason="fixed factor-4 pair does not equal fold_4(T)")
    cross_pairs = None
    if cross_compression_pruning and n % 2:
        from .cross_compression import compatible_cross_compressions
        cross_pairs = compatible_cross_compressions(ca,cb,k,eta,content_profile)
        if not cross_pairs:
            return MultilevelLiftResult(
                "UNSAT", L,k,eta,0,0,0,perf_counter()-started,0.0,False,
                reason="no compatible factor-4/factor-n patterns (exact CRT/margin/PACF test)",
                cross_pruning_applied=True)
    import z3

    deadline = started + timeout_ms / 1000 if timeout_ms else None
    solver, rows, selectors = build_middle_solver(ca,cb,desired,content_profile)
    middle_target = fold_profile(desired, 2)
    construction = perf_counter() - started
    solve_time = 0.0
    models = checks = binary_checks = unknowns = cross_pruned = 0

    def finish(status, reason=None, child=None, ma=None, mb=None):
        final_cross = (None,None)
        if child is not None and child.verified and cross_pairs is not None:
            final_cross = tuple(tuple(sum(1-2*x for x in bits[r::4]) for r in range(4))
                                for bits in (child.a,child.b))
            if final_cross not in cross_pairs:
                raise RuntimeError("multilevel SAT failed cross-pattern membership audit")
        return MultilevelLiftResult(
            status, L, k, eta, L // 2 if child else 0, n + 1,
            4 if content_profile else 0, construction, solve_time,
            bool(checks or binary_checks),
            a=child.a if child else None, b=child.b if child else None,
            profile=child.profile if child else None, verified=bool(child and child.verified),
            cross_pattern_count=child.cross_pattern_count if child else 0,
            reason=reason, middle_models=models, middle_checks=checks,
            binary_checks=binary_checks, unknown_children=unknowns,
            middle_a=ma, middle_b=mb,
            cross_pruning_applied=cross_pairs is not None,
            compatible_cross_pairs=len(cross_pairs or ()), middle_cross_pruned=cross_pruned,
            cross_a=final_cross[0],cross_b=final_cross[1])

    def remaining():
        return None if deadline is None else ceil((deadline - perf_counter()) * 1000)

    while True:
        left = remaining()
        if left is not None and left <= 0:
            return finish("UNKNOWN", "shared time budget exhausted")
        solver.set(timeout=left if left is not None else 0)
        tick = perf_counter()
        outcome = solver.check()
        solve_time += perf_counter() - tick
        checks += 1
        if outcome == z3.unsat:
            return finish("UNKNOWN" if unknowns else "UNSAT",
                          "unresolved binary children" if unknowns else "all middle lifts exhausted")
        if outcome != z3.sat:
            return finish("UNKNOWN", solver.reason_unknown())
        model = solver.model()
        ma, mb = (tuple(sum(x for f, x in entry
                           if z3.is_true(model.evaluate(f, model_completion=True)))
                        for entry in row) for row in rows)
        # Independent exact audit before any child query.
        observed = tuple(x + y for x, y in zip(integer_periodic_autocorrelation_profile(ma),
                                               integer_periodic_autocorrelation_profile(mb)))
        if observed != middle_target or any(ma[i] + ma[i+n] != ca[i]
                or mb[i] + mb[i+n] != cb[i] for i in range(n)):
            raise RuntimeError("middle SAT failed exact compression/profile audit")
        models += 1
        left = remaining()
        if left is not None and left <= 0:
            return finish("UNKNOWN", "shared budget exhausted before binary lift", ma=ma, mb=mb)
        options = {}
        if cross_pairs is not None:
            tick = perf_counter()
            from .cross_compression import middle_matches_cross
            eligible = tuple((a,b) for a,b in cross_pairs
                             if middle_matches_cross(ma,a) and middle_matches_cross(mb,b))
            options["cross_compressions"] = eligible
            construction += perf_counter()-tick
        if cross_pairs is not None and not eligible:
            cross_pruned += 1  # exact impossibility for this middle model
        else:
            left = remaining()
            if left is not None and left <= 0:
                return finish("UNKNOWN", "shared budget exhausted after cross check", ma=ma, mb=mb)
            child = solve_compressed_lift(ma, mb, k, eta, left, content_profile, **options)
            construction += child.construction_time
            solve_time += child.solve_time
            binary_checks += int(child.z3_executed)
            if child.status == "SAT":
                return finish("SAT", child=child, ma=ma, mb=mb)
            unknowns += int(child.status == "UNKNOWN")
        # This excludes one fully specified middle model only. UNKNOWN is
        # recorded above and is never turned into an infeasibility conclusion.
        solver.add(z3.Or([f != model.evaluate(f, model_completion=True) for f in selectors]))
        if max_middle_models is not None and models >= max_middle_models:
            return finish("UNKNOWN", "middle model budget exhausted")
