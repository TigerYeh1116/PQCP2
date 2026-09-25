"""Bounded exact-model enumeration of one factor-2 job, with full orbit audit.

Blocking an exact orientation assignment preserves every OTHER lift of the
fixed compression/content/target. We do not equate PACF or compressed matches
with full-sequence equivalence. The existing exact full orbit key counts
classes; it is not used to block unchecked transformations.

This conservative implementation rebuilds the existing encoding per check.
It reuses mathematical code, not learned clauses. Report rebuilding and total
cost; do not present this as an incremental-solver speed benchmark.
"""

from dataclasses import dataclass
import math
from time import perf_counter
from typing import Callable, Optional, Sequence, Tuple

from .compressed_completion import CompressedLiftJob
from .compressed_lift import solve_compressed_lift
from .solution_equivalence import pqcp_equivalence_key
from .structured_energy import compress_binary_sequence, target_profile
from .target_profiles import pair_content
from .verifier import verify_pqcp


def orientation_assignment(job: CompressedLiftJob, a: Sequence[int], b: Sequence[int]) -> Tuple[int, ...]:
    """Audit a model against its exact job BEFORE it becomes an exclusion."""
    if job.factor != 2:
        raise ValueError("binary lift enumeration currently requires factor 2")
    check = verify_pqcp(a, b)
    t = job.target
    if (not check.is_valid or check.profile != target_profile(t.L, t.k, t.eta)
            or compress_binary_sequence(a, 2) != job.compressed_a
            or compress_binary_sequence(b, 2) != job.compressed_b
            or pair_content(a, b) != (t.a_even_ones, t.a_odd_ones, t.b_even_ones, t.b_odd_ones)):
        raise ValueError("enumeration model does not solve this exact compressed job")
    return tuple(word[i] for word, compressed in ((a, job.compressed_a), (b, job.compressed_b))
                 for i, value in enumerate(compressed) if value == 0)


@dataclass(frozen=True)
class LiftEnumerationResult:
    terminal: str  # EXHAUSTED, UNKNOWN, MODEL_LIMIT or TIME_LIMIT
    models: tuple
    exclusions: tuple
    equivalence_keys: tuple
    checks: int
    elapsed: float
    construction_time: float
    solve_time: float


def enumerate_lifts(job: CompressedLiftJob, *, max_models: int = 32,
                    seconds: float = 2.0, check_timeout_ms: int = 1000,
                    orientation_steps: int = 0, seed: int = 123,
                    excluded_orientations: Sequence[Sequence[int]] = (),
                    on_model: Optional[Callable[[dict], None]] = None) -> LiftEnumerationResult:
    """Yield audited models to a persistence callback before continuing.

    No L.txt access/writes. Resume callers own the exact-job association and
    previously seen equivalence keys. Returned classes cover this call only.
    Limits include repeated/equivalent models. UNKNOWN proves nothing;
    EXHAUSTED means no model outside the supplied+new exact exclusions.
    Wall budget is checked between calls; final build/check/audit may overrun.
    """
    if job.factor != 2:
        raise ValueError("enumeration currently supports factor-2 jobs only")
    if type(max_models) is not int or max_models <= 0 or type(check_timeout_ms) is not int or check_timeout_ms <= 0:
        raise ValueError("model and check limits must be positive integers")
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("seconds must be positive and finite")
    if type(orientation_steps) is not int or orientation_steps < 0:
        raise ValueError("orientation steps must be nonnegative")
    excluded = [tuple(row) for row in excluded_orientations]
    width = job.compressed_a.count(0) + job.compressed_b.count(0)
    if any(len(row) != width or any(type(bit) is not int or bit not in (0, 1) for bit in row) for row in excluded):
        raise ValueError("invalid initial orientation exclusions")
    excluded = list(dict.fromkeys(excluded))
    models, keys = [], set()
    checks = 0
    construction = solve = 0.0
    started = perf_counter()
    terminal = "MODEL_LIMIT"
    while len(models) < max_models:
        remaining = seconds - (perf_counter() - started)
        if remaining <= 0:
            terminal = "TIME_LIMIT"
            break
        result = solve_compressed_lift(job.compressed_a, job.compressed_b, job.target.k, job.target.eta,
            min(check_timeout_ms, max(1, math.ceil(remaining*1000))), job.target,
            orientation_search_steps=orientation_steps if not excluded else 0,
            orientation_search_seed=seed, excluded_orientations=excluded)
        checks += int(result.z3_executed)
        construction += result.construction_time
        solve += result.solve_time
        if result.status != "SAT":
            terminal = "EXHAUSTED" if result.status == "UNSAT" else "UNKNOWN"
            break
        direction = orientation_assignment(job, result.a, result.b)
        if direction in excluded:
            raise RuntimeError("solver returned an excluded orientation")
        key = pqcp_equivalence_key(result.a, result.b)
        model = dict(A=list(result.a), B=list(result.b), profile=list(result.profile),
                     orientation=list(direction), equivalence_key=list(key),
                     new_class_in_call=key not in keys, elapsed=perf_counter()-started,
                     construction_time=result.construction_time, solve_time=result.solve_time,
                     z3_executed=result.z3_executed)
        if on_model is not None:
            on_model(model)  # persistence failure propagates; never silently discard a SAT
        models.append(model)
        excluded.append(direction)
        keys.add(key)
    return LiftEnumerationResult(terminal, tuple(models), tuple(excluded), tuple(sorted(keys)),
                                 checks, perf_counter()-started, construction, solve)
