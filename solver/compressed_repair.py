"""Exact bounded repair of a near-matching odd factor-2 compressed pair.

Words are normalized ternary sums (x[i]+x[i+d])/2. SAT certifies only
the folded target, never a binary PQCP. UNSAT is restricted to this
reference's fixed contents and Hamming ball; UNKNOWN proves nothing.
No approximate PSD tests or heuristic exclusions are used in the encoding.
This experimental module is not enabled by the production entry point.
"""

from dataclasses import dataclass
from functools import lru_cache
from time import perf_counter
from typing import Optional, Tuple

from .structured_energy import fold_profile, integer_periodic_autocorrelation_profile, target_profile
from .target_profiles import TargetContentProfile, target_content_profiles


@dataclass(frozen=True)
class CompressedRepairResult:
    status: str
    construction_seconds: float
    solve_seconds: float
    radius: int
    words: Optional[Tuple[Tuple[int, ...], Tuple[int, ...]]] = None
    distance: Optional[int] = None
    reason: Optional[str] = None


@lru_cache(maxsize=16)
def _valid_targets(length):
    return frozenset(target_content_profiles(length))


def _validated_reference(a, b, target):
    a, b = tuple(a), tuple(b)
    d = len(a)
    if (d < 3 or d % 2 == 0 or len(b) != d or target.L != 2*d
            or any(type(x) is not int or x not in (-1, 0, 1) for x in a+b)):
        raise ValueError('equal odd ternary words with L=2d are required')
    if target not in _valid_targets(target.L):
        raise ValueError('target must meet the necessary content conditions')
    # This also checks that a lift with the specified parity content exists.
    for word, e, o in ((a, target.a_even_ones, target.a_odd_ones),
                       (b, target.b_even_ones, target.b_odd_ones)):
        n, z = word.count(-1), word.count(0)
        if e+o != 2*n+z or not 0 <= e-n <= z or not 0 <= o-n <= z:
            raise ValueError('reference contents do not admit the prescribed parity weights')
    folded = fold_profile(target_profile(target.L, target.k, target.eta), 2)
    if any(v % 4 for v in folded):
        raise ValueError('normalized target must be integral')
    wanted = tuple(v//4 for v in folded)
    if sum(x*x for x in a+b) != wanted[0]:
        raise ValueError('reference contents do not satisfy the zero-shift target')
    return a, b, d, wanted


def repair_compressed_pair(a, b, target: TargetContentProfile, *, radius: int = 4,
                           timeout_ms: int = 1000) -> CompressedRepairResult:
    """Search ALL ternary pairs in the specified content-preserving ball.

    For exclusive positive/negative indicators p,n,
    v_i v_j = [p_i&p_j]+[n_i&n_j]-[p_i&n_j]-[n_i&p_j].
    Every PACF constraint is therefore an exact pseudo-Boolean equality.
    Counts force shift zero; reflection covers the remaining shifts.
    """
    started = perf_counter()
    a, b, d, wanted = _validated_reference(a, b, target)
    if type(radius) is not int or not 0 <= radius <= 2*d:
        raise ValueError('radius must be an integer in 0..2d')
    if type(timeout_ms) is not int or timeout_ms <= 0:
        raise ValueError('timeout_ms must be positive')

    import z3
    solver = z3.Solver()
    solver.set(timeout=timeout_ms)
    positive = [[z3.Bool('p_{}_{}'.format(s, i)) for i in range(d)] for s in range(2)]
    negative = [[z3.Bool('n_{}_{}'.format(s, i)) for i in range(d)] for s in range(2)]
    changed = []
    for s, word in enumerate((a, b)):
        p, n = positive[s], negative[s]
        solver.add(z3.PbEq([(v, 1) for v in p], word.count(1)))
        solver.add(z3.PbEq([(v, 1) for v in n], word.count(-1)))
        for i, value in enumerate(word):
            solver.add(z3.Not(z3.And(p[i], n[i])))
            same = p[i] if value == 1 else n[i] if value == -1 else z3.Not(z3.Or(p[i], n[i]))
            changed.append((z3.Not(same), 1))
    solver.add(z3.PbLe(changed, radius))
    for u in range(1, (d+1)//2):
        terms = []
        for p, n in zip(positive, negative):
            for i in range(d):
                j = (i+u) % d
                terms.extend(((z3.And(p[i], p[j]), 1), (z3.And(n[i], n[j]), 1),
                              (z3.And(p[i], n[j]), -1), (z3.And(n[i], p[j]), -1)))
        solver.add(z3.PbEq(terms, wanted[u]))
    construction = perf_counter()-started
    started = perf_counter()
    status = solver.check()
    elapsed = perf_counter()-started
    if status != z3.sat:
        return CompressedRepairResult('UNSAT_LOCAL' if status == z3.unsat else 'UNKNOWN',
            construction, elapsed, radius, reason=solver.reason_unknown() if status == z3.unknown else None)
    model = solver.model()
    words = tuple(tuple(int(z3.is_true(model.eval(p, model_completion=True))) -
                        int(z3.is_true(model.eval(n, model_completion=True)))
                        for p, n in zip(ps, ns)) for ps, ns in zip(positive, negative))
    distance = sum(x != y for word, ref in zip(words, (a, b)) for x, y in zip(word, ref))
    exact = tuple(sum(v) for v in zip(*(integer_periodic_autocorrelation_profile(w) for w in words)))
    if (exact != wanted or distance > radius or
            any(sorted(w) != sorted(r) for w, r in zip(words, (a, b)))):
        raise RuntimeError('compressed SAT model failed independent integer audit')
    return CompressedRepairResult('SAT_COMPRESSED', construction, elapsed, radius, words, distance)
