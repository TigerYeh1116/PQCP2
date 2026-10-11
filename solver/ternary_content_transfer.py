"""Exact content-changing part of the target-feasible ternary radius-four ball.

For fixed per-side sums, count changes are (dn,dz,dp)=(t,-2t,t).
Fixed total squared norm forces t_A+t_B=0. A nonzero transfer within four
changed coordinates must therefore be 00 <-> +- on opposite sides.
Together with the fixed-content two-swap solver this covers the whole ball
for the specified target parity weights. No claim across other targets.
"""
from time import perf_counter

from .compressed_repair import CompressedRepairResult, _validated_reference
from .structured_energy import integer_periodic_autocorrelation_profile as pacf


def _moves(word, create):
    """Each move has its FULL independent integer PACF delta as a tuple key."""
    d = len(word)
    for p in range(d):
        for q in range(p+1, d):
            if create:
                if word[p] != 0 or word[q] != 0:
                    continue
                replacements = ((1, -1), (-1, 1))
            else:
                if word[p]*word[q] != -1:
                    continue
                replacements = ((0, 0),)
            for vp, vq in replacements:
                ep, eq = vp-word[p], vq-word[q]
                delta = tuple(ep*(word[(p+u)%d]+word[(p-u)%d]) +
                              eq*(word[(q+u)%d]+word[(q-u)%d]) +
                              ep*eq*(int((p+u)%d == q)+int((q+u)%d == p))
                              for u in range(1, d//2+1))
                yield delta, (p, q, vp, vq)


def _apply(word, move):
    p, q, vp, vq = move
    result = list(word)
    result[p], result[q] = vp, vq
    return tuple(result)


def repair_content_transfer(a, b, target):
    """Complete opposite-side 00/+- transfer search, not fixed-content repair.

UNSAT_TRANSFER excludes only this transfer class. Hash equality compares all
integer coordinates; hash collisions cannot discard a valid signature.
One representative per delta suffices because resulting counts, hence parity
lift feasibility, are identical within each direction.
"""
    started = perf_counter()
    a, b, d, wanted = _validated_reference(a, b, target)
    observed = tuple(x+y for x, y in zip(pacf(a), pacf(b)))
    residual = tuple(wanted[u]-observed[u] for u in range(1, d//2+1))
    construction = perf_counter()-started
    started = perf_counter()
    signatures = probes = 0
    for create_a in (True, False):
        feasible = True
        for word, create, e, o in (
                (a, create_a, target.a_even_ones, target.a_odd_ones),
                (b, not create_a, target.b_even_ones, target.b_odd_ones)):
            n = word.count(-1)+(1 if create else -1)
            z = word.count(0)+(-2 if create else 2)
            if min(n, z, d-n-z) < 0 or not (0 <= e-n <= z and 0 <= o-n <= z):
                feasible = False
        if not feasible:
            continue
        table = {}
        for delta, move in _moves(a, create_a):
            table.setdefault(delta, move)
            signatures += 1
        for delta, move_b in _moves(b, not create_a):
            probes += 1
            move_a = table.get(tuple(r-v for r, v in zip(residual, delta)))
            if move_a is None:
                continue
            words = (_apply(a, move_a), _apply(b, move_b))
            _validated_reference(*words, target)
            exact = tuple(x+y for x, y in zip(*(pacf(w) for w in words)))
            distance = sum(x != y for w, old in zip(words, (a, b)) for x, y in zip(w, old))
            if exact != wanted or distance != 4 or any(sum(w) != sum(old) for w, old in zip(words, (a, b))):
                raise RuntimeError('content transfer failed independent integer audit')
            return CompressedRepairResult('SAT_COMPRESSED', construction, perf_counter()-started,
                                          4, words, distance), dict(signatures=signatures, probes=probes)
    return CompressedRepairResult('UNSAT_TRANSFER', construction, perf_counter()-started, 4), dict(
        signatures=signatures, probes=probes)
