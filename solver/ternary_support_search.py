"""Periodic ternary support matching followed by exact sign completion.

This is a coarse search: every hit still needs the existing binary lift.
For U_i=[a_i != 0], M_u=P_U(u)+P_V(u), a necessary condition is
M_u == C_u (mod 2). For fixed supports a_i=(-1)**s_i on U,
the EXACT equation is #edges with s_i XOR s_j == (M_u-C_u)/2.
Its parity is linear over GF(2), allowing exact Gaussian elimination.
All arithmetic and pruning here are integral. No symmetry restrictions.
"""
from time import perf_counter

from .compressed_repair import _validated_reference
from .orientation_gf2 import OrientationSystem
from .structured_energy import integer_periodic_autocorrelation_profile as pacf


def support_mask(word):
    return sum(1 << i for i, v in enumerate(word) if v)


def support_signature(mask, d):
    """Pack every independent periodic PACF parity into d//2 bits."""
    full = (1 << d)-1
    return sum((bin(mask & (((mask << u) | (mask >> (d-u))) & full)).count('1') & 1)
               << (u-1) for u in range(1, d//2+1))


def relocated_supports(mask, d):
    """ALL supports of equal size differing by at most one relocation."""
    yield mask
    occupied = [i for i in range(d) if mask >> i & 1]
    vacant = [i for i in range(d) if not mask >> i & 1]
    for i in occupied:
        for j in vacant:
            yield mask ^ (1 << i) ^ (1 << j)


def support_neighborhood(mask, d):
    """Same ALL one-relocation supports with exact incremental signatures.

    In F2, for D=e_i+e_j, P(U+D)=P(U)+cross(U,D)+P(D).
    Column g_i[u]=U[i+u]+U[i-u]. At independent shift u, P(D)
    has precisely one bit, at min(|i-j|,d-|i-j|), since d is odd.
    Thus the new signature is base XOR g_i XOR g_j XOR distance_bit.
    """
    base = support_signature(mask, d)
    yield mask, base
    columns = [sum((((mask >> ((i+u) % d)) ^ (mask >> ((i-u) % d))) & 1) << (u-1)
                   for u in range(1, d//2+1)) for i in range(d)]
    occupied = [i for i in range(d) if mask >> i & 1]
    vacant = [i for i in range(d) if not mask >> i & 1]
    for i in occupied:
        for j in vacant:
            distance = min(abs(i-j), d-abs(i-j))
            yield mask ^ (1 << i) ^ (1 << j), base ^ columns[i] ^ columns[j] ^ (1 << (distance-1))


def matching_supports(a, b, wanted):
    """MITM on complete mod-2 vectors; keep ALL literal support preimages.

    Scope: at most one support relocation on EACH side, fixed side sizes.
    Signs are unrestricted here, so this is not a ternary Hamming ball.
    """
    d = len(a)
    target = sum((wanted[u] & 1) << (u-1) for u in range(1, d//2+1))
    index = {}
    for mask, signature in support_neighborhood(support_mask(a), d):
        index.setdefault(signature, []).append(mask)
    for mask_b, signature in support_neighborhood(support_mask(b), d):
        needed = target ^ signature
        for mask_a in index.get(needed, ()):
            yield mask_a, mask_b


def support_sign_system(masks, d, wanted, negative_counts):
    """Necessary mod-4 sign equations plus per-side negative-count parity.

    Ordinary periodic edges have NO negacyclic wrap signs. Variables live
    on NONZERO compressed entries, unlike the existing binary-lift system.
    """
    variables = tuple((s, i) for s, mask in enumerate(masks) for i in range(d) if mask >> i & 1)
    ids = {v: i for i, v in enumerate(variables)}
    equations = []
    consistent = True
    for u in range(1, d//2+1):
        row = terms = 0
        for s, mask in enumerate(masks):
            for i in range(d):
                j = (i+u) % d
                if mask >> i & 1 and mask >> j & 1:
                    row ^= (1 << ids[s, i]) ^ (1 << ids[s, j])
                    terms += 1
        if abs(wanted[u]) > terms or (terms-wanted[u]) % 2:
            consistent = False
            break
        equations.append((u, row, ((terms-wanted[u])//2) & 1, terms))
    for s, n in enumerate(negative_counts):
        row = sum(1 << col for col, (side, _) in enumerate(variables) if side == s)
        equations.append((d+s, row, n & 1, bin(row).count('1')))
    basis = {}
    for _, row, rhs, _ in equations:
        while row:
            pivot = row.bit_length()-1
            if pivot not in basis:
                basis[pivot] = row, rhs
                break
            prior, value = basis[pivot]
            row ^= prior
            rhs ^= value
        if not row and rhs:
            consistent = False
    return OrientationSystem(variables, tuple(equations),
                             tuple((p, *basis[p]) for p in sorted(basis)), consistent)


def complete_support_signs(a, b, target, masks, *, timeout_ms=100,
                           backend='native', max_assignments=2**23, conditioning=True, content_pruning=True,
                           quadratic=None, hybrid_tail=None, conditioned_hybrid=None, cardinality_pruning=True):
    """ALL signs on one fixed support, with the reference's symbol counts.

    UNSAT_FIXED_SUPPORT is confined to these masks and these counts.
    SAT returns one integer-audited compressed witness, not all lifts.
    """
    started = perf_counter()
    a, b, d, wanted = _validated_reference(a, b, target)
    masks = tuple(masks)
    if (len(masks) != 2 or any(type(m) is not int or not 0 <= m < 1 << d for m in masks)
            or any(bin(m).count('1') != sum(v != 0 for v in w) for m, w in zip(masks, (a, b)))):
        raise ValueError('supports must preserve both reference support sizes')
    if type(timeout_ms) is not int or timeout_ms <= 0:
        raise ValueError('timeout_ms must be a positive integer')
    if backend not in ('native', 'z3') or type(max_assignments) is not int or not 1 <= max_assignments <= 2**31:
        raise ValueError('native or z3 backend and assignment budget in 1..2**31 required')
    if type(conditioning) is not bool:
        raise ValueError('conditioning must be boolean')
    if type(content_pruning) is not bool:
        raise ValueError('content_pruning must be boolean')
    if quadratic is not None and type(quadratic) is not bool:
        raise ValueError('quadratic must be boolean or None (automatic)')
    if hybrid_tail is not None and (type(hybrid_tail) is not int or not 0 <= hybrid_tail <= 8):
        raise ValueError('hybrid_tail must be None or an integer in 0..8')
    if conditioned_hybrid is not None and type(conditioned_hybrid) is not bool:
        raise ValueError('conditioned_hybrid must be boolean or None (automatic)')
    if type(cardinality_pruning) is not bool:
        raise ValueError('cardinality_pruning must be boolean')
    negative = (a.count(-1), b.count(-1))
    if content_pruning and d == 43 and min(bin(m).count('1') for m in masks) <= 7:
        from .l86_content_certificate import l86_content_certificate
        certificate = l86_content_certificate()
        if min(bin(m).count('1') for m in masks) < certificate.minimum_weight:
            return dict(status='UNSAT_FIXED_SUPPORT', reason='certified L86 content support bound',
                rank=0, free_variables=d, words=None, assignments=0, count_matches=0,
                correlation_checks=0, construction_seconds=perf_counter()-started, solve_seconds=0.0,
                minimum_support=certificate.minimum_weight)
    system = support_sign_system(masks, d, wanted, negative)
    common = dict(rank=system.rank, free_variables=len(system.free_columns), words=None,
                  construction_seconds=0.0, solve_seconds=0.0)
    if not system.consistent:
        common.update(status='UNSAT_FIXED_SUPPORT', reason='integer range / GF2 obstruction',
                      construction_seconds=perf_counter()-started)
        return common
    if backend == 'native':
        from .ternary_signs_native import solve_sign_system
        elapsed = perf_counter()-started
        common.update(solve_sign_system(system, masks, d, wanted, negative, max_assignments,
                                       conditioning=conditioning, quadratic=quadratic, hybrid_tail=hybrid_tail,
                                       conditioned_hybrid=conditioned_hybrid, cardinality_pruning=cardinality_pruning))
        common['construction_seconds'] += elapsed
        if common['words'] is not None:
            _audit_sign_words(common['words'], a, b, target, wanted, masks)
        return common
    import z3
    solver = z3.Solver()
    solver.set(timeout=timeout_ms)
    free = [z3.Bool('f{}'.format(i)) for i in range(len(system.free_columns))]
    signs = {}
    for key, (mask, constant) in zip(system.variables, system.expressions()):
        expr = z3.BoolVal(bool(constant))
        for i in range(len(free)):
            if mask >> i & 1:
                expr = z3.Xor(expr, free[i])
        signs[key] = expr
    for s in range(2):
        terms = [(expr, 1) for (side, _), expr in signs.items() if side == s]
        solver.add(z3.PbEq(terms, negative[s]) if terms else z3.BoolVal(negative[s] == 0))
    for u in range(1, d//2+1):
        terms = [(z3.Xor(signs[s, i], signs[s, (i+u) % d]), 1)
                 for s, i in system.variables if (s, (i+u) % d) in signs]
        count = (len(terms)-wanted[u])//2
        solver.add(z3.PbEq(terms, count) if terms else z3.BoolVal(count == 0))
    common['construction_seconds'] = perf_counter()-started
    started = perf_counter()
    status = solver.check()
    common['solve_seconds'] = perf_counter()-started
    if status != z3.sat:
        common.update(status='UNSAT_FIXED_SUPPORT' if status == z3.unsat else 'UNKNOWN',
                      reason=solver.reason_unknown() if status == z3.unknown else None)
        return common
    model = solver.model()
    words = tuple(tuple(0 if (s, i) not in signs else
                        -1 if z3.is_true(model.eval(signs[s, i], model_completion=True)) else 1
                        for i in range(d)) for s in range(2))
    _audit_sign_words(words, a, b, target, wanted, masks)
    common.update(status='SAT_COMPRESSED', words=words)
    return common


def _audit_sign_words(words, a, b, target, wanted, masks):
    """Full Python PACF is independent of the native popcount equations."""
    _validated_reference(*words, target)
    if (tuple(x+y for x, y in zip(pacf(words[0]), pacf(words[1]))) != wanted
            or any(sorted(w) != sorted(r) for w, r in zip(words, (a, b)))
            or tuple(support_mask(w) for w in words) != masks):
        raise RuntimeError('support sign model failed independent integer audit')


def repair_support_signs(a, b, target, *, max_supports=8, timeout_ms=50,
                         backend='native', max_assignments=2**23, stop_event=None):
    """Bounded completion of parity-matched support neighborhoods.

    Exhaustive UNSAT applies only if EVERY matched support finished UNSAT.
    A cap, solver UNKNOWN, or cancellation never gives that conclusion.
    """
    started = perf_counter()
    a, b, _, wanted = _validated_reference(a, b, target)
    if type(max_supports) is not int or max_supports < 1 or type(timeout_ms) is not int or timeout_ms < 1:
        raise ValueError('positive integer support and solver budgets required')
    if backend not in ('native', 'z3') or type(max_assignments) is not int or not 1 <= max_assignments <= 2**31:
        raise ValueError('native or z3 backend and assignment budget in 1..2**31 required')
    attempts = []
    incomplete = False
    if stop_event is not None and stop_event.is_set():
        return dict(status='UNKNOWN', words=None, attempts=[], elapsed=perf_counter()-started)
    for masks in matching_supports(a, b, wanted):
        if len(attempts) >= max_supports or (stop_event is not None and stop_event.is_set()):
            incomplete = True
            break
        out = complete_support_signs(a, b, target, masks, timeout_ms=timeout_ms,
                                     backend=backend, max_assignments=max_assignments)
        attempts.append(dict(masks=masks, result=out))
        incomplete |= out['status'] == 'UNKNOWN'
        if out['words'] is not None:
            return dict(status='SAT_COMPRESSED', words=out['words'], attempts=attempts,
                        elapsed=perf_counter()-started)
    return dict(status='UNKNOWN' if incomplete else 'UNSAT_SUPPORT_NEIGHBORHOOD',
                words=None, attempts=attempts, elapsed=perf_counter()-started)
