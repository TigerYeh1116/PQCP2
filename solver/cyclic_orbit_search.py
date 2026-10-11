"""Rotation-quotiented global partner supports with exact full PQCP completion.

For odd d, every compression rotation is induced by an even original
binary rotation, preserving even/odd counts. Its modulated difference row
rotates by the same amount. Thus quotienting V by rotations preserves ALL
full solution classes. This is not a fixed-centre neighbourhood solver.
"""
from time import perf_counter

from .compressed_repair import _validated_reference
from .cyclic_support import parity_support_family, rotation_representatives
from .cyclic_support_native import nearest_supports
from .pqcp_support_solver import complete_pqcp_supports
from .ternary_support_search import support_mask


def canonical_support_rotation(mask, d):
    full = (1 << d)-1
    return min([mask]+[((mask << u) | (mask >> (d-u))) & full for u in range(1, d)])


def search_cyclic_orbits(a, b, target, *, max_orbits=4096, max_assignments=2**23, stop_event=None):
    started = perf_counter()
    a, b, d, wanted = _validated_reference(a, b, target)
    if (d not in (3, 5, 11, 43) or type(max_orbits) is not int or not 1 <= max_orbits <= 4096
            or type(max_assignments) is not int or not 1 <= max_assignments <= 2**31):
        raise ValueError('supported prime half-length, orbit and assignment budgets required')
    masks = support_mask(a), support_mask(b)
    k = next(u for u in range(1, d//2+1) if wanted[u] & 1)
    generation = []
    attempts = []
    seen = set()
    incomplete = False

    def finish(status, words=None):
        return dict(status=status, words=words, verified=words is not None, generation=generation,
                    attempts=attempts, elapsed=perf_counter()-started,
                    scope='either literal reference side fixed; other side modulo rotations; fixed contents')

    for fixed in (0, 1):
        if stop_event is not None and stop_event.is_set():
            return finish('UNKNOWN')
        family = parity_support_family(masks[fixed], d, k)
        representatives = rotation_representatives(family)
        # Every fixed-weight mask has the same distance to zero: reuse the
        # certified kernel to enumerate ALL admissible orbit representatives.
        out = nearest_supports(representatives, 0, bin(masks[1-fixed]).count('1'), capacity=max_orbits)
        out['fixed_side'] = fixed
        out['literal_family_size'] = family.size
        generation.append(out)
        incomplete |= not out['complete_nearest']
        for other in out['masks']:
            if stop_event is not None and stop_event.is_set():
                return finish('UNKNOWN')
            pair = (masks[0], other) if fixed == 0 else (other, masks[1])
            key = tuple(canonical_support_rotation(m, d) for m in pair)
            if key in seen:
                continue
            seen.add(key)
            solved = complete_pqcp_supports(a, b, target, pair, max_assignments=max_assignments)
            attempts.append(dict(masks=pair, result=solved))
            incomplete |= solved['status'] == 'UNKNOWN'
            if solved['verified']:
                return finish('SAT_PQCP', solved['words'])
    return finish('UNKNOWN' if incomplete else 'UNSAT_FIXED_SIDE_UNION')
