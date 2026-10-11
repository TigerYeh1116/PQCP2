"""Global mod-2 support generation followed by the existing exact sign solver.

One side's support is fixed, the other is unrestricted before distance
ranking. Try both directions. Distance/attempt caps are scheduling choices:
a miss never excludes the reference, farther supports or a content family.
"""
from time import perf_counter

from .compressed_repair import _validated_reference
from .cyclic_support import parity_support_family
from .cyclic_support_native import nearest_supports
from .ternary_support_search import complete_support_signs, support_mask


def repair_cyclic_supports(a, b, target, *, max_supports=8,
                           max_assignments=2**23, stop_event=None, full_completion=None):
    started = perf_counter()
    a, b, d, wanted = _validated_reference(a, b, target)
    if full_completion is not None and type(full_completion) is not bool:
        raise ValueError('full_completion must be boolean or None (automatic)')
    if full_completion is None:
        full_completion = d == 43
    if d not in (3, 5, 11, 43):
        raise ValueError('cyclic norm generation supports half-lengths 3,5,11,43')
    if type(max_supports) is not int or not 1 <= max_supports <= 4096:
        raise ValueError('max_supports must be in 1..4096')
    if type(max_assignments) is not int or not 1 <= max_assignments <= 2**31:
        raise ValueError('max_assignments must be in 1..2**31')
    shifts = [u for u in range(1, d//2+1) if wanted[u] & 1]
    if wanted[0] != d or len(shifts) != 1:
        raise ValueError('one independent odd sidelobe and total support d required')
    masks = tuple(support_mask(w) for w in (a, b))
    generated = []
    attempts = []
    families = []
    exhausted_layers = []

    def finish(status, words=None, full_words=None):
        if status == 'BOUNDED_MISS' and any(a['result']['status'] == 'UNKNOWN' for a in attempts):
            status = 'UNKNOWN'
        return dict(status=status, words=words, full_words=full_words, generation=generated, attempts=attempts,
                    verified=full_words is not None,
                    exhausted_layers=exhausted_layers,
                    elapsed=perf_counter()-started,
                    completion_scope='binary' if full_completion else 'compressed',
                    scope='ranked parity-compatible supports with either reference side fixed')

    for fixed in (0, 1):
        if stop_event is not None and stop_event.is_set():
            return finish('UNKNOWN')
        construction = perf_counter()
        family = parity_support_family(masks[fixed], d, shifts[0])
        families.append(family)
        family_seconds = perf_counter()-construction
        out = nearest_supports(family, masks[1-fixed], bin(masks[1-fixed]).count('1'),
                               capacity=max_supports)
        out['construction_seconds'] += family_seconds
        out['fixed_side'] = fixed
        generated.append(out)
    # Balance both directions; retain all literal ties up to the shared budget.
    resolved = {}
    active = list(generated)

    def literal_pair(g, other):
        fixed = g['fixed_side']
        return (masks[0], other) if fixed == 0 else (other, masks[1])

    while active:
        for phase in range(max(len(g['masks']) for g in active)):
            for g in active:
                if len(attempts) >= max_supports:
                    return finish('BOUNDED_MISS')
                if stop_event is not None and stop_event.is_set():
                    return finish('UNKNOWN')
                if phase >= len(g['masks']):
                    continue
                pair = literal_pair(g, g['masks'][phase])
                if pair in resolved:
                    continue
                if full_completion:
                    from .pqcp_support_solver import complete_pqcp_supports
                    full = complete_pqcp_supports(a, b, target, pair, max_assignments=max_assignments)
                    coarse = full['attempts'].get('coarse', {})
                    # Unresolved lifts stay eligible for the original queue.
                    coarse_words = coarse.get('words') if full['status'] != 'UNSAT_BINARY_SUPPORT' else None
                    out = dict(full, full_words=full['words'], words=coarse_words,
                               assignments=sum(r.get('assignments', 0) for r in full['attempts'].values()))
                else:
                    out = complete_support_signs(a, b, target, pair, max_assignments=max_assignments)
                attempts.append(dict(masks=pair, result=out))
                resolved[pair] = out['status']
                if out['words'] is not None:
                    return finish('SAT_PQCP' if out.get('full_words') is not None else 'SAT_COMPRESSED',
                                  out['words'], out.get('full_words'))
        if not full_completion:
            break
        following = []
        for g in active:
            if (g['distance'] is None or not g['complete_nearest'] or
                    any(resolved.get(literal_pair(g, other)) != 'UNSAT_BINARY_SUPPORT' for other in g['masks'])):
                continue
            # Advance only after ALL literal ties of this distance layer
            # have full UNSAT proofs. UNKNOWN/truncation cannot skip a layer.
            exhausted_layers.append(dict(fixed_side=g['fixed_side'], distance=g['distance'],
                                         literal_support_pairs=len(g['masks'])))
            if len(attempts) >= max_supports:
                return finish('BOUNDED_MISS')
            if stop_event is not None and stop_event.is_set():
                return finish('UNKNOWN')
            floor = g['distance']+1
            if floor > d:
                continue
            fixed = g['fixed_side']
            out = nearest_supports(families[fixed], masks[1-fixed], bin(masks[1-fixed]).count('1'),
                                   capacity=max_supports-len(attempts), minimum_distance=floor)
            out['fixed_side'] = fixed
            generated.append(out)
            following.append(out)
        active = following
    return finish('BOUNDED_MISS')
