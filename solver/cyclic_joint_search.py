"""Exact nearest supports on a one-relocation/unrestricted union of slices.

Every equal-weight U at distance <=2 is completed by its entire norm family
of V, and conversely. Minimum combined distance is certified on this union.
Output ties and subsequent signs can be capped; a miss is never global UNSAT.
"""
from time import perf_counter

from .compressed_repair import _validated_reference
from .cyclic_support import parity_support_family
from .cyclic_support_native import nearest_supports
from .ternary_support_search import complete_support_signs, relocated_supports, support_mask


def two_bit_supports(mask, d):
    """All supports at distance <=2: relocation, two additions or two removals."""
    yield mask
    for i in range(d):
        for j in range(i+1, d):
            yield mask ^ (1 << i) ^ (1 << j)


def joint_nearest_supports(masks, d, k, *, capacity=256, stop_event=None,
                          minimum_total_distance=0, weight_bounds=None):
    if (type(d) is not int or d not in (3, 5, 11, 43) or len(masks) != 2
            or any(type(m) is not int or not 0 <= m < 1 << d for m in masks)
            or sum(bin(m).count('1') for m in masks) != d
            or type(k) is not int or not 1 <= k <= d//2
            or type(minimum_total_distance) is not int or not 0 <= minimum_total_distance <= d
            or type(capacity) is not int or not 1 <= capacity <= 4096):
        raise ValueError('valid support pair of total weight d, independent shift, capacity required')
    if weight_bounds is not None:
        if (len(weight_bounds) != 2 or any(len(bound) != 2 or
                any(type(v) is not int for v in bound) or not 0 <= bound[0] <= bound[1] <= d
                for bound in weight_bounds)):
            raise ValueError('two inclusive integer weight intervals required')
        if any(not low <= bin(mask).count('1') <= high
               for mask, (low, high) in zip(masks, weight_bounds)):
            raise ValueError('reference support weights must satisfy intervals')
    started = perf_counter()
    slices = []
    winners = []
    minimum = None
    pairs = set()
    complete_nearest = True

    def result(complete_scan):
        return dict(masks=sorted(pairs), minimum_distance=minimum, slices=slices,
            minimum_total_distance=minimum_total_distance,
            scan_complete=complete_scan, complete_nearest=complete_scan and complete_nearest,
            nearest_occurrences=sum(s['nearest_ties'] for s in winners),
            weight_matches=sum(s['weight_matches'] for s in slices), elapsed=perf_counter()-started)

    for fixed in (0, 1):
        neighborhood = relocated_supports if weight_bounds is None else two_bit_supports
        for fixed_mask in neighborhood(masks[fixed], d):
            if stop_event is not None and stop_event.is_set():
                return result(False)
            weight = bin(fixed_mask).count('1')
            other_weight = d-weight
            if weight_bounds is not None and any(not low <= w <= high
                    for w, (low, high) in zip((weight, other_weight) if fixed == 0 else
                                             (other_weight, weight), weight_bounds)):
                continue
            family = parity_support_family(fixed_mask, d, k)
            fixed_distance = bin(fixed_mask ^ masks[fixed]).count('1')
            floor = max(0, minimum_total_distance-fixed_distance)
            # One literal mask suffices to certify each slice's minimum.
            out = nearest_supports(family, masks[1-fixed], other_weight,
                                   capacity=1, minimum_distance=floor)
            distance = None if out['distance'] is None else out['distance']+fixed_distance
            record = dict(fixed_side=fixed, fixed_mask=fixed_mask, distance=distance,
                          nearest_ties=out['nearest_ties'], weight_matches=out['weight_matches'])
            slices.append(record)
            if distance is None:
                continue
            if minimum is None or distance < minimum:
                minimum = distance
                winners = []
            if distance == minimum:
                winners.append(record)
    # Second pass decodes literal ties only in slices attaining the minimum.
    # Occurrences can count a pair twice (both directions); no PACF quotient.
    for winner in winners:
        if stop_event is not None and stop_event.is_set():
            complete_nearest = False
            return result(True)
        fixed = winner['fixed_side']
        family = parity_support_family(winner['fixed_mask'], d, k)
        floor = max(0, minimum_total_distance-bin(winner['fixed_mask'] ^ masks[fixed]).count('1'))
        out = nearest_supports(family, masks[1-fixed], d-bin(winner['fixed_mask']).count('1'),
                               capacity=capacity, minimum_distance=floor)
        complete_nearest &= out['complete_nearest']
        for other in out['masks']:
            pair = (winner['fixed_mask'], other) if fixed == 0 else (other, winner['fixed_mask'])
            if len(pairs) < capacity or pair in pairs:
                pairs.add(pair)
            else:
                complete_nearest = False
        if len(pairs) == capacity:
            # Conservatively report incomplete unless the remaining union is
            # audited. No unexamined tied mask is labeled duplicate/UNSAT.
            complete_nearest = False
            break
    return result(True)


def repair_joint_cyclic_supports(a, b, target, *, max_supports=256,
                                 max_assignments=2**23, stop_event=None,
                                 minimum_total_distance=0, content_changes=False):
    started = perf_counter()
    a, b, d, wanted = _validated_reference(a, b, target)
    if type(max_assignments) is not int or not 1 <= max_assignments <= 2**31:
        raise ValueError('max_assignments must be in 1..2**31')
    if type(content_changes) is not bool:
        raise ValueError('content_changes must be boolean')
    k = next(u for u in range(1, d//2+1) if wanted[u] & 1)
    counts = ((target.a_even_ones, target.a_odd_ones), (target.b_even_ones, target.b_odd_ones))
    bounds = tuple((abs(d-e-o), d-abs(e-o)) for e, o in counts) if content_changes else None
    generation = joint_nearest_supports(tuple(support_mask(w) for w in (a, b)), d, k,
        capacity=max_supports, stop_event=stop_event, minimum_total_distance=minimum_total_distance,
        weight_bounds=bounds)
    attempts = []
    for masks in generation['masks']:
        if stop_event is not None and stop_event.is_set():
            break
        references = (a, b)
        if content_changes:
            # Only the symbol counts enter sign completion. A reference with
            # these counts is a carrier for the existing validated API.
            # w=2n+(d-e-o); both parity weights remain exactly prescribed.
            references = []
            for mask, (e, o) in zip(masks, counts):
                weight = bin(mask).count('1')
                n = (weight-(d-e-o))//2
                references.append(tuple([-1]*n+[0]*(d-weight)+[1]*(weight-n)))
        out = complete_support_signs(*references, target, masks, max_assignments=max_assignments)
        attempts.append(dict(masks=masks, result=out))
        if out['words'] is not None:
            return dict(status='SAT_COMPRESSED', words=out['words'], generation=generation,
                        attempts=attempts, elapsed=perf_counter()-started, layer_exhausted=False)
    unknown = (not generation['scan_complete'] or (stop_event is not None and stop_event.is_set())
               or any(a['result']['status'] == 'UNKNOWN' for a in attempts))
    return dict(status='UNKNOWN' if unknown else 'BOUNDED_MISS', words=None, generation=generation,
        attempts=attempts, elapsed=perf_counter()-started,
        layer_exhausted=not unknown and generation['complete_nearest']
            and len(attempts) == len(generation['masks'])
            and all(a['result']['status'] == 'UNSAT_FIXED_SUPPORT' for a in attempts))
