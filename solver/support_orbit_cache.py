"""Bounded exact UNSAT memoization under independent row dihedral actions.

For each row, rotate/reflecting a signed trit word preserves its entire PACF
and negative count. Its complementary support transforms in the same way.
Thus existence of BOTH C/G sign witnesses is invariant on these support
orbits, with the same target and contents. UNKNOWN/SAT are never exclusions.
"""
from collections import OrderedDict
from functools import lru_cache
from threading import RLock


@lru_cache(maxsize=8192)
def row_dihedral_mask(mask,d):
    if type(d) is not int or d<1 or type(mask) is not int or not 0<=mask<1<<d:
        raise ValueError('valid support mask and length required')
    reflected=sum(((mask>>((-i)%d))&1)<<i for i in range(d))
    all_bits=(1<<d)-1
    return min(((m<<u)|(m>>(d-u)))&all_bits for m in (mask,reflected) for u in range(d))


@lru_cache(maxsize=1)
def _canonical_l86_targets():
    from .target_profiles import canonical_target_content_profiles
    return frozenset(canonical_target_content_profiles(86))


class SupportOrbitCache:
    def __init__(self,capacity=512):
        if type(capacity) is not int or capacity<1:raise ValueError('positive bounded capacity required')
        self.capacity=capacity
        self.entries=OrderedDict()
        self.lock=RLock()

    def key(self,masks,target,negative_counts,*,modulation=False):
        masks,negative_counts=tuple(masks),tuple(negative_counts)
        base=(target,negative_counts,tuple(row_dihedral_mask(m,target.L//2) for m in masks))
        if type(modulation) is not bool:raise ValueError('modulation must be boolean')
        if not modulation or target.L!=86:return base
        from .alternation_orbits import alternate_canonical_profile
        if target not in _canonical_l86_targets():return base
        d=43
        counts=((target.a_even_ones,target.a_odd_ones),(target.b_even_ones,target.b_odd_ones))
        expected=tuple((bin(m).count('1')-d+e+o)//2 for m,(e,o) in zip(masks,counts))
        if (len(masks)!=2 or len(negative_counts)!=2 or sum(bin(m).count('1') for m in masks)!=d
                or expected!=negative_counts
                or any((bin(m).count('1')-d+e+o)%2 for m,(e,o) in zip(masks,counts))):
            raise ValueError('modulation key requires actual support/content signed counts')
        partner=alternate_canonical_profile(target)
        changed_masks=tuple(((1<<d)-1)^m for m in masks)
        changed_negative=tuple(e-n for (e,o),n in zip(counts,negative_counts))
        # These FOUR L86 canonical contents have explicit gauges: odd k
        # needs no row operation; even k requires exchanging the two rows.
        if target.k%2==0:
            changed_masks=changed_masks[::-1]
            changed_negative=changed_negative[::-1]
        other=(partner,changed_negative,tuple(row_dihedral_mask(m,d) for m in changed_masks))
        return min(base,other)

    def get(self,key):
        with self.lock:
            result=self.entries.get(key)
            if result is not None:self.entries.move_to_end(key)
            return result

    def remember(self,key,masks,excluded_by,*,target=None,negative_counts=None):
        with self.lock:
            # Store only immutable provenance, not mutable solver dictionaries,
            # witnesses, tensors or an UNSAT claim from an incomplete run.
            self.entries[key]=(tuple(masks),excluded_by,target,
                               tuple(negative_counts) if negative_counts is not None else None)
            self.entries.move_to_end(key)
            while len(self.entries)>self.capacity:self.entries.popitem(last=False)

    def clear(self):
        with self.lock:self.entries.clear()


FULL_SUPPORT_ORBITS=SupportOrbitCache()
