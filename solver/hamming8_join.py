"""Exact prefix-filtered 4+4 flip join, preserving all profile preimages.

The prefix must match at least one selected target. This is a necessary
integer condition, followed by checking EVERY independent PACF coordinate.
A miss excludes this finite 4+4 neighborhood only, not a full radius-8 ball.
"""
from collections import defaultdict
from time import perf_counter

from .correlation import full_correlation_profile,normalize_binary_sequence
from .hamming6_join import row_four_vectors
from .structured_energy import target_profile
from .target_profiles import pair_content,target_content_profiles
from .verifier import verify_pqcp


def exact_four_four_join(a,b,*,targets=None,prefix=12,stop_event=None,stop_after_hit=True):
    tick=perf_counter()
    a,b=tuple(normalize_binary_sequence(a)),tuple(normalize_binary_sequence(b))
    length=len(a)
    if (len(b)!=length or not 4<=length<=94 or length%2 or type(prefix) is not int
            or not 1<=prefix<=length//2 or type(stop_after_hit) is not bool):
        raise ValueError('equal bounded even rows and a valid prefix required')
    content=pair_content(a,b)
    feasible=tuple(p for p in target_content_profiles(length,decimation_reduced=False) if
        (p.a_even_ones,p.a_odd_ones,p.b_even_ones,p.b_odd_ones)==content)
    targets=feasible if targets is None else tuple(targets)
    if len(set(targets))!=len(targets) or any(p not in feasible for p in targets):
        raise ValueError('distinct compatible targets required')
    original=tuple(full_correlation_profile(a,b))
    if any(v%4 for v in original[1:length//2+1]):
        raise RuntimeError('binary pair PACF lattice failed')
    center=tuple(v//4 for v in original[1:length//2+1])
    allowed={(p.k,p.eta) for p in targets}
    patterns=set(tuple(p.eta if u==p.k else 0 for u in range(1,prefix+1)) for p in targets)
    index=defaultdict(lambda:defaultdict(list))
    witnesses,seen=[],set()
    built=tested=lookups=coordinate_checks=0

    def finish(complete):
        return dict(status='SAT' if witnesses else 'UNSAT_FOUR_FOUR_NEIGHBORHOOD' if complete else 'UNKNOWN',
            witnesses=witnesses,complete=complete,indexed_configurations=built,tested_configurations=tested,
            prefix_groups=len(index),prefix_patterns=len(patterns),exact_prefix_lookups=lookups,
            full_coordinate_checks=coordinate_checks,elapsed=perf_counter()-tick,
            scope='Exactly four flips per row preserving fixed parity contents and selected targets; NOT a complete radius-8 ball or global L86 exclusion.')
    if not targets:
        return finish(True)
    for mask,delta in row_four_vectors(a):
        if stop_event is not None and stop_event.is_set():return finish(False)
        index[delta[:prefix]][delta].append(mask)
        built+=1
    for mask_b,delta_b in row_four_vectors(b):
        if stop_event is not None and stop_event.is_set():return finish(False)
        tested+=1
        base=tuple(256-center[i]-delta_b[i] for i in range(prefix))
        for pattern in patterns:
            required=bytes(x+y for x,y in zip(base,pattern))
            lookups+=1
            for delta_a,masks_a in index.get(required,{}).items():
                found=None
                valid=True
                for i,(x,y,z) in enumerate(zip(delta_a,delta_b,center),1):
                    value=x+y-256+z
                    coordinate_checks+=1
                    if not value:continue
                    if found is not None or (i,value) not in allowed:
                        valid=False;break
                    found=(i,value)
                if not valid or found is None:continue
                k,eta=found
                for mask_a in masks_a:
                    words=tuple(tuple((m>>i)&1 for i in range(length)) for m in (mask_a,mask_b))
                    if words in seen:continue
                    check=verify_pqcp(*words)
                    if (not check.is_valid or tuple(check.profile)!=target_profile(length,k,eta)
                            or pair_content(*words)!=content
                            or tuple(sum(x!=y for x,y in zip(old,new)) for old,new in zip((a,b),words))!=(4,4)):
                        raise RuntimeError('four-four join hit failed independent full integer verifier')
                    seen.add(words);witnesses.append(dict(words=words,k=k,sign=eta))
                    if stop_after_hit:return finish(False)
    return finish(True)
