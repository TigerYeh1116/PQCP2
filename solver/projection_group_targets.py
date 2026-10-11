"""Normalize full target orbits so opposite G targets share one C problem.

For L=2d with d odd, the even orbit represented by 2r may instead use
d-r. An odd unit of Z/(2d) transfers it, preserving parity contents.
The odd representative r and d-r then have the SAME folded C profile.
"""
from collections import OrderedDict
from dataclasses import replace
from math import gcd

from .structured_energy import fold_profile,target_profile


def group_projection_targets(choices):
    groups=OrderedDict()
    for target,contents in choices:
        d=target.L//2
        if target.L!=2*d or not d%2:raise ValueError('odd half-length targets required')
        normalized=target;unit=1
        if not target.k%2:
            desired=d-gcd(d,target.k)
            unit=next((q for q in range(1,target.L,2)
                       if gcd(q,target.L)==1 and q*target.k%target.L==desired),None)
            if unit is None:raise RuntimeError('target unit orbit normalization failed')
            normalized=replace(target,k=desired)
        folded=tuple(v//4 for v in fold_profile(target_profile(target.L,normalized.k,normalized.eta),2))
        key=(folded,tuple(tuple(c) for c in contents))
        if key not in groups:groups[key]=dict(contents=key[1],targets=[],original_targets=[],target_units=[])
        group=groups[key]
        if normalized in group['targets']:raise ValueError('duplicate normalized target/content case')
        group['targets'].append(normalized);group['original_targets'].append(target);group['target_units'].append(unit)
    return tuple(dict(contents=g['contents'],targets=tuple(g['targets']),original_targets=tuple(g['original_targets']),
                      target_units=tuple(g['target_units'])) for g in groups.values())
