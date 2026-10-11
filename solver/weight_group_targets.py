"""Exact row exchange groups targets sharing a support-generation domain.

This groups scheduling domains, not C sign equations: different signed
profiles and negative counts remain separate problems. Every original
target retains its inverse row exchange and unit-decimation provenance.
"""
from collections import OrderedDict
from dataclasses import replace

from .structured_energy import fold_profile,target_profile


def group_weight_targets(projection_groups):
    grouped=OrderedDict()
    for index,group in enumerate(projection_groups):
        contents=tuple(tuple(c) for c in group['contents'])
        if len(contents)!=2 or any(len(c)!=3 for c in contents):
            raise ValueError('two ternary content triples required')
        weights=tuple(n+p for n,z,p in contents)
        swapped=weights[0]>weights[1]
        if swapped:contents=contents[::-1];weights=weights[::-1]
        targets=[]
        for original in group['targets']:
            t=(replace(original,a_even_ones=original.b_even_ones,a_odd_ones=original.b_odd_ones,
                       b_even_ones=original.a_even_ones,b_odd_ones=original.a_odd_ones) if swapped else original)
            if t.L%2 or not (t.L//2)%2:raise ValueError('odd half-length target required')
            targets.append(t)
        if not targets:raise ValueError('nonempty projection group required')
        parities=tuple(tuple((v//4)%2 for v in fold_profile(target_profile(t.L,t.k,t.eta),2)) for t in targets)
        if any(p!=parities[0] for p in parities):raise ValueError('different support norm parities cannot share a family')
        key=(targets[0].L,weights,parities[0])
        entry=dict(projection_index=index,row_swapped=swapped,contents=contents,targets=tuple(targets),
                   original_projection_group=group)
        if key not in grouped:grouped[key]=dict(weights=weights,support_norm_parity=parities[0],c_groups=[])
        grouped[key]['c_groups'].append(entry)
    return tuple(dict(weights=g['weights'],support_norm_parity=g['support_norm_parity'],
                      c_groups=tuple(g['c_groups'])) for g in grouped.values())
