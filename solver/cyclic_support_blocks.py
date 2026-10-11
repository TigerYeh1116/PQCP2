"""Exact support-only block joins, leaving ALL C/G sign coordinates free.

Enumerate every in-block binary support mask; match weight and full parity
PACF signatures. The condition is necessary for a full PQCP. Literal masks
with the same signature are all retained, not treated as equivalent.
"""
from collections import defaultdict
import ctypes
from dataclasses import dataclass
from time import perf_counter

from .cyclic_support_native import _library
from .ternary_support_search import support_signature


def support_parity_energy_lower_bound(masks, d, k):
    """Full binary energy >= number of mismatched independent parity lags.

    Requires antipodal supports of total C weight d and valid parity contents.
    Complementary G supports have the SAME unsigned PACF sum modulo two.
    Both C/G error norms are therefore >= this count, and E_full=(Ec+Eg)/2.
    This is a fixed-support bound, not an exclusion of other supports.
    """
    masks = tuple(masks)
    if (type(d) is not int or not 3 <= d <= 47 or d % 2 != 1 or len(masks) != 2
            or any(type(m) is not int or not 0 <= m < 1 << d for m in masks)
            or sum(bin(m).count('1') for m in masks) != d
            or type(k) is not int or not 1 <= k <= d//2):
        raise ValueError('odd supports of total weight d and independent target lag required')
    error = support_signature(masks[0], d)^support_signature(masks[1], d)^(1 << (k-1))
    return bin(error).count('1')


@dataclass(frozen=True)
class SupportBlockOptions:
    masks: tuple
    signatures: tuple
    weights: tuple
    covered_configurations: int


def support_block_options(reference,d,positions,allowed_weights):
    positions,allowed_weights=tuple(positions),tuple(allowed_weights)
    if (type(d) is not int or not 3<=d<=47 or d%2!=1 or type(reference) is not int
            or not 0<=reference<1<<d or not 1<=len(positions)<=min(d,20)
            or len(set(positions))!=len(positions) or any(type(p) is not int or not 0<=p<d for p in positions)
            or len(set(allowed_weights))!=len(allowed_weights)
            or any(type(w) is not int or not 0<=w<=d for w in allowed_weights)):
        raise ValueError('odd bounded length, literal positions and support weights required')
    total=1<<len(positions)
    u=ctypes.c_uint64
    masks,signatures,weights=(u*total)(),(ctypes.c_uint32*total)(),(ctypes.c_ubyte*total)()
    stats=(u*2)()
    function=_library().pqcp_support_block_options
    function.argtypes=[u,ctypes.c_int,ctypes.POINTER(ctypes.c_int),ctypes.c_int,u,
        ctypes.POINTER(u),ctypes.POINTER(ctypes.c_uint32),ctypes.POINTER(ctypes.c_ubyte),ctypes.POINTER(u)]
    function.restype=ctypes.c_int
    status=function(reference,d,(ctypes.c_int*len(positions))(*positions),len(positions),
        sum(1<<w for w in allowed_weights),masks,signatures,weights,stats)
    if status!=0 or stats[0]!=total:raise RuntimeError('native support block did not cover every literal configuration')
    count=int(stats[1])
    return SupportBlockOptions(tuple(masks[:count]),tuple(signatures[:count]),tuple(weights[:count]),total)


def join_support_blocks(masks,d,k,positions,weight_pairs,*,audit_all=False):
    tick=perf_counter()
    masks,positions,weight_pairs=tuple(masks),tuple(map(tuple,positions)),tuple(map(tuple,weight_pairs))
    if (len(masks)!=2 or len(positions)!=2 or type(k) is not int or not 1<=k<=d//2
            or any(len(p)!=2 or sum(p)!=d for p in weight_pairs) or type(audit_all) is not bool):
        raise ValueError('two blocks, an independent target lag and weights of total d required')
    choices=tuple(support_block_options(mask,d,ps,sorted({p[side] for p in weight_pairs}))
                  for side,(mask,ps) in enumerate(zip(masks,positions)))
    index=defaultdict(list)
    for mask,signature,weight in zip(choices[1].masks,choices[1].signatures,choices[1].weights):
        index[(weight,signature)].append(mask)
    pairs=[]
    allowed=set(weight_pairs)
    for first,signature,weight in zip(choices[0].masks,choices[0].signatures,choices[0].weights):
        if (weight,d-weight) not in allowed:continue
        for second in index.get((d-weight,signature ^ (1<<(k-1))),()):
            pairs.append((first,second))
    # Every matched pair is audited in the original cyclic support arithmetic.
    for first,second in pairs:
        if (support_signature(first,d) ^ support_signature(second,d))!=1<<(k-1):
            raise RuntimeError('support-block parity match failed independent integer audit')
    if audit_all:
        for choice in choices:
            for mask,signature,weight in zip(choice.masks,choice.signatures,choice.weights):
                if support_signature(mask,d)!=signature or bin(mask).count('1')!=weight:
                    raise RuntimeError('native Gray support signature failed independent full audit')
    return dict(masks=pairs,complete=True,row_retained_configurations=tuple(len(c.masks) for c in choices),
        covered_row_configurations=tuple(c.covered_configurations for c in choices),
        exact_signature_lookups=len(choices[0].masks),elapsed=perf_counter()-tick,
        scope='ALL in-block support masks with selected weight pairs and exact parity PACF; signs remain unrestricted. Not all global supports.')
