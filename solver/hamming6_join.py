"""Exact streamed 4+2 and 2+4 flip repair at fixed parity contents.

Enumerate the larger row-four neighborhood once. Join exact integer PACF
delta vectors to target residual minus the smaller row-two deltas. All
preimages remain available; a miss excludes only these selected neighborhoods.
The one-row-six neighborhoods are NOT covered by this method.
"""
from collections import defaultdict
import ctypes
from math import comb
from time import perf_counter

from .correlation import full_correlation_profile,normalize_binary_sequence
from .exact_swap_repair import row_swap_deltas
from .hamming4_repair import _library
from .interleaving import periodic_sign_profile
from .structured_energy import target_profile
from .target_profiles import pair_content,target_content_profiles
from .verifier import verify_pqcp


def row_four_flip_count(row):
    groups = [row[p::2] for p in (0,1)]
    counts = [(g.count(0),g.count(1)) for g in groups]
    return sum(comb(a,2)*comb(b,2) for a,b in counts)+counts[0][0]*counts[0][1]*counts[1][0]*counts[1][1]


def row_four_vectors(row,*,batch_size=8192):
    row = tuple(normalize_binary_sequence(row))
    length = len(row)
    if not 4 <= length <= 94 or length%2 or type(batch_size) is not int or not 1 <= batch_size <= 65536:
        raise ValueError('valid even binary row and bounded stream size required')
    mask = sum(v<<i for i,v in enumerate(row))
    profile = periodic_sign_profile(tuple(1-2*v for v in row))
    base = (ctypes.c_int*length)(*profile)
    function = _library().pqcp_row_four_vectors
    u = ctypes.c_uint64
    function.argtypes = [u,u,ctypes.c_int,ctypes.POINTER(ctypes.c_int),u,ctypes.c_int,
                        ctypes.POINTER(u),ctypes.POINTER(ctypes.c_ubyte),ctypes.POINTER(u)]
    function.restype = ctypes.c_int
    masks = (u*(2*batch_size))()
    vectors = (ctypes.c_ubyte*(batch_size*(length//2)))()
    stats = (u*2)()
    view = memoryview(vectors).cast('B')
    total, width = row_four_flip_count(row),length//2
    for offset in range(0,total,batch_size):
        expected = min(batch_size,total-offset)
        status = function(mask & ((1<<64)-1),mask>>64,length,base,offset,batch_size,masks,vectors,stats)
        if status not in (0,1) or stats[0] != expected:
            raise RuntimeError('native four-flip stream failed complete enumeration')
        for i in range(expected):
            yield int(masks[2*i])+(int(masks[2*i+1])<<64),bytes(view[i*width:(i+1)*width])


def exact_four_two_join(a,b,*,targets=None,stop_event=None):
    tick = perf_counter()
    a,b = tuple(normalize_binary_sequence(a)),tuple(normalize_binary_sequence(b))
    length = len(a)
    if len(b)!=length or not 4 <= length <= 94 or length%2:
        raise ValueError('equal even binary lengths 4..94 required')
    content = pair_content(a,b)
    feasible = tuple(p for p in target_content_profiles(length,decimation_reduced=False) if
        (p.a_even_ones,p.a_odd_ones,p.b_even_ones,p.b_odd_ones)==content)
    targets = feasible if targets is None else tuple(targets)
    if len(set(targets)) != len(targets) or any(p not in feasible for p in targets):
        raise ValueError('distinct compatible targets required')
    original = tuple(full_correlation_profile(a,b))
    witnesses, rows, seen, cancelled = [],[],set(),False
    for side in (0,1):
        pair = (a,b)
        small = row_swap_deltas(pair[1-side])
        needed = defaultdict(list)
        for target in targets:
            wanted = target_profile(length,target.k,target.eta)
            residual = tuple(x-y for x,y in zip(wanted[1:length//2+1],original[1:length//2+1]))
            if any(v%4 for v in residual):continue
            residual = tuple(v//4 for v in residual)
            for delta,swaps in small.items():
                key = bytes(128+x-y for x,y in zip(residual,delta))
                needed[key].extend((target,swap) for swap in swaps if swap)
        tested = 0
        if needed:
            for changed,key in row_four_vectors(pair[side]):
                if stop_event is not None and stop_event.is_set():
                    cancelled=True;break
                tested+=1
                for target,swap in needed.get(key,()):
                    large = tuple((changed>>i)&1 for i in range(length))
                    other = tuple(v ^ int(i in swap) for i,v in enumerate(pair[1-side]))
                    words = (large,other) if side==0 else (other,large)
                    if words in seen:continue
                    check = verify_pqcp(*words)
                    if (not check.is_valid or tuple(check.profile)!=target_profile(length,target.k,target.eta)
                            or pair_content(*words)!=content
                            or sum(x!=y for old,new in zip(pair,words) for x,y in zip(old,new))!=6):
                        raise RuntimeError('four-two join hit failed independent integer full verifier')
                    seen.add(words)
                    witnesses.append(dict(words=words,k=target.k,sign=target.eta))
        rows.append(dict(four_flip_side=side,configurations=tested,expected=row_four_flip_count(pair[side]),
                         needed_vectors=len(needed),complete=not cancelled))
        if cancelled:break
    return dict(status='SAT' if witnesses else 'UNKNOWN' if cancelled else 'UNSAT_FOUR_TWO_NEIGHBORHOOD',
        witnesses=witnesses,rows=rows,complete=not cancelled,elapsed=perf_counter()-tick,
        scope='Exactly 4+2 or 2+4 binary flips, preserving four parity contents and selected targets. Does NOT include single-row six flips or the full L86 space.')
