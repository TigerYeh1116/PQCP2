"""Complete radius-four repair at fixed binary even/odd contents."""
import ctypes
from functools import lru_cache
import hashlib
from math import comb
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from time import perf_counter

from .correlation import normalize_binary_sequence
from .exact_swap_repair import exact_two_row_swap_repair
from .interleaving import periodic_sign_profile
from .structured_energy import target_profile
from .target_profiles import pair_content, target_content_profiles
from .verifier import verify_pqcp


@lru_cache(maxsize=1)
def _library():
    root = Path(__file__).resolve().parents[1]
    source = root/'csrc/hamming4_repair.c'
    build = root/'build'
    build.mkdir(exist_ok=True)
    suffix = '.dylib' if sys.platform == 'darwin' else '.so'
    path = build/('hamming4_'+hashlib.sha256(source.read_bytes()).hexdigest()[:16]+suffix)
    if not path.exists():
        fd, temporary = tempfile.mkstemp(dir=build, suffix=suffix)
        os.close(fd)
        try:
            subprocess.run(['cc','-O3','-std=c11','-Wall','-Wextra','-Werror','-fPIC',
                '-dynamiclib' if sys.platform == 'darwin' else '-shared',str(source),'-o',temporary],
                check=True,capture_output=True,timeout=30)
            os.replace(temporary,path)
        finally:
            if os.path.exists(temporary): os.unlink(temporary)
    library = ctypes.CDLL(str(path))
    u, i = ctypes.c_uint64, ctypes.c_int
    library.pqcp_row_four_flips.argtypes = [u,u,i,ctypes.POINTER(i),u,u,ctypes.POINTER(u),ctypes.POINTER(u)]
    library.pqcp_row_four_flips.restype = i
    return library


def row_four_flip_search(row, other, targets):
    row, other = tuple(normalize_binary_sequence(row)), tuple(normalize_binary_sequence(other))
    length = len(row)
    targets = tuple(targets)
    if (length != len(other) or not 4 <= length <= 94 or length % 2
            or any(p.L != length or not 1 <= p.k < length//2 or p.eta not in (-1,1) for p in targets)):
        raise ValueError('valid equal even binary rows and target positions required')
    positive = sum(1 << k for k in {p.k for p in targets if p.eta == 1})
    negative = sum(1 << k for k in {p.k for p in targets if p.eta == -1})
    mask = sum(v << i for i,v in enumerate(row))
    profile = periodic_sign_profile(tuple(1-2*v for v in other))
    values = (ctypes.c_int*length)(*profile)
    output, stats = (ctypes.c_uint64*4)(), (ctypes.c_uint64*2)()
    status = _library().pqcp_row_four_flips(mask & ((1<<64)-1),mask>>64,length,values,
        positive,negative,output,stats)
    if status not in (0,1): raise RuntimeError('native four-flip search failed')
    result = dict(status='SAT' if status else 'UNSAT_NEIGHBORHOOD', configurations=int(stats[0]),
                  correlation_checks=int(stats[1]), word=None)
    if status:
        changed = int(output[0])+(int(output[1])<<64)
        result.update(word=tuple((changed>>i)&1 for i in range(length)),k=int(output[2]),eta=1 if output[3] else -1)
    else:
        groups = [row[parity::2] for parity in (0,1)]
        counts = [(g.count(0),g.count(1)) for g in groups]
        expected = sum(comb(a,2)*comb(b,2) for a,b in counts)+counts[0][0]*counts[0][1]*counts[1][0]*counts[1][1]
        if result['configurations'] != expected:
            raise RuntimeError('native four-flip search did not cover complete literal neighborhood')
    return result


def exact_hamming4_repair(a,b,*,targets=None):
    tick = perf_counter()
    a,b = tuple(normalize_binary_sequence(a)),tuple(normalize_binary_sequence(b))
    if len(a) != len(b) or len(a) < 4 or len(a)>94 or len(a)%2:
        raise ValueError('equal even lengths 4..94 required')
    content = pair_content(a,b)
    feasible = tuple(p for p in target_content_profiles(len(a),decimation_reduced=False)
        if (p.a_even_ones,p.a_odd_ones,p.b_even_ones,p.b_odd_ones) == content)
    targets = feasible if targets is None else tuple(targets)
    if len(set(targets)) != len(targets) or any(p not in feasible for p in targets):
        raise ValueError('distinct targets preserving the center contents required')
    joined = exact_two_row_swap_repair(a,b,targets=targets)
    rows = []
    witnesses = list(joined['witnesses'])
    if not witnesses:
        for side in (0,1):
            pair = (a,b)
            out = row_four_flip_search(pair[side],pair[1-side],targets)
            rows.append(out)
            if out['word'] is not None:
                words = (out['word'],b) if side == 0 else (a,out['word'])
                check = verify_pqcp(*words)
                if (not check.is_valid or tuple(check.profile) != target_profile(len(a),out['k'],out['eta'])
                        or pair_content(*words) != content or sum(x!=y for old,new in zip(pair,words) for x,y in zip(old,new)) != 4):
                    raise RuntimeError('native four-flip hit failed independent full verification')
                witnesses.append(dict(words=words,k=out['k'],sign=out['eta']))
                break
    return dict(status='SAT' if witnesses else 'UNSAT_RADIUS4_FIXED_CONTENT',witnesses=witnesses,
        complete=not witnesses,meet_in_middle=joined,row_four_flips=rows,elapsed=perf_counter()-tick,
        scope='ALL selected-target pairs at total Hamming distance <=4 with the same four parity contents; no exclusion beyond this ball.')
