"""Low-energy truncated C-sign search (csrc/truncated_signs.c).

C solutions concentrate on C_A sign patterns with small sum_u R_A(u)^2: at L62 the
lowest 100 of 7690 patterns per support carry 53% of all C witnesses (41x the work
efficiency of exhaustive per-pair enumeration, 0 false positives). For one C_A
support this keeps the M lowest-energy sign patterns, walks one C_B support per
cyclic-rotation orbit of the GF(2) family and completes C_B exactly per kept
pattern. Supported: odd prime d whose X^d+1 has exactly one reciprocal pair of
nontrivial factors (d = 23, 47). Witnesses are exact; the truncation only
decides which patterns are searched, never excludes anything.
"""
import ctypes
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from .cyclic_support import field_mul, field_pow, poly_divmod, poly_mul
from .gf2_support_family import _reverse, gf2_support_ring


@lru_cache(maxsize=1)
def _library():
    root = Path(__file__).resolve().parents[1]
    source = root/'csrc/truncated_signs.c'
    build = root/'build'
    build.mkdir(exist_ok=True)
    suffix = '.dylib' if sys.platform == 'darwin' else '.so'
    path = build/('truncated_signs_'+hashlib.sha256(source.read_bytes()).hexdigest()[:16]+suffix)
    if not path.exists():
        fd, temporary = tempfile.mkstemp(dir=build, suffix=suffix)
        os.close(fd)
        try:
            subprocess.run(['cc', '-O3', '-std=c11', '-Wall', '-Wextra', '-Werror', '-fPIC',
                            '-dynamiclib' if sys.platform == 'darwin' else '-shared', str(source), '-o', temporary],
                           check=True, capture_output=True, timeout=60)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    lib = ctypes.CDLL(str(path))
    i32, u32, u64 = ctypes.c_int, ctypes.c_uint32, ctypes.c_uint64
    P = ctypes.POINTER
    lib.pqcp_top_signs.argtypes = [i32, u64, i32, i32, P(i32), P(u64), P(u64)]
    lib.pqcp_top_signs.restype = i32
    lib.pqcp_truncated_scan.argtypes = [i32, i32, i32, P(i32), i32, P(i32), i32, u32, u32, u32, u32, u64, u64,
                                        P(u64), P(u64), P(u64), P(u32), P(u64), i32, P(u64)]
    lib.pqcp_truncated_scan.restype = i32
    return lib


def _prime_factors(n):
    out, p = [], 2
    while p*p <= n:
        if n % p == 0:
            out.append(p)
            while n % p == 0:
                n //= p
        p += 1
    return out+([n] if n > 1 else [])


@lru_cache(maxsize=4)
def family_constants(d):
    """f, its generator gamma, and the linear maps K_f -> masks for the two CRT components."""
    ring = gf2_support_ring(d)
    if len(ring.factors) != 3 or ring.partner[1] != 2:
        raise ValueError('exactly one reciprocal pair of factors required (d = 23, 47)')
    f, g, m = ring.factors[1], ring.factors[2], ring.m
    e_f, e_g, modulus = ring.idempotents[1], ring.idempotents[2], (1 << d) | 1
    order = (1 << m)-1
    if order % d:
        raise RuntimeError('rotation subgroup must divide the multiplicative group')
    lf = [poly_divmod(poly_mul(1 << b, e_f), modulus)[1] for b in range(m)]
    def other(b):  # V_g with rho(V_g) = b
        return poly_divmod(_reverse(poly_divmod(poly_mul(b, e_f), modulus)[1], d), g)[1]
    lg = [poly_divmod(poly_mul(other(1 << b), e_g), modulus)[1] for b in range(m)]
    primes = _prime_factors(order)
    gamma = next(c for c in range(2, 1 << m) if all(field_pow(c, order//p, f) != 1 for p in primes))
    return dict(d=d, m=m, f=f, gamma=gamma, gamma_inv=field_pow(gamma, order-1, f), reps=order//d,
                lf=tuple(lf), lg=tuple(lg), e0=ring.idempotents[0])


def family_target(d, support_a, k):
    """(r_f, base) for U = support_a; r_f = 0 (probability 2^-m) is the unsupported zero branch."""
    c = family_constants(d)
    modulus = (1 << d) | 1
    desired = 1 ^ (1 << k) ^ (1 << (d-k))
    r = desired ^ poly_divmod(poly_mul(support_a, _reverse(support_a, d)), modulus)[1]
    base = (1 ^ bin(support_a).count('1') & 1)*c['e0']
    return poly_divmod(r, c['f'])[1], base


def top_signs(d, support_a, negatives, keep):
    lib, H = _library(), d//2
    R = (ctypes.c_int*(keep*H))()
    neg = (ctypes.c_uint64*keep)()
    stats = (ctypes.c_uint64*1)()
    kept = lib.pqcp_top_signs(d, support_a, negatives, keep, R, neg, stats)
    if kept < 1:
        raise RuntimeError('top sign selection failed: {}'.format(kept))
    return [tuple(R[s*H:(s+1)*H]) for s in range(kept)], [neg[s] for s in range(kept)], stats[0]


def truncated_scan(d, support_a, k, weight_b, negatives_b, wanted, patterns, max_out=4096):
    """Exact C witnesses (S_B, pattern index, C_B negative mask), one per rotation orbit of S_B."""
    c = family_constants(d)
    rf, base = family_target(d, support_a, k)
    if not rf:
        return None, None
    lib, H, M = _library(), d//2, len(patterns)
    T = (ctypes.c_int*H)(*(wanted[u] for u in range(1, H+1)))
    RA = (ctypes.c_int*(M*H))(*(v for p in patterns for v in p))
    LF = (ctypes.c_uint64*c['m'])(*c['lf'])
    LG = (ctypes.c_uint64*c['m'])(*c['lg'])
    out_sb, out_sig, out_x = (ctypes.c_uint64*max_out)(), (ctypes.c_uint32*max_out)(), (ctypes.c_uint64*max_out)()
    stats = (ctypes.c_uint64*6)()
    found = lib.pqcp_truncated_scan(d, weight_b, negatives_b, T, M, RA, c['m'], c['f'], c['gamma'], c['gamma_inv'],
                                    rf, c['reps'], base, LF, LG, out_sb, out_sig, out_x, max_out, stats)
    if found < 0:
        raise RuntimeError('truncated scan failed: {}'.format(found))
    hits = [(out_sb[i], out_sig[i], out_x[i]) for i in range(min(found, max_out))]
    return hits, dict(reps=stats[0], weight_matches=stats[1], evaluations=stats[2], consistent=stats[3],
                      candidates=stats[4], wide_kernel_skipped=stats[5], found=found)
