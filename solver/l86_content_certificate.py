"""Finite coarse support bounds: runtime >=7, authenticated completed tier >=8.

The analytic parity bound first gives weight >=5. Ordinary-character
squares force absolute row sums (6,3) for eta=+1 or (5,4) for eta=-1.
Weights 5/6 therefore have just four sign-count cases. Enumerate ALL
weight-5/6 bracelets and all full-parity partner preimages, then exhaust
the exact sign systems. Rotation/reversal/complementation and common unit
decimation transfer the result to all positions, signs and independent k.
If any case is SAT/UNKNOWN, keep the weaker analytic bound: never guess.
Completed weight-7 coarse records plus an independently checked exceptional
leaf extend the bound to 8; absent or altered records retain the runtime tier.
"""
from dataclasses import dataclass
from functools import lru_cache
from math import comb
import hashlib
import json
from pathlib import Path

from .sparse_supports import sparse_bracelets
from .ternary_signs_native import solve_sign_system
from .ternary_support_search import support_sign_system


@dataclass(frozen=True)
class ContentCertificate:
    verified: bool
    minimum_weight: int
    cases: tuple
    assignments: int
    recorded_cases: tuple = ()


WEIGHT7_RECORDS = (
    ('L86_sparse_coarse_n1_negative_verified_20261007/report_completed.json',
     '8002b01ffdd779b0fcb3b5c39782d11062736b7c1abfb8da95a0ab768d8beff4', -1, (1, 16), (17, 21, 17, 22)),
    ('L86_sparse_coarse_n15_positive_20261007/report.json',
     'b43e30234733835de96222baf6433c6565e5635f07eae1d0a23812d0fa01a459', 1, (15, 2), (17, 20, 17, 23)),
)


def recorded_weight7_cases(root=None):
    """Authenticated complete coarse runs; missing/modified data returns no tier.

    Digests authenticate observed exhaustive evidence, not independent UNSAT
    certificates. The one initially UNKNOWN leaf is additionally checked by
    an independent Python GF2 projection and literal one-negative patterns.
    """
    root = Path(__file__).resolve().parents[1] if root is None else Path(root)
    records = []
    for name, digest, eta, negative, counts in WEIGHT7_RECORDS:
        try:
            raw = (root/'results'/name).read_bytes()
            if hashlib.sha256(raw).hexdigest() != digest:
                return ()
            data = json.loads(raw)
            s = data['summary']
            target = dict(L=86, k=2, eta=eta, a_even_ones=counts[0], a_odd_ones=counts[1],
                          b_even_ones=counts[2], b_odd_ones=counts[3])
            g = s['bracelet_generation']
            if (s['complete'] is not True or s['completion_scope'] != 'compressed' or s['target'] != target
                    or s['restricted_family'] != 'n_A={}, k=2, eta={:+d}'.format(negative[0], eta)
                    or s['tested'] != 130142 or s['matched_pairs'] != 130142
                    or s['statuses'] != {'UNSAT_FIXED_SUPPORT': 130142}
                    or s['coarse_verified'] != 0 or s['verified'] != 0 or s['new'] != 0
                    or g['complete'] is not True or g['total'] != 375364 or g['candidates'] != comb(42, 6)):
                return ()
            if eta == -1:
                if s.get('refinements') != 1 or len(data['rows']) != 1:
                    return ()
                row = data['rows'][0]
                masks = (row['u'], ((1 << 43)-1)^row['w'])
                wanted = (43, 0, -1)+(0,)*38+(-1, 0)
                from .affine_weight import condition_sign_row
                system = support_sign_system(masks, 43, wanted, negative)
                if not system.consistent:
                    return ()
                projected = condition_sign_row(system, 0)
                if len(projected.pattern_variables) != 7 or any(
                    all((bin((1 << i)&mask).count('1') & 1) == rhs for mask, rhs in projected.checks)
                    for i in range(7)):
                    return ()
                if (row['initial_result']['status'] != 'UNKNOWN' or row['result']['status'] != 'UNSAT_FIXED_SUPPORT'
                        or row['result']['attempts']['coarse']['assignments'] != 0):
                    return ()
        except (OSError, ValueError, KeyError, TypeError):
            return ()
        records.append((7, eta, negative, 130142))
    return tuple(records)


@lru_cache(maxsize=1)
def l86_content_certificate():
    cases = []
    assignments = 0
    full = (1 << 43)-1
    parity_target = ((1 << 21)-1) ^ 2
    for weight in (5, 6):
        generated = sparse_bracelets(43, weight, capacity=comb(43, weight)//43)
        expected = (comb(43, weight)//43+comb(21, weight//2))//2
        if not generated['complete'] or generated['total'] != expected:
            return ContentCertificate(False, 5, tuple(cases), assignments)
        index = {}
        for mask, signature in generated['records']:
            index.setdefault(signature, []).append(mask)
        # Sparse large-sum positive row, or sparse small-sum row with one
        # negative. The latter is encoded in B using complement supports.
        for large in (False, True):
            eta = (-1 if weight == 5 else 1)*(-1 if large else 1)
            negative = (16, 1) if large else (0, 17)
            wanted = tuple(43 if u == 0 else eta if u in (2, 41) else 0 for u in range(43))
            tested = 0
            for u, signature in generated['records']:
                for w in index.get(signature ^ parity_target, ()):
                    masks = (full ^ u, w) if large else (u, full ^ w)
                    system = support_sign_system(masks, 43, wanted, negative)
                    tested += 1
                    if not system.consistent:
                        continue
                    out = solve_sign_system(system, masks, 43, wanted, negative, 2**23)
                    assignments += out['assignments']
                    if out['status'] != 'UNSAT_FIXED_SUPPORT':
                        return ContentCertificate(False, 5, tuple(cases), assignments)
            cases.append((weight, eta, negative, tested))
    recorded = recorded_weight7_cases()
    return ContentCertificate(True, 8 if recorded else 7, tuple(cases), assignments, recorded)
