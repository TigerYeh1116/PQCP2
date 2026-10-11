"""Recorded exhaustive binary-content proofs, scoped to canonical L86 targets.

All four n_A=1 subfamilies were exhausted by the exact two-problem solver.
Odd original alternating modulation exchanges coarse/difference rows and
maps n_A to 16-n_A, proving odd n_A=15 infeasible as well.
Digests authenticate the observed complete run records; they are NOT an
independent UNSAT proof checker. The mathematical basis is the complete
bracelet generation and exhaustive sign algorithms described in docs.
Missing/modified evidence disables this stronger pruning. Reproduction
must set binary_content_pruning=False to avoid circular use of these results.
"""
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from math import comb
from pathlib import Path


RECORDS = (
    ('L86_sparse_n1_negative_20261007', '2e56717c6ef8298c71fdcaad9661d8807ce8f7f64f191b4b3696630ea6536f54',
     2, -1, (17, 21, 17, 22), 130142, 7),
    ('L86_sparse_n1_negative_odd_20261007', '548ea40632e58588e1c229588318f6371172d9612deadab136c37d48dd51c2a0',
     41, -1, (16, 22, 18, 21), 130142, 7),
    ('L86_sparse_n1_positive_20261007', '6f7603a510123f807415a89d6cf6b6ca3901d889055ed9509c0f2f53c5620ac0',
     2, 1, (17, 20, 17, 23), 2716258, 8),
    ('L86_sparse_n1_positive_odd_native_20261007', '3830503ed214f8e9b1503014e7e6ba9cf0effa2db1eaeecd12d987e975f046eb',
     41, 1, (16, 21, 18, 22), 2716258, 8),
)


@dataclass(frozen=True)
class RecordedBinaryContentProof:
    verified: bool
    cases: int
    sources: tuple
    reason: str


def read_recorded_binary_content_proof(root):
    cases, sources = 0, []
    for name, digest, k, eta, counts, expected, weight in RECORDS:
        path = Path(root)/'results'/name/'report.json'
        try:
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != digest:
                return RecordedBinaryContentProof(False, cases, tuple(sources), 'record digest mismatch')
            summary = json.loads(raw)['summary']
            target = dict(L=86, k=k, eta=eta, a_even_ones=counts[0], a_odd_ones=counts[1],
                          b_even_ones=counts[2], b_odd_ones=counts[3])
            generation = summary['bracelet_generation']
            bracelets = (comb(43, weight)//43+comb(21, weight//2))//2
            if (summary['target'] != target or summary['complete'] is not True
                    or summary['restricted_family'] != 'n_A=1, k={}, eta={:+d}'.format(k, eta)
                    or summary['tested'] != expected or summary['matched_pairs'] != expected
                    or summary['statuses'] != {'UNSAT_BINARY_SUPPORT': expected}
                    or summary['verified'] != 0 or summary['new'] != 0
                    or generation['complete'] is not True or generation['total'] != bracelets
                    or generation['candidates'] != comb(42, weight-1)
                    or sum(summary['excluded_by'].values()) != expected):
                return RecordedBinaryContentProof(False, cases, tuple(sources), 'record coverage/scope mismatch')
        except (OSError, ValueError, KeyError, TypeError):
            return RecordedBinaryContentProof(False, cases, tuple(sources), 'record unavailable or malformed')
        cases += expected
        sources.append(str(path))
    return RecordedBinaryContentProof(True, cases, tuple(sources), 'authenticated complete finite enumeration')


@lru_cache(maxsize=1)
def l86_binary_content_proof():
    return read_recorded_binary_content_proof(Path(__file__).resolve().parents[1])


def binary_content_bounds(target):
    """Return additional per-row negative-count intervals, or no restriction.

    The interval shortcut requires the separate support>=7 theorem; without
    it n_A=0/16 could remain and exclusion of isolated points is not an interval.
    Canonical numeric contents only. General public profiles stay unchanged.
    """
    if target.L != 86 or target.eta not in (-1, 1) or not 0 < target.k < 43:
        return None
    counts = (target.a_even_ones, target.a_odd_ones, target.b_even_ones, target.b_odd_ones)
    expected = {(0, -1): (17, 21, 17, 22), (0, 1): (17, 20, 17, 23),
                (1, -1): (16, 22, 18, 21), (1, 1): (16, 21, 18, 22)}
    if counts != expected[target.k % 2, target.eta]:
        return None
    from .l86_content_certificate import l86_content_certificate
    coarse = l86_content_certificate()
    if not coarse.verified or coarse.minimum_weight < 7 or not l86_binary_content_proof().verified:
        return None
    sigma = 43-target.a_even_ones-target.a_odd_ones
    lower = max(2, (coarse.minimum_weight-sigma+1)//2)
    upper = min(14 if target.k % 2 else 15, (43-coarse.minimum_weight-sigma)//2)
    return ((lower, upper), (17-upper, 17-lower))
