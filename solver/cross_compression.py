"""Exact compatibility of factor-4 and factor-n compression for L=4n, n odd.

CRT maps index i bijectively to (i mod 4, i mod n). A binary word is thus a
4-by-n binary matrix. Factor-4 sign sums give column degrees (4-c[j])/2;
factor-n sign sums give row degrees (n-r[j])/2. Gale--Ryser characterizes
existence of a binary matrix with these margins: equal totals and, for
sorted row degrees d, sum(d[:t]) <= sum(min(t,c) for c in columns), t=1..4.

Enumerate ALL four-entry patterns with the required total/parity content,
then hash-match their entire integer PACFs to fold_n(T). Compatibility is
necessary for a PQCP, not sufficient. This test is intended for independently
matched compressed patterns, and never uses known PQCPs or floating point.
For even n CRT is not bijective; callers must retain the previous path.
"""

from functools import lru_cache
from typing import Optional, Sequence, Tuple

from .structured_energy import fold_profile, integer_periodic_autocorrelation_profile, target_profile
from .target_profiles import TargetContentProfile

Word = Tuple[int, ...]
CrossPair = Tuple[Word, Word]


def binary_margins_feasible(rows: Sequence[int], columns: Sequence[int]) -> bool:
    """Exact Gale--Ryser test for unrestricted 0/1 matrices (no fixed cells).

    Labels of rows/columns are retained by the caller. Sorting degrees for
    the inequalities is not a claim that arbitrary position permutations
    preserve PACF. Fixed cells require a different feasibility test.
    """
    rows, columns = tuple(rows), tuple(columns)
    if any(not isinstance(x, int) or isinstance(x, bool) for x in rows + columns):
        raise ValueError("matrix margins must be integers")
    if any(x < 0 or x > len(columns) for x in rows) or any(
            x < 0 or x > len(rows) for x in columns):
        return False
    if sum(rows) != sum(columns):
        return False
    total = 0
    for t, value in enumerate(sorted(rows, reverse=True), 1):
        total += value
        if total > sum(min(t, c) for c in columns):
            return False
    return True


def _factor4_word(word: Sequence[int]) -> Word:
    word = tuple(word)
    if not word or len(word) % 2 == 0 or any(
            not isinstance(x, int) or isinstance(x, bool) or x not in (-4,-2,0,2,4)
            for x in word):
        raise ValueError("cross compression requires odd-length factor-4 words")
    return word


@lru_cache(maxsize=4096)
def _compatible_rows(columns: Word, parity: Optional[Tuple[int, int]],
                     energy_bound: int) -> Tuple[Tuple[Word, Word], ...]:
    """Enumerate row degrees exhaustively; return sign words and exact PACFs.

    Column order does not affect margin feasibility, so callers cache the
    sorted degrees. Row order DOES affect PACF and is never canonicalized.
    Squared norm <= fold_n(T)[0] is safe since the other norm is nonnegative.
    """
    n, weight = len(columns), sum(columns)
    candidates = []
    for d0 in range(n + 1):
        for d1 in range(n + 1):
            d2s = (parity[0] - d0,) if parity is not None else range(n + 1)
            for d2 in d2s:
                d3 = weight - d0 - d1 - d2
                if not (0 <= d2 <= n and 0 <= d3 <= n):
                    continue
                if parity is not None and d1 + d3 != parity[1]:
                    continue
                degrees = (d0, d1, d2, d3)
                word = tuple(n - 2 * d for d in degrees)
                if sum(x*x for x in word) > energy_bound:
                    continue
                if binary_margins_feasible(degrees, columns):
                    candidates.append((word, integer_periodic_autocorrelation_profile(word)))
    return tuple(candidates)


def compatible_cross_compressions(
    ca: Sequence[int], cb: Sequence[int], k: int, eta: int,
    content_profile: Optional[TargetContentProfile] = None,
) -> Tuple[CrossPair, ...]:
    """All compatible factor-n pattern pairs for fixed factor-4 patterns.

    Empty output proves impossibility only for these patterns/target/content.
    Enumeration is complete, with no sampling, truncation or heuristic cap.
    Total row sums equal the given column sums. Optional even/odd content is
    imposed as d0+d2=w_even, d1+d3=w_odd for each word.
    """
    ca, cb = _factor4_word(ca), _factor4_word(cb)
    if len(ca) != len(cb):
        raise ValueError("compressed words must have equal lengths")
    L, n = 4 * len(ca), len(ca)
    desired = target_profile(L, k, eta)
    parities = (None, None)
    if content_profile is not None:
        if not isinstance(content_profile, TargetContentProfile) or (
                content_profile.L, content_profile.k, content_profile.eta) != (L,k,eta):
            raise ValueError("content must match L, k and eta")
        parities = ((content_profile.a_even_ones, content_profile.a_odd_ones),
                    (content_profile.b_even_ones, content_profile.b_odd_ones))
        if any(not isinstance(x,int) or isinstance(x,bool) or not 0 <= x <= L//2
               for p in parities for x in p):
            raise ValueError("invalid parity content")
    target = desired if n == 1 else fold_profile(desired, n)
    a_rows, b_rows = (_compatible_rows(tuple(sorted((4-x)//2 for x in word)), parity, target[0])
                      for word, parity in zip((ca,cb), parities))
    index = {}
    for word, profile in b_rows:
        index.setdefault(profile, []).append(word)
    return tuple((a,b) for a, profile in a_rows
                 for b in index.get(tuple(t-x for t,x in zip(target,profile)), ()))


def middle_matches_cross(word: Sequence[int], cross: Sequence[int]) -> bool:
    """Exact feasibility of factor-2 orientations with given mod-4 row sums.

    For L=4n, n odd, antipodal indices i,i+2n are in residues r,r+2 mod 4.
    Nonzero middle entries fix both signs; every zero assigns exactly one 1
    to either row in that parity pair. The two parity groups are independent,
    so totals plus interval bounds are necessary AND sufficient here.
    This does not assert feasibility of the full correlation constraints.
    """
    word, cross = tuple(word), tuple(cross)
    if not word or len(word) % 4 != 2 or any(
            not isinstance(x,int) or isinstance(x,bool) or x not in (-2,0,2) for x in word):
        raise ValueError("expected factor-2 pattern for L=4n with n odd")
    n = len(word)//2
    if len(cross) != 4 or any(not isinstance(x,int) or isinstance(x,bool)
            or abs(x) > n or (n-x)%2 for x in cross):
        raise ValueError("invalid cross-compression alphabet")
    fixed, zeros = [0]*4, [0]*2
    for i, value in enumerate(word):
        if value == -2:
            fixed[i%4] += 1
            fixed[(i+len(word))%4] += 1
        elif value == 0:
            zeros[i%2] += 1
    degrees = tuple((n-x)//2 for x in cross)
    for p in (0,1):
        a, b = degrees[p]-fixed[p], degrees[p+2]-fixed[p+2]
        if min(a,b) < 0 or a+b != zeros[p]:
            return False
    return True


def normalize_cross_pairs(pairs: Sequence[CrossPair], L: int, k: int, eta: int) -> Tuple[CrossPair, ...]:
    """Validate exact factor-(L/4) pair constraints supplied to binary Z3 lift."""
    if L % 4 or (L//4)%2 == 0:
        raise ValueError("cross constraints require L=4n, n odd")
    n = L//4
    desired = target_profile(L,k,eta)
    if n != 1:
        desired = fold_profile(desired, n)
    normalized = []
    for pair in pairs:
        if len(pair) != 2:
            raise ValueError("expected two cross patterns")
        a,b = (tuple(w) for w in pair)
        if any(len(w) != 4 or any(not isinstance(x,int) or isinstance(x,bool)
               or abs(x)>n or (n-x)%2 for x in w) for w in (a,b)):
            raise ValueError("invalid cross-compression pattern")
        if tuple(x+y for x,y in zip(integer_periodic_autocorrelation_profile(a),
                                    integer_periodic_autocorrelation_profile(b))) != desired:
            raise ValueError("cross patterns do not match folded target")
        normalized.append((a,b))
    return tuple(dict.fromkeys(normalized))
