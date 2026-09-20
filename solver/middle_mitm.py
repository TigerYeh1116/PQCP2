"""Exact fixed-coarse middle MITM with bounded memory and replayable order.

This is NOT full coarse enumeration. For L=4n, v=c2/2 and
v[i]+v[i+n]=c4[i]/2. Content/norm suffix DP and complete integer PACF
signatures retain every match. Resource limits mean DEFERRED, never UNSAT.
The caller owns child proofs, persistence and the Z3 fallback.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Dict, Iterator, Optional, Sequence, Tuple, Iterable

from .compressed_equivalence import compressed_job_equivalence_key
from .compressed_lift import _validate_content
from .structured_energy import (
    fold_profile, integer_periodic_autocorrelation_profile as pacf, target_profile,
)
from .target_profiles import TargetContentProfile

ORDER_VERSION = 1

def popcount(value: int) -> int:
    """Python 3.9 compatible exact population count."""
    return bin(value).count("1")


if hasattr(int, "bit_count"):
    popcount = int.bit_count


class BudgetExpired(Exception):
    """Cooperative wall-clock stop, not infeasibility."""


@dataclass
class MiddlePlan:
    """Finite choices and exact suffix counts for one fixed coarse word."""

    n: int
    even_sum: int
    options: tuple
    suffix: list
    masks: list

    def norm_counts(self) -> Dict[int, int]:
        return {norm: count for (even, norm), count in self.suffix[0].items()
                if even == self.even_sum}


def make_plan(coarse: Sequence[int], even_ones: int, odd_ones: int) -> MiddlePlan:
    """Compute exact completion counts; no approximate bound or RNG."""
    n = len(coarse)
    if not n or any(type(v) is not int or v not in (-4,-2,0,2,4) for v in coarse):
        raise ValueError("invalid factor-4 word")
    if any(type(w) is not int or not 0 <= w <= 2*n for w in (even_ones,odd_ones)):
        raise ValueError("invalid content")
    options = []
    for i, value in enumerate(coarse):
        q = value//2
        choices = []
        for x in (-1,0,1):
            y = q-x
            if y in (-1,0,1):
                even = (x if i%2 == 0 else 0) + (y if (i+n)%2 == 0 else 0)
                choices.append((x,y,even,x*x+y*y))
        options.append(tuple(choices))
    suffix = [Counter() for _ in range(n+1)]
    if sum(coarse)//2 == 2*n-even_ones-odd_ones:
        suffix[n][(0,0)] = 1
    for i in range(n-1,-1,-1):
        for _, _, even, norm in options[i]:
            for (e,q), count in suffix[i+1].items():
                suffix[i][(e+even,q+norm)] += count
    masks = []
    for counts in suffix:
        by_even = defaultdict(int)
        for e,q in counts:
            by_even[e] |= 1 << q
        masks.append(dict(by_even))
    return MiddlePlan(n,n-even_ones,tuple(options),suffix,masks)


def generate_middle(plan: MiddlePlan, norms: Sequence[int],
                    tick: Optional[Callable[[], None]] = None) -> Iterator[Tuple[int,int]]:
    """Yield (nonzero mask, negative mask), without duplicating any word."""
    wanted = sum(1 << q for q in set(norms))
    visits = 0

    def visit(i, even, norm, occupied, negative):
        nonlocal visits
        visits += 1
        if tick is not None and visits % 1024 == 0:
            tick()
        attainable = plan.masks[i].get(plan.even_sum-even,0)
        if not ((attainable << norm) & wanted):
            return
        if i == plan.n:
            yield occupied, negative
            return
        for x,y,e,q in plan.options[i]:
            occ = occupied | ((1 << i) if x else 0) | ((1 << (i+plan.n)) if y else 0)
            neg = negative | ((1 << i) if x < 0 else 0) | ((1 << (i+plan.n)) if y < 0 else 0)
            yield from visit(i+1,even+e,norm+q,occ,neg)

    yield from visit(0,0,0,0,0)


def decode_middle(occupied: int, negative: int, length: int) -> tuple:
    """Return actual c2 entries {-2,0,2}, not scaled ternary values."""
    return tuple(0 if not occupied >> i & 1 else -2 if negative >> i & 1 else 2
                 for i in range(length))


def bit_signature(occupied: int, negative: int, length: int) -> tuple:
    """Exact independent PACF of v=c2/2 using two popcounts per shift.

    The product is nonzero exactly on the overlap of occupied masks. It is
    -1 iff the signs differ. Hence P(v;u)=overlap_count-2*mismatch_count.
    This includes shifts 0 and length/2 and is audited against scalar PACF.
    """
    mask = (1 << length)-1
    values = [popcount(occupied)]
    for u in range(1,length//2+1):
        ro = ((occupied >> u) | (occupied << (length-u))) & mask
        rn = ((negative >> u) | (negative << (length-u))) & mask
        overlap = occupied & ro
        values.append(popcount(overlap)-2*popcount(overlap & (negative ^ rn)))
    return tuple(values)


def middle_matches(
    ca: Sequence[int], cb: Sequence[int], profile: TargetContentProfile,
    max_index_entries: int, tick: Callable[[], None],
) -> Iterator[Tuple[tuple, tuple]]:
    """Yield all exact matches deterministically, including equivalent ones.

    All signature collisions retain all words. The index cap is checked
    BEFORE materialization. No RNG, solution archive or floating point PACF.
    """
    ca, cb = tuple(ca), tuple(cb)
    n, L = len(ca), profile.L
    if not n or len(cb) != n or L != 4*n:
        raise ValueError("factor-4 length mismatch")
    _validate_content(profile, L, profile.k, profile.eta)
    if type(max_index_entries) is not int or max_index_entries < 1:
        raise ValueError("positive integer index cap required")
    plans = (make_plan(ca, profile.a_even_ones, profile.a_odd_ones),
             make_plan(cb, profile.b_even_ones, profile.b_odd_ones))
    desired = target_profile(L, profile.k, profile.eta)
    if tuple(x+y for x,y in zip(pacf(ca),pacf(cb))) != fold_profile(desired,4):
        return
    folded = fold_profile(desired,2)
    if any(v%4 for v in folded):
        return  # every middle product is a multiple of four
    target = tuple(v//4 for v in folded[:n+1])
    counts = [plan.norm_counts() for plan in plans]
    norms = [tuple(q for q in counts[s] if target[0]-q in counts[1-s])
             for s in (0,1)]
    totals = [sum(counts[s][q] for q in norms[s]) for s in (0,1)]
    tick()
    if not min(totals):
        return
    build = 0 if totals[0] < totals[1] else 1
    if totals[build] > max_index_entries:
        raise BudgetExpired("middle index entry cap")
    index = defaultdict(list)
    for number, (occupied, negative) in enumerate(
            generate_middle(plans[build],norms[build],tick)):
        if number%128 == 0:
            tick()
        sig = bit_signature(occupied,negative,2*n)
        index[sig].append((occupied,negative))
    probe = 1-build
    for number, (occupied, negative) in enumerate(
            generate_middle(plans[probe],norms[probe],tick)):
        if number%128 == 0:
            tick()
        needed = tuple(t-x for t,x in zip(target,bit_signature(occupied,negative,2*n)))
        for bo,bn in index.get(needed,()):
            tick()
            middle = [None,None]
            middle[probe] = decode_middle(occupied,negative,2*n)
            middle[build] = decode_middle(bo,bn,2*n)
            a,b = middle
            if tuple(x+y for x,y in zip(pacf(a),pacf(b))) != folded:
                raise RuntimeError("MITM match failed independent scalar PACF audit")
            if any(w[i]+w[i+n] != c[i]
                   for w,c in zip(middle,(ca,cb)) for i in range(n)):
                raise RuntimeError("MITM match changed fixed compression")
            yield a,b


def next_middle(
    ca: Sequence[int], cb: Sequence[int], profile: TargetContentProfile,
    timeout_ms: int, max_index_entries: int = 250000, cursor: int = 0,
    excluded: Iterable[tuple] = (),
    batch_size: int = 1,
) -> dict:
    """Replay to a durable raw-match cursor and return a bounded unseen batch.

    The cursor counts ALL raw matches, including duplicates. Excluded keys
    must refer to durably recorded equivalent factor-2 children with the
    full target/content transformed by the existing canonical key. Pending
    children are not discarded. EXHAUSTED proves only middle enumeration
    exhausted; it says nothing about the status of those recorded children.
    Rebuild/replay exceeding its slice returns DEFERRED so the caller may
    switch to the old Z3 enumeration with recorded patterns blocked.
    """
    if type(cursor) is not int or cursor < 0 or type(timeout_ms) is not int or timeout_ms <= 0:
        raise ValueError("nonnegative cursor and positive integer time budget required")
    if type(batch_size) is not int or not 1<=batch_size<=16:
        raise ValueError("middle batch size must be in 1..16")
    started = perf_counter()
    deadline = started + timeout_ms/1000
    seen = set(excluded)
    consumed, duplicates, traversed = cursor, 0, 0
    matches = []

    def tick() -> None:
        if perf_counter() >= deadline:
            raise BudgetExpired("middle MITM time slice")

    def result(status: str, **extra) -> dict:
        if matches:
            extra.update(a=matches[0][0],b=matches[0][1])  # single-result compatibility
        return dict(status=status, cursor=consumed, duplicates=duplicates,
                    elapsed=perf_counter()-started, order_version=ORDER_VERSION,
                    matches=tuple(matches),**extra)

    try:
        for position,(a,b) in enumerate(middle_matches(ca,cb,profile,max_index_entries,tick)):
            traversed = position+1
            if position < cursor:
                continue
            key = compressed_job_equivalence_key(a,b,profile,2)
            consumed = position+1
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            matches.append((a,b))
            if len(matches)>=batch_size:
                return result("FOUND",exhausted=False)
        if traversed < cursor:
            raise ValueError("saved MITM cursor beyond complete raw match stream")
        return result("FOUND" if matches else "EXHAUSTED",exhausted=True)
    except BudgetExpired as error:
        # Partial batches are durable progress, not discarded on a time cap.
        return result("FOUND" if matches else "DEFERRED",reason=str(error),exhausted=False)
