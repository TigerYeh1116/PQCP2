"""Experimental exact middle-layer MITM. Not connected to production search.

For L=4n use v=c2/2 in {-1,0,1}, with v[i]+v[i+n]=c4[i]/2.
Suffix DP enforces the exact even/odd sums and partner-compatible squared
norms. All middle PACF signatures are integer, including the half shift.
Limits return UNKNOWN, never a mathematical pruning verdict.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Dict, Iterator, Optional, Sequence, Tuple

from solver.compressed_lift import solve_compressed_lift
from solver.cross_compression import compatible_cross_compressions, middle_matches_cross
from solver.structured_energy import (
    compress_binary_sequence, fold_profile, integer_periodic_autocorrelation_profile,
    target_profile,
)
from solver.target_profiles import TargetContentProfile, pair_content
from solver.verifier import verify_pqcp


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


def solve_middle_mitm(ca, cb, profile: TargetContentProfile, seconds: float,
                      max_index_entries: int = 250000, max_middle_models: int = 16,
                      progress: Optional[Callable[[dict],None]] = None,
                      binary_slice_ms: int = 0) -> dict:
    """Exact fixed-subproblem search with bounded memory and existing Z3 lift.

    Build the smaller complete side, then stream the other. All equal keys
    retain all words. No rotation/decimation of the fixed input is performed.
    A final UNSAT is allowed only after complete joining and all children
    UNSAT. In particular UNKNOWN children remain unresolved.
    The benchmark also enforces an external deadline covering Z3 construction.
    """
    if seconds <= 0 or max_index_entries < 1 or max_middle_models < 1 or binary_slice_ms < 0:
        raise ValueError("positive budgets required")
    started = perf_counter()
    deadline = started + seconds
    stats = dict(status="UNKNOWN", stage="prepare", middle_models=0, middle_checks=0,
                 binary_checks=0, binary_started=0, unknown_children=0,
                 binary_unsat=0, generated=[0,0], indexed=0, verified=False,
                 construction_seconds=0.0, solve_seconds=0.0,binary_history=[])
    last_emit = started-1

    def tick(force=False):
        nonlocal last_emit
        now = perf_counter()
        stats["elapsed"] = now-started
        if progress and (force or now-last_emit >= .2):
            progress(dict(stats))
            last_emit = now
        if now >= deadline:
            raise BudgetExpired()

    def finish(status, reason):
        stats.update(status=status,reason=reason,elapsed=perf_counter()-started)
        if progress:
            progress(dict(stats))
        return stats

    try:
        n,L = len(ca),profile.L
        if len(cb) != n or L != 4*n:
            raise ValueError("length mismatch")
        desired = target_profile(L,profile.k,profile.eta)
        if tuple(x+y for x,y in zip(integer_periodic_autocorrelation_profile(ca),
                                   integer_periodic_autocorrelation_profile(cb))) != fold_profile(desired,4):
            return finish("UNSAT","coarse mismatch")
        cross = (compatible_cross_compressions(ca,cb,profile.k,profile.eta,profile)
                 if n%2 else None)
        if cross == ():
            return finish("UNSAT","no compatible cross patterns")
        plans = (make_plan(ca,profile.a_even_ones,profile.a_odd_ones),
                 make_plan(cb,profile.b_even_ones,profile.b_odd_ones))
        target = tuple(t//4 for t in fold_profile(desired,2)[:n+1])
        assert all(t%4 == 0 for t in fold_profile(desired,2))
        counts = [p.norm_counts() for p in plans]
        norms = [tuple(q for q in counts[s] if target[0]-q in counts[1-s]) for s in (0,1)]
        totals = [sum(counts[s][q] for q in norms[s]) for s in (0,1)]
        stats["dp_candidate_counts"] = totals
        tick(True)
        if min(totals) == 0:
            return finish("UNSAT","exact content/norm DP exhausted")
        build = 0 if totals[0] < totals[1] else 1
        if totals[build] > max_index_entries:
            return finish("UNKNOWN","index budget exceeded; not a pruning verdict")
        cache = ({},{})
        even_mask = sum(1 << i for i in range(0,2*n,2))
        all_mask = (1 << (2*n))-1
        parity_masks = (even_mask,all_mask ^ even_mask)

        def cross_mask(side,occ,neg):
            if cross is None:
                return 1
            # Fixed negative entries put a one in BOTH antipodal mod-4 rows.
            # Zeros allocate a one to one of them; feasibility depends only
            # on negative and zero counts per parity, not their order.
            key = tuple((popcount(neg&m),popcount((all_mask ^ occ)&m)) for m in parity_masks)
            if key not in cache[side]:
                word = decode_middle(occ,neg,2*n)
                cache[side][key] = sum(1 << i for i,pair in enumerate(cross)
                                      if middle_matches_cross(word,pair[side]))
            return cache[side][key]

        index = defaultdict(list)
        stats["stage"] = "index"
        for occ,neg in generate_middle(plans[build],norms[build],tick):
            stats["generated"][build] += 1
            cm = cross_mask(build,occ,neg)
            if cm:
                sig = bit_signature(occ,neg,2*n)
                index[bytes(x+2*n for x in sig)].append((occ,neg,cm))
                stats["indexed"] += 1
            if stats["generated"][build] % 256 == 0:
                tick()
        stats.update(index_seconds=perf_counter()-started, buckets=len(index),stage="probe")
        tick(True)
        probe = 1-build
        for occ,neg in generate_middle(plans[probe],norms[probe],tick):
            stats["generated"][probe] += 1
            pm = cross_mask(probe,occ,neg)
            if not pm:
                continue
            sig = bit_signature(occ,neg,2*n)
            needed = tuple(t-x for t,x in zip(target,sig))
            if min(needed) < -2*n or max(needed) > 2*n:
                continue
            for bo,bn,bm in index.get(bytes(x+2*n for x in needed),()):
                mask = pm & bm
                if not mask:
                    continue
                middle = [None,None]
                middle[probe] = decode_middle(occ,neg,2*n)
                middle[build] = decode_middle(bo,bn,2*n)
                actual = tuple(x+y for x,y in zip(
                    integer_periodic_autocorrelation_profile(middle[0]),
                    integer_periodic_autocorrelation_profile(middle[1])))
                if actual != fold_profile(desired,2) or any(
                        w[i]+w[i+n] != c[i] for w,c in zip(middle,(ca,cb)) for i in range(n)):
                    raise RuntimeError("MITM middle failed scalar audit")
                stats["middle_models"] += 1
                stats.setdefault("first_middle_seconds",perf_counter()-started)
                eligible = None if cross is None else tuple(p for i,p in enumerate(cross) if mask >> i & 1)
                tick(True)
                stats.update(stage="binary",binary_started=stats["binary_started"]+1)
                timeout = max(1,int((deadline-perf_counter())*1000))
                if binary_slice_ms:
                    timeout = min(timeout,binary_slice_ms)
                attempt = dict(middle_a=middle[0],middle_b=middle[1],status="IN_FLIGHT",
                               started_seconds=perf_counter()-started,check_timeout_ms=timeout)
                stats["binary_history"].append(attempt)
                tick(True)
                child = solve_compressed_lift(*middle,profile.k,profile.eta,
                    timeout,profile,cross_compressions=eligible)
                attempt.update(status=child.status,construction_seconds=child.construction_time,
                               solve_seconds=child.solve_time)
                stats["construction_seconds"] += child.construction_time
                stats["solve_seconds"] += child.solve_time
                stats["binary_checks"] += int(child.z3_executed)
                stats["unknown_children"] += int(child.status == "UNKNOWN")
                stats["binary_unsat"] += int(child.status == "UNSAT")
                if child.status == "SAT":
                    check = verify_pqcp(child.a,child.b)
                    if (not check.is_valid or check.profile != desired
                            or pair_content(child.a,child.b) != (profile.a_even_ones,
                            profile.a_odd_ones,profile.b_even_ones,profile.b_odd_ones)
                            or compress_binary_sequence(child.a,4) != tuple(ca)
                            or compress_binary_sequence(child.b,4) != tuple(cb)):
                        raise RuntimeError("SAT failed independent full audit")
                    stats.update(verified=True,A=child.a,B=child.b,profile=check.profile)
                    return finish("SAT","independently verified")
                if stats["middle_models"] >= max_middle_models:
                    return finish("UNKNOWN","middle model budget reached")
                stats["stage"] = "probe"
                tick(True)
            if stats["generated"][probe] % 256 == 0:
                tick()
        return finish("UNKNOWN" if stats["unknown_children"] else "UNSAT",
                      "unresolved binary children" if stats["unknown_children"] else "complete exact join and all children UNSAT")
    except BudgetExpired:
        return finish("UNKNOWN","wall-clock budget exhausted")
