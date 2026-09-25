"""Bounded free-coordinate navigation AFTER an exact odd-half compressed match.

Each free assignment decodes through b=b0 XOR N*z, so ALL proposals satisfy
the necessary GF(2) equations. Proposed full words are then checked for exact
parity content BEFORE entering the local search. Rejected content assignments
are not hard pruning of their neighbours. A miss never proves infeasibility,
never restricts the subsequent Z3 model and never blocks an unexplored branch.

The neighbourhood flips one/two FREE coordinates, not necessarily one/two
physical bits. Periodic independent assignments permit different components
of the content-feasible subset. No saved solutions, training data or GPU needed.
"""

from dataclasses import dataclass
import random
from time import perf_counter
from typing import Optional, Sequence, Tuple

from .compressed_fkm import lift_factor2_compressed
from .correlation import full_correlation_profile
from .orientation_gf2 import odd_orientation_system
from .structured_energy import fold_profile, integer_periodic_autocorrelation_profile, target_profile
from .target_profiles import TargetContentProfile, pair_content
from .verifier import verify_pqcp


@dataclass(frozen=True)
class OrientationSearchResult:
    status: str  # SAT or MISS; never UNSAT from bounded search
    pair: Optional[Tuple[Tuple[int, ...], Tuple[int, ...]]]
    residual_energy: Optional[int]
    evaluations: int
    feasible: int
    elapsed: float


def search_orientations(ca: Sequence[int], cb: Sequence[int], target: TargetContentProfile,
                        *, steps: int = 128, seed: int = 123,
                        preferred_pair=None, trace: Optional[list] = None) -> OrientationSearchResult:
    """Return a verified solution or a content-feasible Z3 preference (or None).

    Proposal budget includes content-rejected proposals. Every 16th proposal
    restarts from an independent free assignment; all other proposals flip
    one/two free coordinates, accepting non-increasing exact residual energy.
    The schedule is an untuned bounded heuristic. With no feasible current
    point, independent assignments continue. No claim of connectedness.
    """
    started = perf_counter()
    if type(steps) is not int or steps <= 0:
        raise ValueError("steps must be a positive integer")
    ca, cb = tuple(ca), tuple(cb)
    d = len(ca)
    if target.L != 2*d or len(cb) != d or d % 2 != 1:
        raise ValueError("odd-half target and compression lengths must agree")
    desired = target_profile(target.L, target.k, target.eta)
    compressed = tuple(x+y for x, y in zip(integer_periodic_autocorrelation_profile(ca),
                                          integer_periodic_autocorrelation_profile(cb)))
    if compressed != fold_profile(desired, 2):
        raise ValueError("orientation search requires an exact compressed match")
    residual = tuple((desired[u]-desired[u+d])//4 for u in range(d))
    system = odd_orientation_system(ca, cb, residual)
    if not system.consistent:
        # Exact solver already has the safe inconsistency proof. This helper
        # only reports that it found no candidate, never fabricates a SAT hint.
        return OrientationSearchResult("MISS", None, None, 0, 0, perf_counter()-started)
    expressions = system.expressions()
    free_count = len(system.free_columns)
    ids = {position: j for j, position in enumerate(system.variables)}
    # A product's sign is an affine parity in free coordinates, including the
    # negacyclic wrap sign. Its +/-1 sum reconstructs N(u) exactly.
    rows = []
    for u in range(1, (d+1)//2):
        terms = []
        for side, word in enumerate((ca, cb)):
            for i in range(d):
                j = (i+u) % d
                if word[i] == word[j] == 0:
                    mi, ci = expressions[ids[side, i]]
                    mj, cj = expressions[ids[side, j]]
                    terms.append((mi ^ mj, ci ^ cj ^ int(i+u >= d)))
        rows.append((residual[u], terms))
    expected_content = (target.a_even_ones, target.a_odd_ones,
                        target.b_even_ones, target.b_odd_ones)
    rng = random.Random(seed)
    current = current_energy = best_energy = best = None
    feasible = evaluations = 0
    for step in range(steps):
        restart = current is None or step % 16 == 0
        if step == 0 and preferred_pair is not None:
            assignment = sum(int(preferred_pair[side][i]) << j
                             for j, (side, i) in enumerate(system.variables[c] for c in system.free_columns))
        elif restart:
            assignment = rng.getrandbits(free_count)
        else:
            assignment = current
            if free_count:
                for column in rng.sample(range(free_count), min(free_count, 1 + rng.randrange(2))):
                    assignment ^= 1 << column
        bits = tuple((bin(mask & assignment).count("1") & 1) ^ constant
                     for mask, constant in expressions)
        pair = tuple(lift_factor2_compressed(word, [bits[j] for j, (s, _) in enumerate(system.variables) if s == side])
                     for side, word in enumerate((ca, cb)))
        evaluations += 1
        if pair_content(*pair) != expected_content:
            if trace is not None:
                trace.append((assignment, pair, False, None))
            continue
        feasible += 1
        energy = sum((sum(1 - 2*((bin(mask & assignment).count("1") & 1) ^ constant)
                          for mask, constant in terms) - wanted)**2 for wanted, terms in rows)
        if trace is not None:
            trace.append((assignment, pair, True, energy))
        if best_energy is None or energy < best_energy:
            best, best_energy = pair, energy
        if energy == 0:
            check = verify_pqcp(*pair)
            if not check.is_valid or tuple(full_correlation_profile(*pair)) != desired:
                raise RuntimeError("orientation zero energy failed independent exact-target verification")
            return OrientationSearchResult("SAT", pair, 0, evaluations, feasible, perf_counter()-started)
        if restart or energy <= current_energy:
            current, current_energy = assignment, energy
    return OrientationSearchResult("MISS", best, best_energy, evaluations, feasible, perf_counter()-started)
