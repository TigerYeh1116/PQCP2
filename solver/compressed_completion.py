"""Bounded compression matchmaking and persistence for background Z3 lifts.

The pool is a search heuristic, not an exhaustive compressed enumeration.
Only exact factor-2 or factor-4 PACF matches become jobs. Dropping an old pool entry or
not finding a match is never an UNSAT claim. The original radius completion
is independent and remains available as a fallback.
"""

from collections import OrderedDict
from dataclasses import asdict, dataclass
from functools import cached_property
import hashlib
import json
from pathlib import Path
from typing import List, Sequence, Tuple

from .checkpoint import atomic_write_json
from .compressed_pairing import compressed_pacf_signature, required_partner_signature
from .compressed_equivalence import CompressedJobKey, compressed_job_equivalence_key
from .correlation import normalize_binary_sequence
from .objective import pqcp_objective, pqcp_objective_breakdown
from .search_runner import append_verified_solution_if_new
from .structured_energy import compress_binary_sequence, target_profile
from .target_profiles import TargetContentProfile, pair_content, target_content_profiles
from .verifier import verify_pqcp


@dataclass(frozen=True)
class CompressedLiftJob:
    """Fixed compressed patterns, exact content and traceable source words."""

    compressed_a: Tuple[int, ...]
    compressed_b: Tuple[int, ...]
    target: TargetContentProfile
    source_a: Tuple[int, ...]
    source_b: Tuple[int, ...]
    factor: int = 2

    @property
    def key(self):
        return self.compressed_a, self.compressed_b, self.target, self.factor

    @cached_property
    def canonical_key(self) -> CompressedJobKey:
        """Equivalent fixed lift problems share this key, not source words."""
        return compressed_job_equivalence_key(
            self.compressed_a, self.compressed_b, self.target, self.factor)

    @property
    def identifier(self) -> str:
        encoded = json.dumps((self.factor, self.compressed_a, self.compressed_b, asdict(self.target)),
                             sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:20]


class CompressedLiftPool:
    """Remember bounded sides and cross-match each newly offered candidate.

    A and B sides share a pool indexed by exact parity content, since their
    role does not affect their individual PACF. At most capacity compressed
    patterns are retained per content. A job's key includes its content;
    when L/2 is odd the same compressed pattern can have different valid
    alternating-sum constraints. With canonical admission, content and the
    full target transform along with both compressed words. Only proven
    subproblem symmetries are used; a representative stays in its original
    coordinates. submit() deduplicates only scheduled jobs,
    so an overfull background queue does not mark unattempted jobs UNSAT.
    """

    def __init__(self, L: int, capacity: int = 64, history_capacity: int = 512, factor: int = 2,
                 deduplicate_equivalents: bool = True):
        if any(not isinstance(x, int) or isinstance(x, bool) or x <= 0
               for x in (capacity, history_capacity)):
            raise ValueError("pool capacities must be positive integers")
        self.L = L
        if factor not in (2, 4) or L % factor:
            raise ValueError("pool factor must be 2 or 4 and divide L")
        self.factor = factor
        if not isinstance(deduplicate_equivalents, bool):
            raise ValueError("deduplicate_equivalents must be boolean")
        self.deduplicate_equivalents = deduplicate_equivalents
        self.duplicates_skipped = 0
        self.targets = frozenset(target_content_profiles(L))
        self.capacity, self.history_capacity = capacity, history_capacity
        self.pools = {}
        self.signature_index = {}
        self.submitted = OrderedDict()

    def offer(self, a: Sequence[int], b: Sequence[int], k: int, eta: int) -> List[CompressedLiftJob]:
        """Return new exact matches for this actual target; no score guard."""
        a, b = normalize_binary_sequence(a), normalize_binary_sequence(b)
        if len(a) != self.L or len(b) != self.L:
            raise ValueError("offered candidate length does not match pool")
        if (not isinstance(k, int) or isinstance(k, bool) or not 0 < k < self.L // 2
                or not isinstance(eta, int) or isinstance(eta, bool) or eta not in (-1, 1)):
            raise ValueError("invalid target shift/sign")
        target = TargetContentProfile(self.L, k, eta, *pair_content(a, b))
        if target not in self.targets:
            return []  # this content cannot realize this specific target
        ac, bc = pair_content(a, b)[:2], pair_content(a, b)[2:]
        for content, word in ((ac, a), (bc, b)):
            pool = self.pools.setdefault(content, OrderedDict())
            compressed = compress_binary_sequence(word, self.factor)
            pool[compressed] = word
            pool.move_to_end(compressed)
            signature = compressed_pacf_signature(word, (self.factor,))
            self.signature_index.setdefault((content, signature), OrderedDict())[compressed] = word
            if len(pool) > self.capacity:
                old_compressed, old_word = pool.popitem(last=False)
                index_key = (content, compressed_pacf_signature(old_word, (self.factor,)))
                bucket = self.signature_index[index_key]
                del bucket[old_compressed]
                if not bucket:
                    del self.signature_index[index_key]
        # At least one side must be newly offered: no quadratic rescan of
        # all archive pairs is necessary on every observation.
        b_bucket = self.signature_index.get((bc, required_partner_signature(a, target, (self.factor,))), {})
        a_bucket = self.signature_index.get((ac, required_partner_signature(b, target, (self.factor,))), {})
        matches = [(a, old_b) for old_b in b_bucket.values()]
        matches.extend((old_a, b) for old_a in a_bucket.values())
        jobs = {}
        for match_a, match_b in matches:
            job = CompressedLiftJob(
                compress_binary_sequence(match_a, self.factor), compress_binary_sequence(match_b, self.factor),
                target, match_a, match_b, self.factor,
            )
            key = job.canonical_key if self.deduplicate_equivalents else job.key
            if key in self.submitted or key in jobs:
                self.duplicates_skipped += 1
            else:
                jobs[key] = job
        return list(jobs.values())

    def submit(self, job: CompressedLiftJob) -> None:
        """Remember an actually queued job in a bounded LRU history."""
        key = job.canonical_key if self.deduplicate_equivalents else job.key
        self.submitted[key] = None
        self.submitted.move_to_end(key)
        if len(self.submitted) > self.history_capacity:
            self.submitted.popitem(last=False)


def run_compressed_lift_job(
    job: CompressedLiftJob, timeout_ms: int, root: Path, seed: int,
    max_middle_models: int = 16,
    cross_compression_pruning: bool = True,
    orientation_preprocessing: bool = True,
    orientation_search_steps: int = 0,
) -> dict:
    """Solve, audit and save; official results use the shared symmetry-aware writer."""
    from .compressed_lift import solve_compressed_lift

    if job.factor == 4:
        from .multilevel_lift import solve_multilevel_lift
        result = solve_multilevel_lift(
            job.compressed_a, job.compressed_b, job.target.k, job.target.eta,
            timeout_ms=timeout_ms, content_profile=job.target, max_middle_models=max_middle_models,
            cross_compression_pruning=cross_compression_pruning)
    elif job.factor == 2:
        result = solve_compressed_lift(
            job.compressed_a, job.compressed_b, job.target.k, job.target.eta,
            timeout_ms=timeout_ms, content_profile=job.target,
            orientation_preprocessing=orientation_preprocessing,
            **({"orientation_search_steps": orientation_search_steps,
                "orientation_search_seed": seed ^ int(job.identifier[:8], 16),
                "preferred_pair": (job.source_a, job.source_b)} if orientation_search_steps else {}))
    else:
        raise ValueError("unsupported lift factor")
    return persist_compressed_lift_result(job,result,root,seed,{
        "timeout_ms": timeout_ms,
        "orientation_preprocessing": orientation_preprocessing,
        "orientation_search_steps": orientation_search_steps,
        "max_middle_models": max_middle_models if job.factor == 4 else None,
        "cross_compression_pruning": cross_compression_pruning if job.factor == 4 else None})


def persist_compressed_lift_result(job, result, root, seed, parameters):
    """Common independent audit/persistence for one-shot and staged completion."""
    payload = {
        "method": ("gf2_orientation_search" if result.verified and not result.z3_executed
                   and result.orientation_search_evaluations else
                   "affine_compressed_lift" if parameters.get("binary_backend") == "affine"
                   else "middle_mitm" if parameters.get("middle_backend") == "mitm"
                   else "dual_compression_rectangle" if parameters.get("work_kind") == "rectangle"
                   else "multilevel_z3_lift" if job.factor == 4 else "compressed_z3_lift"),
        "factor": job.factor, "seed": seed, "job": job.identifier,
        "compressed_a": list(job.compressed_a), "compressed_b": list(job.compressed_b),
        "algorithm_parameters": parameters,
        "content_profile": asdict(job.target),
        "source_A": list(job.source_a), "source_B": list(job.source_b),
        **asdict(result),
    }
    new = False
    if result.status == "SAT":
        if result.a is None or result.b is None:
            raise RuntimeError("compressed lift SAT has no candidate")
        check = verify_pqcp(result.a, result.b)
        if (not result.verified or not check.is_valid or check.profile != result.profile
                or check.profile != target_profile(job.target.L, job.target.k, job.target.eta)
                or compress_binary_sequence(result.a, job.factor) != job.compressed_a
                or compress_binary_sequence(result.b, job.factor) != job.compressed_b
                or pair_content(result.a, result.b) != (
                    job.target.a_even_ones, job.target.a_odd_ones,
                    job.target.b_even_ones, job.target.b_odd_ones)):
            raise RuntimeError("compressed lift persistence rejected unverified SAT")
        # Save the full candidate BEFORE any official append, so even an I/O
        # failure cannot turn a discovered model into an untraceable event.
        nonzero = list(check.nonzero_shifts)
        payload.update(A=list(result.a), B=list(result.b), profile=list(check.profile),
                       score=pqcp_objective(check.profile), nonzero_shifts=nonzero,
                       nonzero_values=[check.profile[u] for u in nonzero],
                       objective_components=pqcp_objective_breakdown(check.profile))
    destination = Path(root) / "results/compressed_lift" / (
        "L{}_seed{}_{}.json".format(result.L, seed, job.identifier)
    )
    atomic_write_json(destination, payload)
    if result.status == "SAT":
        new = append_verified_solution_if_new(result.L, result.a, result.b, root)
        if new:
            verified = Path(root) / "results/verified" / destination.name
            suffix = 1
            while verified.exists():
                verified = verified.with_name(destination.stem + "_{}".format(suffix) + ".json")
                suffix += 1
            atomic_write_json(verified, payload)
            print("找到新的 ({},4)-PQCP，已寫入 {}.txt".format(result.L, result.L), flush=True)
    return {
        "status": "LIFT_{}".format(result.status), "solved": result.verified,
        "z3_executed": result.z3_executed, "new_solution_written": new,
        "reason": result.reason, "orientation_variables": result.orientation_variables,
        "correlation_constraints": result.correlation_constraints,
        "construction_time": result.construction_time, "solve_time": result.solve_time,
        "gf2_applied": result.gf2_applied, "gf2_rank": result.gf2_rank,
        "gf2_free_bits": result.gf2_free_bits,
        "gf2_preprocessing_time": result.gf2_preprocessing_time,
        "orientation_search_evaluations": result.orientation_search_evaluations,
        "orientation_search_feasible": result.orientation_search_feasible,
        "orientation_search_time": result.orientation_search_time,
        "elapsed": result.construction_time + result.solve_time,
        "snapshot": str(destination),
        "factor": job.factor,
        "middle_models": getattr(result, "middle_models", 0),
        "unknown_children": getattr(result, "unknown_children", 0),
        "cross_pruning_applied": getattr(result, "cross_pruning_applied", False),
        "compatible_cross_pairs": getattr(result, "compatible_cross_pairs", 0),
        "middle_cross_pruned": getattr(result, "middle_cross_pruned", 0),
        "z3_calls": (result.middle_checks + result.binary_checks if job.factor == 4
                     else int(result.z3_executed)),
    }
