"""Durable cooperative time slices for exact completion, without new pruning.

One lift turn performs at most one Z3 check (middle enumeration OR binary
completion). Optional exact MITM/affine stages can finish without Z3.
UNKNOWN goes to the back of the queue with doubled check budget.
At the configured maximum it is parked on disk, not treated as UNSAT. The
next run reopens parked timeouts; a model-count limit reopens only if raised.

JSON stores mathematical subproblems and audited middle patterns, not Z3
objects/pickle. Solver assertions are reconstructed each turn; learned clauses
are not checkpointed. This trades construction overhead for bounded live
solver memory and recoverability. Encoding and independent verification are
shared with the existing one-shot implementation.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple

from .checkpoint import atomic_write_json
from .compressed_completion import CompressedLiftJob, persist_compressed_lift_result
from .compressed_lift import solve_compressed_lift
from .multilevel_lift import MultilevelLiftResult, build_middle_solver
from .structured_energy import fold_profile, integer_periodic_autocorrelation_profile, target_profile
from .target_profiles import TargetContentProfile
from .disk_index import DiskIndex


@dataclass(frozen=True)
class SlicePolicy:
    initial_ms: int = 1000
    maximum_ms: int = 60000
    max_middle_models: int = 16
    cross_pruning: bool = True
    rectangle_search: bool = True
    rectangle_restarts: int = 4
    rectangle_moves: int = 1000
    rectangle_samples: int = 8
    # Standalone callers retain the old policy; main enables the new stages.
    middle_mitm: bool = False
    affine_lift: bool = False
    mitm_index_entries: int = 250000
    affine_max_assignments: int = 1048576
    affine_backend: str = "auto"
    mitm_batch_size: int = 4
    orientation_preprocessing: bool = True
    orientation_search_steps: int = 0
    # Opt-in: total raw factor-2 models, INCLUDING the first SAT. Additional
    # checks share a small per-job budget, separate from first-solution work.
    lift_max_models: int = 1
    lift_extra_ms: int = 200

    def __post_init__(self) -> None:
        for name in ("initial_ms", "maximum_ms", "max_middle_models", "rectangle_restarts",
                     "rectangle_moves", "rectangle_samples", "mitm_index_entries",
                     "affine_max_assignments", "lift_max_models", "lift_extra_ms"):
            value = getattr(self,name)
            if not isinstance(value,int) or isinstance(value,bool) or value < (0 if name == "maximum_ms" else 1):
                raise ValueError("invalid staged completion {}".format(name))
        if not isinstance(self.cross_pruning,bool):
            raise ValueError("cross_pruning must be boolean")
        if not isinstance(self.rectangle_search,bool):
            raise ValueError("rectangle_search must be boolean")
        if not isinstance(self.middle_mitm,bool) or not isinstance(self.affine_lift,bool):
            raise ValueError("middle_mitm and affine_lift must be boolean")
        if self.affine_backend not in ("auto","python","native"):
            raise ValueError("invalid affine backend")
        if type(self.orientation_preprocessing) is not bool:
            raise ValueError("orientation_preprocessing must be boolean")
        if type(self.orientation_search_steps) is not int or self.orientation_search_steps < 0:
            raise ValueError("orientation_search_steps must be a nonnegative integer")
        if self.orientation_search_steps and not self.orientation_preprocessing:
            raise ValueError("orientation search requires GF(2) preprocessing")
        if type(self.mitm_batch_size) is not int or not 1<=self.mitm_batch_size<=16:
            raise ValueError("mitm_batch_size must be in 1..16")
        if self.rectangle_samples > 1024 or self.rectangle_moves > 2**31-1:
            raise ValueError("rectangle work limit exceeds kernel bounds")

    def clamp(self, milliseconds: int) -> int:
        return min(milliseconds,self.maximum_ms) if self.maximum_ms else milliseconds


def _work(policy: SlicePolicy) -> Dict[str, Any]:
    return {"status":"READY", "budget_ms":policy.clamp(policy.initial_ms), "attempts":0}


def _unknown(work: Dict[str, Any], policy: SlicePolicy) -> None:
    """Park at the ceiling; UNKNOWN never becomes UNSAT through budgeting."""
    if policy.maximum_ms and work["budget_ms"] >= policy.maximum_ms:
        work["status"] = "WAITING"
    else:
        work["budget_ms"] = policy.clamp(work["budget_ms"]*2)
        work["status"] = "READY"


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def _job_key(kind: str, spec: Dict[str, Any]) -> str:
    """Identify the fixed subproblem, not the incidental source provenance."""
    identity = ({name: spec[name] for name in ("factor", "compressed_a", "compressed_b", "target")}
                if kind == "lift" else spec)
    return _digest((kind, identity))


def job_from_spec(spec: Dict[str, Any]) -> CompressedLiftJob:
    """Restore JSON arrays as the immutable tuples used by the exact solver."""
    return CompressedLiftJob(tuple(spec["compressed_a"]),tuple(spec["compressed_b"]),
                             TargetContentProfile(**spec["target"]),
                             tuple(spec["source_a"]),tuple(spec["source_b"]),spec["factor"])


class CompletionWorkStore:
    """Per-job atomic checkpoints; all queued work is saved before enqueue.

    Owner is one search process for a given root/L/seed. A checksum detects
    damaged files; malformed checkpoints fail closed without being overwritten.
    Completed records are retained to avoid solving identical jobs on resume.
    """

    def __init__(self, root: Path, L: int, seed: int, policy: SlicePolicy,
                 deduplicate_equivalents: bool = True) -> None:
        self.root, self.L, self.seed, self.policy = Path(root),L,seed,policy
        self.directory = self.root/"checkpoints/z3"/"L{}_seed{}".format(L,seed)
        if not isinstance(deduplicate_equivalents, bool):
            raise ValueError("deduplicate_equivalents must be boolean")
        self.deduplicate_equivalents = deduplicate_equivalents
        self.duplicates_skipped = 0
        self.recovery_duplicates = 0
        self._index_loaded = False
        self._representatives = DiskIndex()

    def _equivalence_identity(self, kind: str, spec: Dict[str, Any]) -> tuple:
        # Radius centres have extra coordinate-dependent constraints. They
        # deliberately retain EXACT identity, not the compressed orbit key.
        if kind == "lift" and self.deduplicate_equivalents:
            return kind, job_from_spec(spec).canonical_key
        return kind, _job_key(kind, spec)

    def _resume_rank(self, record: Dict[str, Any]) -> tuple:
        """Prefer a proof, then resumable work, then already invested progress."""
        terminal = record["phase"] in ("SAT", "UNSAT", "DONE")
        state = record["state"]
        if record["kind"] == "radius":
            ready = not terminal
        elif record["spec"]["factor"] == 2:
            ready = state["binary"]["status"] in ("READY", "WAITING")
        else:
            ready = (state["enum"]["status"] in ("READY", "WAITING")
                     or (state["enum"]["status"] == "LIMIT"
                         and len(state["children"]) < self.policy.max_middle_models)
                     or any(c["status"] in ("READY", "WAITING") for c in state["children"]))
        return terminal, ready or self._has_rectangle_work(record), state.get("total_checks", 0), record["turns"]

    def _has_rectangle_work(self, record: Dict[str, Any]) -> bool:
        return (record["kind"] == "lift" and record["spec"]["factor"] == 4
                and self.L >= 12 and self.L % 8 == 4
                and self.policy.rectangle_search and self.policy.cross_pruning
                and record["state"].get("rectangle",{}).get("restarts_done",0) < self.policy.rectangle_restarts)

    def _load_index(self) -> None:
        """Index legacy files without renaming or transforming their states.

        The selected representative retains its own coordinates and blocked
        middle patterns. Other equivalent files stay intact for rollback;
        their histories are NOT merged without a coordinate mapping.
        """
        self._index_loaded = False
        self._representatives.close()
        self._representatives = DiskIndex()
        count = 0
        for path in self.directory.glob('*.json'):
            key = path.stem; record = self.read(key); count += 1
            identity = self._equivalence_identity(record["kind"], record["spec"])
            previous = self._representatives.get(identity)
            phases = (previous['phases'] if previous else 0) | {'SAT':1, 'UNSAT':2}.get(record['phase'], 0)
            if phases == 3:
                raise ValueError("equivalent completion checkpoints disagree: SAT and UNSAT")
            rank = self._resume_rank(record)
            if (previous is None or rank > tuple(previous['rank'])
                    or (rank == tuple(previous['rank']) and key < previous['key'])):
                self._representatives[identity] = dict(key=key, rank=rank, phases=phases)
            elif phases != previous['phases']:
                self._representatives[identity] = dict(previous, phases=phases)
        self.recovery_duplicates = count - len(self._representatives)
        self._index_loaded = True

    def path(self, key: str) -> Path:
        if not isinstance(key,str) or len(key)!=64 or any(c not in "0123456789abcdef" for c in key):
            raise ValueError("invalid completion checkpoint key")
        return self.directory/(key+".json")

    def register(self, kind: str, spec: Dict[str, Any]) -> Optional[str]:
        """Return a new durable key, or None for an already recorded subproblem."""
        if kind not in ("lift","radius"):
            raise ValueError("unsupported completion job")
        if not self._index_loaded:
            self._load_index()
        key = _job_key(kind,spec)
        identity = self._equivalence_identity(kind, spec)
        previous = self._representatives.get(identity)
        if previous is not None:
            self.read(previous['key'])  # never silently ignore corruption
            self.duplicates_skipped += 1
            return None
        state = ({"order":["enum" if spec["factor"]==4 else "binary"],
                  "enum":_work(self.policy), "binary":_work(self.policy),
                  "children":[], "cross_pruned":0, "total_checks":0}
                 if kind=="lift" else {"work":_work(self.policy)})
        record = {"kind":kind,"spec":spec,"L":self.L,"seed":self.seed,
                  "phase":"READY","state":state,"turns":0}
        self.save(key,record)
        self._representatives[identity] = dict(key=key, rank=self._resume_rank(record), phases=0)
        return key

    def save(self, key: str, record: Dict[str, Any]) -> None:
        if key != _job_key(record["kind"],record["spec"]):
            raise ValueError("completion job identity changed")
        record["policy"] = asdict(self.policy)
        atomic_write_json(self.path(key),{"version":1,"record":record,"sha256":_digest(record)})

    def read(self, key: str) -> Dict[str, Any]:
        try:
            data = json.loads(self.path(key).read_text(encoding="utf-8"))
            record = data["record"]
            if (data["version"] != 1 or data["sha256"] != _digest(record)
                    or record["L"] != self.L or record["seed"] != self.seed
                    or key != _job_key(record["kind"],record["spec"])
                    or record["phase"] not in ("READY","WAITING","SAT","UNSAT","DONE")):
                raise ValueError("invalid version/checksum/identity")
            return record
        except (ValueError,KeyError,TypeError) as error:
            raise ValueError("invalid completion checkpoint {}: {}".format(self.path(key),error)) from error

    def recover(self) -> List[str]:
        """Reopen one representative per orbit; preserve unresolved histories."""
        keys = []
        self._load_index()
        for entry in self._representatives.values():
            key = entry['key']; record = self.read(key)
            if record["phase"] not in ("READY","WAITING"):
                continue
            if record.get("lift_enumeration"):
                self._enumeration_phase(record)
                self.save(key, record)
                if record["phase"] == "READY":
                    keys.append(key)
                continue
            state = record["state"]
            if record["kind"] == "radius":
                state["work"]["status"] = "READY"
                state["work"]["budget_ms"] = self.policy.clamp(state["work"]["budget_ms"])
                ready = True
            else:
                items = [("enum",state["enum"]),("binary",state["binary"])]
                items += [(i,c) for i,c in enumerate(state["children"])]
                for label,work in items:
                    applicable = label != ("binary" if record["spec"]["factor"]==4 else "enum")
                    if not applicable: continue
                    if work["status"]=="LIMIT" and len(state["children"])<self.policy.max_middle_models:
                        work["status"] = "READY"
                    if work["status"]=="WAITING": work["status"]="READY"
                    if work["status"]=="READY":
                        work["budget_ms"] = self.policy.clamp(work["budget_ms"])
                        if label not in state["order"]: state["order"].append(label)
                ready = bool(state["order"]) or self._has_rectangle_work(record)
            if ready:
                record["phase"]="READY"
                self.save(key,record)
                keys.append(key)
        return keys

    def close(self):
        self._representatives.close()
        self._index_loaded = False

    def finish_radius(self, key: str, record: Dict[str, Any], unknown: bool, result: str) -> None:
        work=record["state"]["work"]
        work["attempts"]+=1
        if unknown:
            _unknown(work,self.policy)
            record["phase"]="READY" if work["status"]=="READY" else "WAITING"
        else:
            work["status"]="DONE"
            record["phase"]="DONE"
        record["turns"]+=1
        record["last_status"]=result
        self.save(key,record)

    def advance_lift(self, key: str) -> Dict[str, Any]:
        record=self.read(key)
        if record["phase"]!="READY" or record["kind"]!="lift":
            raise ValueError("lift is not ready")
        job=job_from_spec(record["spec"])
        if record.get("lift_enumeration"):
            return self._advance_enumeration(key, record, job)
        turn = _rectangle_turn(job,record["state"],self.policy,self.seed)
        result,detail = turn if turn is not None else advance_lift(job,record["state"],self.policy)
        record["phase"]=(result.status if result.status in ("SAT","UNSAT") else
                         "READY" if record["state"]["order"] or (turn is not None and self._has_rectangle_work(record))
                         else "WAITING")
        record["turns"]+=1
        record["last_status"]=result.status
        # Save verified candidate/official result BEFORE marking the job SAT.
        report=persist_compressed_lift_result(job,result,self.root,self.seed,{
            "staged":True, **asdict(self.policy), **detail})
        if result.status == "SAT" and job.factor == 2 and self.policy.lift_max_models > 1:
            record["lift_enumeration"] = {"models": [], "spent_ms": 0.0,
                                          "checks": 0, "status": "READY"}
            self._remember_lift(record, job, result, report)
            self._enumeration_phase(record)
            report.update(lift_models=1, lift_enumeration_status=record["lift_enumeration"]["status"])
        self.save(key,record)
        report.update(staged=True, phase=record["phase"], checkpoint=str(self.path(key)), **detail)
        if detail["work_kind"] == "rectangle":
            report["status"] = "RECTANGLE_SAT" if result.verified else "RECTANGLE_MISS"
            report["elapsed"] = detail["rectangle_elapsed"]
        return report

    def _remember_lift(self, record, job, result, report):
        """Block only an independently audited, already persisted full model."""
        from .lift_enumeration import orientation_assignment
        from .solution_equivalence import pqcp_equivalence_key
        direction = orientation_assignment(job, result.a, result.b)
        models = record["lift_enumeration"]["models"]
        if any(tuple(m["orientation"]) == direction for m in models):
            raise RuntimeError("enumeration returned an excluded model")
        models.append(dict(A=list(result.a), B=list(result.b), profile=list(result.profile),
                           orientation=list(direction),
                           equivalence_key=list(pqcp_equivalence_key(result.a, result.b)),
                           new_solution_written=report["new_solution_written"]))

    def _enumeration_phase(self, record):
        """A previous SAT remains a proof of existence, never becomes UNSAT.

        Limits stop this optional continuation, not the main search. Terminal
        SAT jobs are not automatically reopened on later runs. UNKNOWN with
        budget left is parked and can resume from the exact saved exclusions.
        """
        progress = record["lift_enumeration"]
        if progress["status"] == "EXHAUSTED":
            record["phase"] = "SAT"
        elif len(progress["models"]) >= self.policy.lift_max_models:
            progress["status"], record["phase"] = "MODEL_LIMIT", "SAT"
        elif progress["spent_ms"] >= self.policy.lift_extra_ms:
            progress["status"], record["phase"] = "TIME_LIMIT", "SAT"
        else:
            progress["status"], record["phase"] = "READY", "READY"

    def _advance_enumeration(self, key, record, job):
        """One extra exact check per queue turn; no new heuristic or pruning.

        Every direction vector is audited from its saved full A/B before
        blocking. This rebuilds the old solver, not a retained clause database.
        Budget includes construction/audit/persistence; a check is bounded by
        remaining time, but Python work can overrun that soft wall budget.
        """
        from .lift_enumeration import orientation_assignment
        started = perf_counter()
        progress = record["lift_enumeration"]
        self._enumeration_phase(record)
        report = dict(status="LIFT_ENUM_" + progress["status"], solved=False,
                      z3_executed=False, z3_calls=0, new_solution_written=False,
                      work_kind="lift_enumeration", slice_ms=0, elapsed=0.0)
        if record["phase"] == "READY":
            excluded = []
            for model in progress["models"]:
                direction = orientation_assignment(job, model["A"], model["B"])
                if list(direction) != model["orientation"]:
                    raise ValueError("saved lift orientation does not match its full candidate")
                excluded.append(direction)
            remaining = max(1, int(self.policy.lift_extra_ms - progress["spent_ms"]))
            budget = min(self.policy.clamp(self.policy.initial_ms), remaining)
            result = solve_compressed_lift(job.compressed_a, job.compressed_b,
                job.target.k, job.target.eta, budget, job.target,
                orientation_preprocessing=self.policy.orientation_preprocessing,
                excluded_orientations=excluded)
            progress["checks"] += int(result.z3_executed)
            record["state"]["total_checks"] += int(result.z3_executed)
            if result.status == "SAT":
                report = persist_compressed_lift_result(job, result, self.root, self.seed,
                    {"staged": True, "work_kind": "lift_enumeration",
                     "excluded_models": len(excluded), **asdict(self.policy)})
                self._remember_lift(record, job, result, report)
            else:
                # Do not overwrite the previous SAT snapshot with a residual
                # UNSAT/UNKNOWN result: its scope is only unblocked models.
                progress["status"] = "EXHAUSTED" if result.status == "UNSAT" else "UNKNOWN"
                report.update(status="LIFT_ENUM_" + progress["status"],
                              z3_executed=result.z3_executed,
                              z3_calls=int(result.z3_executed), reason=result.reason,
                              construction_time=result.construction_time, solve_time=result.solve_time)
            progress["spent_ms"] += (perf_counter() - started) * 1000
            status = progress["status"]
            self._enumeration_phase(record)
            if status == "UNKNOWN" and record["phase"] == "READY":
                progress["status"], record["phase"] = "UNKNOWN", "WAITING"
            report.update(slice_ms=budget, elapsed=perf_counter() - started)
        record["turns"] += 1
        record["last_status"] = report["status"]
        self.save(key, record)
        report.update(staged=True, phase=record["phase"], work_kind="lift_enumeration",
                      checkpoint=str(self.path(key)), lift_models=len(progress["models"]),
                      lift_enumeration_status=progress["status"])
        return report


def _rectangle_turn(
    job: CompressedLiftJob, state: Dict[str, Any], policy: SlicePolicy, seed: int,
) -> Optional[Tuple[MultilevelLiftResult, Dict[str, Any]]]:
    """A bounded C restart before exact Z3; a miss leaves the ENTIRE job open.

    Restarts rotate through compatible labeled cross patterns. On interruption
    an unfinished short turn is replayed deterministically; completed turns
    and the best candidate are checkpointed. No Z3 branch is blocked here.
    """
    L = job.target.L
    if (not policy.rectangle_search or not policy.cross_pruning or job.factor != 4
            or L < 12 or L % 8 != 4):
        return None
    progress = state.get("rectangle", {})
    index = progress.get("restarts_done", 0)
    if index >= policy.rectangle_restarts:
        return None
    actual = tuple(x+y for x,y in zip(
        integer_periodic_autocorrelation_profile(job.compressed_a),
        integer_periodic_autocorrelation_profile(job.compressed_b)))
    if actual != fold_profile(target_profile(L,job.target.k,job.target.eta),4):
        return None
    from .cross_compression import compatible_cross_compressions
    from .rectangle_search import initialize_from_margins, search_rectangles
    import random
    started = perf_counter()
    patterns = compatible_cross_compressions(job.compressed_a,job.compressed_b,
                                            job.target.k,job.target.eta,job.target)
    if not patterns:
        return None  # original exact precheck will decide, not the heuristic
    base = int(_digest((seed,job.identifier,"rectangle-v1"))[:16],16)
    order = list(range(len(patterns)))
    random.Random(base).shuffle(order)
    selected = order[index % len(order)]
    row_a,row_b = patterns[selected]
    derived = int(_digest((base,index))[:16],16)
    a = initialize_from_margins(job.compressed_a,row_a,derived)
    b = initialize_from_margins(job.compressed_b,row_b,derived+1)
    run = search_rectangles(a,b,job.target,derived,policy.rectangle_moves,policy.rectangle_samples)
    if not progress or run.best_energy < progress["best_energy"]:
        progress.update(best_energy=run.best_energy,best_a=list(run.a),best_b=list(run.b),
                        best_profile=list(run.profile),best_cross_index=selected,
                        best_cross_a=list(row_a),best_cross_b=list(row_b))
    progress.update(restarts_done=index+1,pattern_count=len(patterns),last_cross_index=selected,
                    moves=progress.get("moves",0)+run.moves,
                    evaluations=progress.get("evaluations",0)+run.evaluations)
    state["rectangle"] = progress
    result = MultilevelLiftResult(
        "SAT" if run.verified else "UNKNOWN",L,job.target.k,job.target.eta,0,0,0,0.0,0.0,False,
        a=run.a if run.verified else None,b=run.b if run.verified else None,
        profile=run.profile if run.verified else None,verified=run.verified,
        reason=None if run.verified else "bounded rectangle miss; no Z3 branch excluded",
        cross_a=row_a,cross_b=row_b)
    detail = {"work_kind":"rectangle","slice_ms":0,"rectangle_restart":index+1,
              "rectangle_restart_limit":policy.rectangle_restarts,"rectangle_pattern":selected,
              "rectangle_energy":run.best_energy,"rectangle_initial_energy":run.initial_energy,
              "rectangle_moves":run.moves,"rectangle_evaluations":run.evaluations,
              "rectangle_no_moves":run.no_legal_moves,"rectangle_elapsed":perf_counter()-started}
    return result,detail


def _audit_middle(a: Tuple[int, ...], b: Tuple[int, ...], job: CompressedLiftJob) -> None:
    L=job.target.L
    n=L//4
    if len(a)!=2*n or len(b)!=2*n or any(type(x) is not int or x not in (-2,0,2) for x in a+b):
        raise ValueError("invalid saved middle pattern")
    if any(a[i]+a[i+n]!=job.compressed_a[i] or b[i]+b[i+n]!=job.compressed_b[i] for i in range(n)):
        raise ValueError("middle pattern does not fold to this job")
    if tuple(x+y for x,y in zip(integer_periodic_autocorrelation_profile(a),
                               integer_periodic_autocorrelation_profile(b))) != fold_profile(
                                   target_profile(L,job.target.k,job.target.eta),2):
        raise ValueError("middle pattern does not satisfy folded target")
    for word,even,odd in ((a,job.target.a_even_ones,job.target.a_odd_ones),
                          (b,job.target.b_even_ones,job.target.b_odd_ones)):
        if sum(word[0::2])!=L//2-2*even or sum(word[1::2])!=L//2-2*odd:
            raise ValueError("middle pattern violates saved content")


def _mitm_turn(job: CompressedLiftJob, state: Dict[str, Any],
               policy: SlicePolicy, budget: int) -> dict:
    """Replay a deterministic MITM cursor; equivalent children retain a proof owner.

    A canonical duplicate is skipped only if its representative already lives
    in children (including UNKNOWN children). All possible cross patterns are
    necessary consequences of the fixed compression, target and content, so
    the existing full factor-2 equivalence key identifies the same lift task.
    A deferred MITM permanently switches this enum work to the old Z3 path;
    that path blocks only saved raw models, not unproved missing branches.
    """
    from .middle_mitm import ORDER_VERSION, next_middle
    from .compressed_equivalence import compressed_job_equivalence_key
    work = state["enum"]
    progress = work.get("mitm", {})
    if progress.get("order_version") != ORDER_VERSION:
        progress = {"order_version":ORDER_VERSION, "cursor":0}
    excluded = set()
    for old in state["children"]:
        a,b = tuple(old["a"]),tuple(old["b"])
        _audit_middle(a,b,job)
        excluded.add(compressed_job_equivalence_key(a,b,job.target,2))
    result = next_middle(job.compressed_a,job.compressed_b,job.target,budget,
                         policy.mitm_index_entries,progress["cursor"],excluded,
                         batch_size=min(policy.mitm_batch_size,policy.max_middle_models-len(state["children"])))
    progress.update(cursor=result["cursor"],
                    duplicates=progress.get("duplicates",0)+result["duplicates"],
                    last_status=result["status"],last_elapsed=result["elapsed"])
    work["mitm"] = progress
    if result["status"] == "DEFERRED":
        work["mitm_fallback"] = result["reason"]
    elif result["status"] == "EXHAUSTED":
        work["status"] = "UNSAT"  # enumeration only; children still own proofs
    else:
        labels=[]
        for a,b in result["matches"]:
            _audit_middle(a,b,job)
            labels.append(len(state["children"]))
            state["children"].append({**_work(policy),"a":list(a),"b":list(b),
                                      "source":"middle_mitm"})
        state["order"][0:0]=labels
        work["budget_ms"] = policy.clamp(policy.initial_ms)
        if result["exhausted"]:
            work["status"]="UNSAT"  # children remain individually unresolved
        else:
            state["order"].append("enum")
    return result


def _affine_turn(ma: tuple, mb: tuple, job: CompressedLiftJob, work: dict,
                 policy: SlicePolicy, budget: int, cross: Optional[tuple]):
    """Resume only a checksum-protected cursor for the same mathematical task.

    Changing the order version or cross constraints restarts this cheap stage
    conservatively; a cursor is never transferred to a different lift space.
    A size cap means deferred-to-Z3, not an infeasibility certificate.
    """
    from .affine_lift import ORDER_VERSION, solve_affine_lift
    identity = _digest((ORDER_VERSION,ma,mb,asdict(job.target),cross))
    previous = work.get("affine",{})
    cursor = previous.get("next_index",0) if previous.get("identity") == identity else 0
    result = solve_affine_lift(ma,mb,job.target.k,job.target.eta,budget,job.target,
                               cross_compressions=cross,
                               max_assignments=policy.affine_max_assignments,start_index=cursor,
                               backend=policy.affine_backend)
    work["affine"] = dict(identity=identity,order_version=ORDER_VERSION,
                          next_index=result.next_index,rank=result.rank,free_bits=result.free_bits,
                          assignments=result.assignments,last_examined=result.examined,
                          status=result.status,deferred=result.deferred,scan_backend=result.scan_backend)
    return result


def advance_lift(
    job: CompressedLiftJob, state: Dict[str, Any], policy: SlicePolicy,
) -> Tuple[MultilevelLiftResult, Dict[str, Any]]:
    """One exact stage, with at most one Z3 check as fallback.

    All middle models discovered are blocked in enumeration, INCLUDING pending
    ones, but each pending child remains separately queued or parked. Therefore
    exhausted enumeration is globally UNSAT only when ALL children are UNSAT.
    """
    started=perf_counter()
    L,k,eta=job.target.L,job.target.k,job.target.eta
    desired=target_profile(L,k,eta)
    cross=None
    solve_time=0.0
    enum_checks=binary_checks=0
    preferred_values=0
    detail={"work_kind":"precheck","slice_ms":0}
    ma=mb=None

    def finish(status,reason=None,child=None):
        construction=max(0.0,perf_counter()-started-solve_time)
        final_cross=(None,None)
        if child is not None and child.verified and cross is not None:
            final_cross=tuple(tuple(sum(1-2*x for x in bits[r::4]) for r in range(4))
                              for bits in (child.a,child.b))
            if final_cross not in cross: raise RuntimeError("staged SAT failed cross audit")
        result=MultilevelLiftResult(
            status,L,k,eta,L//2,L//4+1 if job.factor==4 else detail.get("encoded_correlation_constraints",L//2-1),4,
            construction,solve_time,bool(enum_checks or binary_checks),
            a=child.a if child else None,b=child.b if child else None,
            profile=child.profile if child else None,verified=bool(child and child.verified),reason=reason,
            cross_pattern_count=child.cross_pattern_count if child else 0,
            preferred_values=preferred_values,
            gf2_applied=bool(detail.get("gf2_applied", False)),
            gf2_rank=detail.get("gf2_rank", 0),
            gf2_free_bits=detail.get("gf2_free_bits", 0),
            gf2_preprocessing_time=detail.get("gf2_preprocessing_time", 0.0),
            orientation_search_evaluations=detail.get("orientation_search_evaluations", 0),
            orientation_search_feasible=detail.get("orientation_search_feasible", 0),
            orientation_search_time=detail.get("orientation_search_time", 0.0),
            orientation_hint=child.orientation_hint if child else None,
            orientation_exact_assignments=detail.get("orientation_exact_assignments", 0),
            orientation_exact_time=detail.get("orientation_exact_time", 0.0),
            middle_models=len(state["children"]),middle_checks=enum_checks,binary_checks=binary_checks,
            unknown_children=sum(c["status"] in ("READY","WAITING") for c in state["children"]),
            middle_a=ma,middle_b=mb,cross_pruning_applied=cross is not None,
            compatible_cross_pairs=len(cross or ()),middle_cross_pruned=state["cross_pruned"],
            cross_a=final_cross[0],cross_b=final_cross[1])
        return result,detail

    actual=tuple(x+y for x,y in zip(integer_periodic_autocorrelation_profile(job.compressed_a),
                                   integer_periodic_autocorrelation_profile(job.compressed_b)))
    if actual!=fold_profile(desired,job.factor):
        return finish("UNSAT","fixed compressed pair does not match folded target")
    if job.factor==4 and policy.cross_pruning and (L//4)%2:
        from .cross_compression import compatible_cross_compressions
        cross=compatible_cross_compressions(job.compressed_a,job.compressed_b,k,eta,job.target)
        if not cross: return finish("UNSAT","no exact cross-compatible patterns")
    if not state["order"]:
        raise ValueError("ready lift has no queued subproblem")
    label=state["order"].pop(0)
    work=state[label] if isinstance(label,str) else state["children"][label]
    if work["status"]!="READY": raise ValueError("invalid subproblem queue")
    budget=policy.clamp(work["budget_ms"])
    work["budget_ms"]=budget
    detail={"work_kind":label if isinstance(label,str) else "child_{}".format(label),"slice_ms":budget}

    if label=="enum":
        mitm = None
        if len(state["children"])>=policy.max_middle_models:
            work["status"]="LIMIT"
        elif (policy.middle_mitm and L%8==4 and not work.get("mitm_fallback")):
            mitm = _mitm_turn(job,state,policy,budget)
            detail.update(middle_backend="mitm",mitm_status=mitm["status"],
                          mitm_batch_count=len(mitm["matches"]),
                          mitm_cursor=mitm["cursor"],mitm_duplicates=mitm["duplicates"],
                          mitm_elapsed=mitm["elapsed"])
            if mitm["status"] == "FOUND":
                ma,mb = mitm["a"],mitm["b"]
            if mitm["status"] != "DEFERRED":
                work["attempts"] += 1
        if work["status"] == "READY" and (mitm is None or mitm["status"] == "DEFERRED"):
            detail["middle_backend"] = "z3"
            if work.get("mitm_fallback"):
                detail["mitm_fallback"] = work["mitm_fallback"]
            import z3
            solver,rows,_=build_middle_solver(job.compressed_a,job.compressed_b,desired,job.target)
            for old in state["children"]:
                a,b=tuple(old["a"]),tuple(old["b"])
                _audit_middle(a,b,job)
                solver.add(z3.Or([z3.Sum([z3.If(f,x,0) for f,x in entry])!=value
                                  for row,word in zip(rows,(a,b)) for entry,value in zip(row,word)]))
            solver.set(timeout=budget)
            tick=perf_counter()
            outcome=solver.check()
            solve_time=perf_counter()-tick
            enum_checks=1
            work["attempts"]+=1
            if outcome==z3.unsat:
                work["status"]="UNSAT"
            elif outcome==z3.sat:
                model=solver.model()
                ma,mb=(tuple(sum(x for f,x in entry if z3.is_true(model.evaluate(f,model_completion=True)))
                             for entry in row) for row in rows)
                _audit_middle(ma,mb,job)
                child={**_work(policy),"a":list(ma),"b":list(mb)}
                state["children"].append(child)
                # A fresh child gets its first short attempt before retrying
                # old difficult children. Enumeration itself remains queued.
                state["order"].insert(0,len(state["children"])-1)
                work["budget_ms"]=policy.clamp(policy.initial_ms)
                state["order"].append("enum")
            else:
                work["reason"]=solver.reason_unknown()
                _unknown(work,policy)
                if work["status"]=="READY": state["order"].append("enum")
    else:
        options={}
        if label=="binary":
            ma,mb=job.compressed_a,job.compressed_b
        else:
            ma,mb=tuple(work["a"]),tuple(work["b"])
            _audit_middle(ma,mb,job)
        eligible=True
        if cross is not None:
            from .cross_compression import middle_matches_cross
            patterns=tuple((a,b) for a,b in cross if middle_matches_cross(ma,a) and middle_matches_cross(mb,b))
            eligible=bool(patterns)
            options["cross_compressions"]=patterns
        if not eligible:
            work["status"]="UNSAT"
            state["cross_pruned"]+=1
        else:
            if policy.rectangle_search and state.get("rectangle"):
                options["preferred_pair"] = (state["rectangle"]["best_a"],state["rectangle"]["best_b"])
            child = None
            affine_time = 0.0
            if policy.affine_lift and L%8==4:
                candidate = _affine_turn(ma,mb,job,work,policy,budget,
                                         options.get("cross_compressions"))
                detail.update(affine_rank=candidate.rank,affine_free_bits=candidate.free_bits,
                              affine_backend=candidate.scan_backend,
                              affine_assignments=candidate.assignments,affine_examined=candidate.examined,
                              affine_next_index=candidate.next_index,affine_status=candidate.status,
                              affine_deferred=candidate.deferred,
                              affine_elapsed=candidate.construction_time+candidate.solve_time)
                if not candidate.deferred:
                    child = candidate
                    detail["binary_backend"] = "affine"
                else:
                    affine_time = candidate.solve_time
                    detail["affine_fallback"] = candidate.reason
            if child is None:
                detail["binary_backend"] = "z3"
                if L % 4 == 2:
                    options["orientation_preprocessing"] = policy.orientation_preprocessing
                    if work.get("orientation_hint") is not None:
                        options["preferred_pair"] = work["orientation_hint"]
                    if policy.orientation_search_steps and not work.get("orientation_searched", False):
                        options.update(orientation_search_steps=policy.orientation_search_steps,
                                       orientation_search_seed=int(job.identifier[:8], 16))
                        options.setdefault("preferred_pair", (job.source_a, job.source_b))
                child=solve_compressed_lift(ma,mb,k,eta,budget,job.target,**options)
                if options.get("orientation_search_steps"):
                    work["orientation_searched"] = True
                    work["orientation_hint"] = child.orientation_hint
                if child.verified and not child.z3_executed and child.orientation_search_evaluations:
                    detail["binary_backend"] = "orientation_search"
                if child.orientation_exact_time:
                    detail["binary_backend"] = "native_odd_lift"
                detail.update(orientation_exact_assignments=child.orientation_exact_assignments,
                              orientation_exact_time=child.orientation_exact_time)
                detail.update(orientation_search_evaluations=child.orientation_search_evaluations,
                              orientation_search_feasible=child.orientation_search_feasible,
                              orientation_search_time=child.orientation_search_time)
                detail.update(gf2_applied=child.gf2_applied, gf2_rank=child.gf2_rank,
                              gf2_free_bits=child.gf2_free_bits,
                              gf2_preprocessing_time=child.gf2_preprocessing_time,
                              encoded_correlation_constraints=child.correlation_constraints)
            preferred_values=child.preferred_values
            detail["preferred_values"] = preferred_values
            solve_time=child.solve_time+affine_time
            binary_checks=int(child.z3_executed)
            work["attempts"]+=1
            work["reason"]=child.reason
            if child.status=="SAT":
                state["total_checks"]+=binary_checks
                return finish("SAT",child=child)
            if child.status=="UNSAT":
                work["status"]="UNSAT"
            else:
                _unknown(work,policy)
                if work["status"]=="READY": state["order"].append(label)
    state["total_checks"]+=enum_checks+binary_checks
    done=(state["binary"]["status"]=="UNSAT" if job.factor==2 else
          state["enum"]["status"]=="UNSAT" and all(c["status"]=="UNSAT" for c in state["children"]))
    if done: return finish("UNSAT","all exact subproblems exhausted")
    return finish("UNKNOWN","yielded to other work" if state["order"] else
                  "unresolved work saved; time/model budget reached")
