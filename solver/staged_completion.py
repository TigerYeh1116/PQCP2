"""Durable cooperative time slices for exact completion, without new pruning.

One lift turn performs at most one Z3 check (middle enumeration OR binary
completion). UNKNOWN goes to the back of the queue with doubled check budget.
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

    def __post_init__(self) -> None:
        for name in ("initial_ms", "maximum_ms", "max_middle_models", "rectangle_restarts",
                     "rectangle_moves", "rectangle_samples"):
            value = getattr(self,name)
            if not isinstance(value,int) or isinstance(value,bool) or value < (0 if name == "maximum_ms" else 1):
                raise ValueError("invalid staged completion {}".format(name))
        if not isinstance(self.cross_pruning,bool):
            raise ValueError("cross_pruning must be boolean")
        if not isinstance(self.rectangle_search,bool):
            raise ValueError("rectangle_search must be boolean")
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
        self._representatives = {}

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

    def _load_index(self) -> Dict[str, Dict[str, Any]]:
        """Index legacy files without renaming or transforming their states.

        The selected representative retains its own coordinates and blocked
        middle patterns. Other equivalent files stay intact for rollback;
        their histories are NOT merged without a coordinate mapping.
        """
        records = {path.stem: self.read(path.stem)
                   for path in sorted(self.directory.glob("*.json"))}
        representatives = {}
        phases = {}
        for key, record in records.items():
            identity = self._equivalence_identity(record["kind"], record["spec"])
            statuses = phases.setdefault(identity, set())
            statuses.add(record["phase"])
            if {"SAT", "UNSAT"}.issubset(statuses):
                raise ValueError("equivalent completion checkpoints disagree: SAT and UNSAT")
            previous = representatives.get(identity)
            if previous is None or self._resume_rank(record) > self._resume_rank(records[previous]):
                representatives[identity] = key
        self._representatives = representatives
        self.recovery_duplicates = len(records) - len(representatives)
        self._index_loaded = True
        return records

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
            self.read(previous)  # never silently ignore corruption
            self.duplicates_skipped += 1
            return None
        state = ({"order":["enum" if spec["factor"]==4 else "binary"],
                  "enum":_work(self.policy), "binary":_work(self.policy),
                  "children":[], "cross_pruned":0, "total_checks":0}
                 if kind=="lift" else {"work":_work(self.policy)})
        record = {"kind":kind,"spec":spec,"L":self.L,"seed":self.seed,
                  "phase":"READY","state":state,"turns":0}
        self.save(key,record)
        self._representatives[identity] = key
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
        records = self._load_index()
        for key in sorted(self._representatives.values()):
            record = records[key]
            if record["phase"] not in ("READY","WAITING"):
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
        self.save(key,record)
        report.update(staged=True, phase=record["phase"], checkpoint=str(self.path(key)), **detail)
        if detail["work_kind"] == "rectangle":
            report["status"] = "RECTANGLE_SAT" if result.verified else "RECTANGLE_MISS"
            report["elapsed"] = detail["rectangle_elapsed"]
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


def advance_lift(
    job: CompressedLiftJob, state: Dict[str, Any], policy: SlicePolicy,
) -> Tuple[MultilevelLiftResult, Dict[str, Any]]:
    """One exact Z3 check, or a zero-check mathematical/budget decision.

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
            status,L,k,eta,L//2,L//4+1 if job.factor==4 else L//2-1,4,
            construction,solve_time,bool(enum_checks or binary_checks),
            a=child.a if child else None,b=child.b if child else None,
            profile=child.profile if child else None,verified=bool(child and child.verified),reason=reason,
            cross_pattern_count=child.cross_pattern_count if child else 0,
            preferred_values=preferred_values,
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
        if len(state["children"])>=policy.max_middle_models:
            work["status"]="LIMIT"
        else:
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
            child=solve_compressed_lift(ma,mb,k,eta,budget,job.target,**options)
            preferred_values=child.preferred_values
            detail["preferred_values"] = preferred_values
            solve_time=child.solve_time
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
