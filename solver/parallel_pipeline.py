"""Spawn-based production orchestration for independent PQCP trajectories.

This module deliberately does not implement a search move.  Each child runs
the existing :func:`solver.pipeline.run_pipeline` unchanged, with repair and
Z3 disabled, below a private worker root.  The parent process treats all child
output as untrusted observations: it recomputes the exact correlation profile,
objective, and independent verification before selecting a global best or
writing the project's official ``L.txt``.

The use of the ``spawn`` multiprocessing context is explicit.  It is the safe
and portable context for macOS, and prevents children from inheriting a live
SA/RNG object.  Worker seeds are ``base_seed + index * 10**12``; consequently
adding workers does not change any previously assigned trajectory.
"""

from dataclasses import dataclass
import json
import math
import multiprocessing as mp
from multiprocessing.connection import Connection, wait
import os
from pathlib import Path
import signal
from time import perf_counter
import traceback
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .checkpoint import atomic_write_json
from .correlation import full_correlation_profile
from .objective import pqcp_objective
from .pipeline import PipelineConfig, run_pipeline
from .search_runner import append_verified_solution_if_new
from .verifier import verify_pqcp


SEED_STRIDE = 10 ** 12


class ParallelWorkerError(RuntimeError):
    """Raised after parent-side merging when one or more workers failed."""

    def __init__(self, result: "ParallelPipelineResult") -> None:
        self.result = result
        details = "; ".join(
            "worker {}: {}".format(report.worker_index, report.error)
            for report in result.failures
        )
        super().__init__("parallel PQCP worker failure: {}".format(details))


@dataclass(frozen=True)
class ParallelPipelineConfig:
    """Parent-owned settings for a finite independent-worker portfolio.

    ``repair`` and ``z3`` are intentionally not configurable here.  The first
    production version measures independent structured-SA trajectories only;
    every child receives ``repair=False`` and ``z3=False``.
    """

    L: int
    workers: int
    seconds: float
    base_seed: int = 123
    root: Path = Path(".")
    gcp: bool = True
    enhanced: bool = False
    reference_port: bool = False
    checkpoint_interval: float = 60.0
    progress_interval: float = 60.0
    interrupt_grace_seconds: float = 10.0
    raise_on_worker_failure: bool = True
    run_name: Optional[str] = None
    resume: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.L, int) or isinstance(self.L, bool) or self.L <= 0:
            raise ValueError("L must be a positive integer")
        if (
            not isinstance(self.workers, int)
            or isinstance(self.workers, bool)
            or self.workers <= 0
        ):
            raise ValueError("workers must be a positive integer")
        if math.isnan(self.seconds) or self.seconds <= 0:
            raise ValueError("seconds must be a positive number")
        if not isinstance(self.base_seed, int) or isinstance(self.base_seed, bool):
            raise ValueError("base_seed must be an integer")
        for name, value in (
            ("checkpoint_interval", self.checkpoint_interval),
            ("progress_interval", self.progress_interval),
            ("interrupt_grace_seconds", self.interrupt_grace_seconds),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("{} must be a finite positive number".format(name))
        if self.enhanced and self.gcp:
            raise ValueError("the existing pipeline does not combine enhanced mode with GCP")
        if self.reference_port and (self.enhanced or self.gcp):
            raise ValueError("reference_port cannot be combined with GCP or enhanced mode")
        if not isinstance(self.resume, bool):
            raise ValueError("resume must be boolean")
        if self.run_name is not None and (
            not self.run_name
            or self.run_name in (".", "..")
            or Path(self.run_name).name != self.run_name
        ):
            raise ValueError("run_name must be one safe path component")


@dataclass(frozen=True)
class ReviewedCandidate:
    """One child candidate after exact parent-side recomputation."""

    a: Tuple[int, ...]
    b: Tuple[int, ...]
    profile: Tuple[int, ...]
    score: int
    verified: bool
    source: str


@dataclass(frozen=True)
class WorkerReport:
    """Parent-reviewed result of one isolated child process."""

    worker_index: int
    seed: int
    worker_root: Path
    ok: bool
    interrupted: bool
    elapsed: float
    iteration: int
    restart_index: int
    checkpoint_path: Optional[Path]
    best: Optional[ReviewedCandidate]
    verified_candidates: Tuple[ReviewedCandidate, ...]
    error: Optional[str] = None


@dataclass(frozen=True)
class ParallelPipelineResult:
    """Aggregate global best and verified discoveries from all workers."""

    L: int
    base_seed: int
    workers: int
    seconds: float
    elapsed: float
    interrupted: bool
    reports: Tuple[WorkerReport, ...]
    global_best: Optional[ReviewedCandidate]
    global_best_worker: Optional[int]
    new_solution_count: int
    summary_path: Path
    start_method: str = "spawn"

    @property
    def failures(self) -> Tuple[WorkerReport, ...]:
        """Return every worker whose process or parent integrity review failed."""
        return tuple(report for report in self.reports if not report.ok)


def derived_worker_seed(base_seed: int, worker_index: int) -> int:
    """Return a stable seed whose assignment is unchanged when workers are added."""
    if not isinstance(base_seed, int) or isinstance(base_seed, bool):
        raise ValueError("base_seed must be an integer")
    if (
        not isinstance(worker_index, int)
        or isinstance(worker_index, bool)
        or worker_index < 0
    ):
        raise ValueError("worker_index must be a non-negative integer")
    return base_seed + worker_index * SEED_STRIDE


def parallel_run_root(config: ParallelPipelineConfig) -> Path:
    """Return the private directory holding every isolated child root."""
    name = config.run_name or "L{}_seed{}".format(config.L, config.base_seed)
    return Path(config.root) / "parallel" / name


def isolated_worker_root(config: ParallelPipelineConfig, worker_index: int) -> Path:
    """Return a deterministic root disjoint from all other worker roots."""
    seed = derived_worker_seed(config.base_seed, worker_index)
    return parallel_run_root(config) / "workers" / "w{:03d}_seed{}".format(
        worker_index, seed
    )


def run_parallel_pipeline(
    config: ParallelPipelineConfig,
    progress_callback: Optional[Callable[[Dict[str, object]], None]] = None,
) -> ParallelPipelineResult:
    """Run independent existing pipelines and merge only parent-verified output.

    A terminal ``Ctrl+C`` is relayed to live children as ``SIGINT``.  Their
    existing pipeline catches it and atomically saves each private checkpoint.
    The parent waits for a bounded grace period before force-stopping a broken
    worker; successfully returned candidates are still reviewed and merged.
    """
    context = mp.get_context("spawn")
    processes: Dict[int, mp.Process] = {}
    receivers: Dict[int, Connection] = {}
    started = perf_counter()
    try:
        for worker_index in range(config.workers):
            receiver, sender = context.Pipe(duplex=False)
            payload = {
                "L": config.L,
                "seed": derived_worker_seed(config.base_seed, worker_index),
                "seconds": config.seconds,
                "worker_index": worker_index,
                "worker_root": str(isolated_worker_root(config, worker_index)),
                "gcp": config.gcp,
                "enhanced": config.enhanced,
                "reference_port": config.reference_port,
                "checkpoint_interval": config.checkpoint_interval,
                "progress_interval": config.progress_interval,
                "resume": config.resume,
            }
            process = context.Process(
                target=_worker_entry,
                args=(payload, sender),
                name="pqcp-worker-{:03d}".format(worker_index),
            )
            process.start()
            sender.close()
            processes[worker_index] = process
            receivers[worker_index] = receiver
    except BaseException:
        _interrupt_then_join(processes.values(), config.interrupt_grace_seconds)
        for receiver in receivers.values():
            receiver.close()
        raise

    raw_messages, parent_interrupted, live_new_solutions = _collect_worker_messages(
        config, processes, receivers, config.seconds,
        config.interrupt_grace_seconds, progress_callback,
    )
    elapsed = perf_counter() - started
    result = _finalize_worker_messages(
        config, raw_messages, parent_interrupted, elapsed,
        premerged_solution_count=live_new_solutions,
    )
    if result.failures and config.raise_on_worker_failure and not result.interrupted:
        raise ParallelWorkerError(result)
    return result


def _worker_entry(payload: Dict[str, object], connection: Connection) -> None:
    """Child entry point: run one existing trajectory below a private root."""
    # A terminal Ctrl+C should be handled once by the parent, which then sends
    # one SIGINT to each child.  Giving spawned children their own process
    # group avoids a terminal-group SIGINT followed by the parent's relay from
    # interrupting an atomic checkpoint write twice.
    if hasattr(os, "setpgrp"):
        try:
            os.setpgrp()
        except OSError:  # pragma: no cover - unusual restricted Unix runtime
            pass
    worker_index = int(payload["worker_index"])
    seed = int(payload["seed"])
    worker_root = Path(str(payload["worker_root"]))
    try:
        checkpoint_path = worker_root / "checkpoints" / (
            "L{}_enhanced.json".format(payload["L"])
            if bool(payload["enhanced"])
            else "L{}_reference.json".format(payload["L"])
            if bool(payload.get("reference_port", False))
            else "L{}.json".format(payload["L"])
        )
        if bool(payload.get("resume", False)) and not checkpoint_path.exists():
            raise FileNotFoundError(
                "parallel resume checkpoint does not exist: {}".format(checkpoint_path)
            )
        result = run_pipeline(PipelineConfig(
            L=int(payload["L"]),
            seed=seed,
            seconds=float(payload["seconds"]),
            enhanced=bool(payload["enhanced"]),
            gcp=bool(payload["gcp"]),
            reference_port=bool(payload.get("reference_port", False)),
            target_profile_offset=worker_index,
            # The conservative first parallel version is SA-only.
            repair=False,
            z3=False,
            checkpoint_interval=float(payload["checkpoint_interval"]),
            progress_interval=float(payload["progress_interval"]),
            root=worker_root,
            resume=checkpoint_path if bool(payload.get("resume", False)) else None,
        ))
        verified_candidates = []
        for path in result.verified_paths:
            with Path(path).open("r", encoding="utf-8") as handle:
                record = json.load(handle)
            verified_candidates.append({
                "A": record["A"],
                "B": record["B"],
                "profile": record.get("profile"),
                "source": str(path),
            })
        connection.send({
            "ok": True,
            "worker_index": worker_index,
            "seed": seed,
            "worker_root": str(worker_root),
            "interrupted": bool(result.interrupted),
            "elapsed": float(result.elapsed),
            "iteration": int(result.state.iteration),
            "restart_index": int(result.state.restart_index),
            "checkpoint_path": str(result.checkpoint_path),
            "best": {
                "A": list(result.state.best_a),
                "B": list(result.state.best_b),
                "score": int(result.state.best_score),
                "profile": list(result.verification.profile),
                "source": str(result.best_path),
            },
            "verified_candidates": verified_candidates,
        })
    except BaseException as error:  # pragma: no cover - child failure plumbing
        try:
            connection.send({
                "ok": False,
                "worker_index": worker_index,
                "seed": seed,
                "worker_root": str(worker_root),
                "interrupted": isinstance(error, KeyboardInterrupt),
                "error": "{}: {}".format(type(error).__name__, error),
                "traceback": traceback.format_exc(),
            })
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()


def _collect_worker_messages(
    config: ParallelPipelineConfig,
    processes: Dict[int, mp.Process],
    receivers: Dict[int, Connection],
    seconds: float,
    interrupt_grace_seconds: float,
    progress_callback: Optional[Callable[[Dict[str, object]], None]] = None,
) -> Tuple[List[Dict[str, object]], bool, int]:
    """Collect workers while immediately merging their atomic verified files."""
    by_connection = {connection: index for index, connection in receivers.items()}
    pending = dict(receivers)
    messages: List[Dict[str, object]] = []
    interrupted = False
    processed_verified_paths = set()
    live_new_solutions = 0
    last_progress = perf_counter()
    shutdown_deadline: Optional[float] = None
    # Initialization and atomic final writes are outside the child run budget.
    normal_deadline = perf_counter() + seconds + max(30.0, interrupt_grace_seconds)
    while pending:
        live_new_solutions += _merge_new_verified_files(
            config, processed_verified_paths
        )
        try:
            ready = wait(tuple(pending.values()), timeout=0.1)
        except KeyboardInterrupt:
            if not interrupted:
                interrupted = True
                _signal_interrupt(processes.values())
                shutdown_deadline = perf_counter() + interrupt_grace_seconds
            continue
        for connection in ready:
            worker_index = by_connection[connection]
            try:
                message = connection.recv()
            except EOFError:
                process = processes[worker_index]
                message = _failed_message(
                    worker_index,
                    process,
                    "worker exited without returning a result",
                )
            messages.append(message)
            connection.close()
            pending.pop(worker_index, None)

        now = perf_counter()
        if (
            progress_callback is not None
            and now - last_progress >= config.progress_interval
        ):
            progress_callback(_parallel_progress_snapshot(config, now))
            last_progress = now
        deadline = shutdown_deadline if interrupted else normal_deadline
        if deadline is not None and now >= deadline:
            if not interrupted:
                _signal_interrupt(
                    processes[index] for index in pending
                )
                shutdown_deadline = now + interrupt_grace_seconds
                interrupted = True
                continue
            for worker_index in tuple(pending):
                process = processes[worker_index]
                if process.is_alive():
                    process.terminate()
                messages.append(_failed_message(
                    worker_index, process,
                    "worker did not exit during the checkpoint grace period",
                ))
                pending[worker_index].close()
                pending.pop(worker_index, None)

    for process in processes.values():
        process.join(timeout=interrupt_grace_seconds)
        if process.is_alive():  # pragma: no cover - last-resort broken child
            process.terminate()
            process.join()
    live_new_solutions += _merge_new_verified_files(
        config, processed_verified_paths
    )
    messages.sort(key=lambda item: int(item.get("worker_index", -1)))
    return messages, interrupted, live_new_solutions


def _parallel_progress_snapshot(
    config: ParallelPipelineConfig,
    now: Optional[float] = None,
) -> Dict[str, object]:
    """Read only atomic worker files and return an aggregate progress view."""
    worker_rows = []
    for worker_index in range(config.workers):
        seed = derived_worker_seed(config.base_seed, worker_index)
        root = isolated_worker_root(config, worker_index)
        method = (
            "reference" if config.reference_port
            else "enhanced" if config.enhanced
            else "baseline"
        )
        best_path = root / "results" / "best" / "L{}_{}_seed{}.json".format(
            config.L, method, seed
        )
        checkpoint_path = root / "checkpoints" / (
            "L{}_enhanced.json".format(config.L)
            if config.enhanced else "L{}.json".format(config.L)
            if not config.reference_port
            else "L{}_reference.json".format(config.L)
        )
        row = {
            "worker": worker_index,
            "seed": seed,
            "iteration": 0,
            "restart": 0,
            "current_score": None,
            "best_score": None,
            "elapsed": 0.0,
            "proposal_samples": 1,
        }
        try:
            with best_path.open("r", encoding="utf-8") as handle:
                best = json.load(handle)
            row["best_score"] = int(best["score"])
            row["iteration"] = int(best.get("iteration", 0))
            row["restart"] = int(best.get("restart_index", 0))
            row["elapsed"] = float(best.get("elapsed", 0.0))
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            pass
        try:
            with checkpoint_path.open("r", encoding="utf-8") as handle:
                checkpoint = json.load(handle)["state"]
            row["iteration"] = int(checkpoint["iteration"])
            row["restart"] = int(checkpoint["restart_index"])
            row["current_score"] = int(checkpoint["current_score"])
            row["best_score"] = int(checkpoint["best_score"])
            row["elapsed"] = float(checkpoint["elapsed_seconds"])
            row["proposal_samples"] = int(
                checkpoint.get("algorithm_parameters", {}).get(
                    "proposal_samples", 1
                )
            )
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            pass
        worker_rows.append(row)
    scores = [row["best_score"] for row in worker_rows if row["best_score"] is not None]
    return {
        "L": config.L,
        "workers": config.workers,
        "reporting_workers": len(scores),
        "elapsed": max((float(row["elapsed"]) for row in worker_rows), default=0.0),
        "iterations": sum(int(row["iteration"]) for row in worker_rows),
        # SearchState numbers the initial trajectory as restart_index == 0;
        # the reference project's display counts that trajectory as restart 1.
        "restarts": sum(
            int(row["restart"]) + 1
            for row in worker_rows if row["best_score"] is not None
        ),
        "swap_evaluations": sum(
            int(row["iteration"]) * int(row["proposal_samples"])
            for row in worker_rows
        ),
        "best_score": min(scores) if scores else None,
        "worker_rows": worker_rows,
    }


def _merge_new_verified_files(
    config: ParallelPipelineConfig,
    processed_paths: set,
) -> int:
    """Parent-review newly persisted worker solutions and update official L.txt.

    Workers write these JSON snapshots atomically below private roots before
    continuing their search.  Polling them keeps the user's established
    "verify, append if new, then continue" behavior even for an infinite
    parallel run; no child ever writes the official project file.
    """
    appended = 0
    for worker_index in range(config.workers):
        directory = isolated_worker_root(
            config, worker_index
        ) / "results" / "verified"
        for path in sorted(directory.glob("*.json")):
            key = str(path.resolve())
            if key in processed_paths:
                continue
            try:
                with path.open("r", encoding="utf-8") as handle:
                    raw = json.load(handle)
                candidate = _review_candidate(raw, str(path))
            except (OSError, json.JSONDecodeError):
                # Atomic rename normally makes partial JSON impossible.  A
                # transient filesystem visibility issue is retried next poll.
                continue
            if not candidate.verified:
                raise RuntimeError(
                    "worker verified snapshot failed parent independent verifier"
                )
            processed_paths.add(key)
            if append_verified_solution_if_new(
                config.L, candidate.a, candidate.b, Path(config.root)
            ):
                appended += 1
    return appended


def _signal_interrupt(processes: Iterable[mp.Process]) -> None:
    """Ask live macOS/Unix spawn children to take their atomic save path."""
    for process in processes:
        if process.pid is not None and process.is_alive():
            try:
                os.kill(process.pid, signal.SIGINT)
            except ProcessLookupError:
                pass


def _interrupt_then_join(processes: Iterable[mp.Process], grace: float) -> None:
    """Conservatively stop partially started children during parent failure."""
    values = tuple(processes)
    _signal_interrupt(values)
    for process in values:
        process.join(timeout=grace)
        if process.is_alive():
            process.terminate()
            process.join()


def _failed_message(worker_index: int, process: mp.Process, reason: str) -> Dict[str, object]:
    return {
        "ok": False,
        "worker_index": worker_index,
        "seed": 0,
        "worker_root": "",
        "interrupted": False,
        "error": "{} (exitcode={})".format(reason, process.exitcode),
    }


def _review_candidate(raw: Dict[str, object], source_default: str) -> ReviewedCandidate:
    """Recompute, score, and independently verify one child candidate."""
    a = tuple(raw["A"])
    b = tuple(raw["B"])
    profile = tuple(full_correlation_profile(a, b))
    score = pqcp_objective(profile)
    verification = verify_pqcp(a, b)
    if verification.profile != profile:
        raise RuntimeError("parent verifier profile disagrees with full recomputation")
    child_profile = raw.get("profile")
    if child_profile is not None and tuple(child_profile) != profile:
        raise RuntimeError("child profile disagrees with parent recomputation")
    child_score = raw.get("score")
    if child_score is not None and int(child_score) != score:
        raise RuntimeError("child score disagrees with parent recomputation")
    return ReviewedCandidate(
        a, b, profile, score, verification.is_valid,
        str(raw.get("source", source_default)),
    )


def _review_worker_message(
    config: ParallelPipelineConfig, raw: Dict[str, object]
) -> WorkerReport:
    """Validate metadata and independently review all candidates from one child."""
    worker_index = int(raw.get("worker_index", -1))
    expected_seed = derived_worker_seed(config.base_seed, worker_index)
    expected_root = isolated_worker_root(config, worker_index)
    seed = int(raw.get("seed", expected_seed))
    worker_root = Path(str(raw.get("worker_root", expected_root)))
    if not raw.get("ok", False):
        return WorkerReport(
            worker_index, seed, worker_root, False,
            bool(raw.get("interrupted", False)), 0.0, 0, 0, None, None, (),
            str(raw.get("error", "unknown worker failure")),
        )
    errors = []
    if seed != expected_seed:
        errors.append("worker seed does not match deterministic schedule")
    if worker_root != expected_root:
        errors.append("worker root is not the assigned isolated root")
    best = None
    verified: List[ReviewedCandidate] = []
    try:
        best = _review_candidate(raw["best"], "worker best")
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        errors.append("invalid best candidate: {}".format(error))
    for index, candidate in enumerate(raw.get("verified_candidates", ())):
        try:
            reviewed = _review_candidate(candidate, "verified candidate {}".format(index))
            if not reviewed.verified:
                raise RuntimeError("child claimed a verifier-failing candidate")
            verified.append(reviewed)
        except (KeyError, TypeError, ValueError, RuntimeError) as error:
            errors.append("invalid verified candidate {}: {}".format(index, error))
    checkpoint_value = raw.get("checkpoint_path")
    return WorkerReport(
        worker_index=worker_index,
        seed=seed,
        worker_root=worker_root,
        ok=not errors,
        interrupted=bool(raw.get("interrupted", False)),
        elapsed=float(raw.get("elapsed", 0.0)),
        iteration=int(raw.get("iteration", 0)),
        restart_index=int(raw.get("restart_index", 0)),
        checkpoint_path=Path(str(checkpoint_value)) if checkpoint_value else None,
        best=best,
        verified_candidates=tuple(verified),
        error="; ".join(errors) if errors else None,
    )


def _canonical_pair(candidate: ReviewedCandidate) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Canonicalize only A/B exchange, matching official project deduplication."""
    return min((candidate.a, candidate.b), (candidate.b, candidate.a))


def _finalize_worker_messages(
    config: ParallelPipelineConfig,
    raw_messages: Sequence[Dict[str, object]],
    interrupted: bool,
    elapsed: float,
    premerged_solution_count: int = 0,
) -> ParallelPipelineResult:
    """Parent-review messages, merge official solutions, and save one summary."""
    reports = tuple(
        sorted(
            (_review_worker_message(config, raw) for raw in raw_messages),
            key=lambda report: report.worker_index,
        )
    )
    best_entries = [
        (report.best.score, report.worker_index, report.best)
        for report in reports if report.best is not None
    ]
    if best_entries:
        _, global_best_worker, global_best = min(best_entries, key=lambda item: (item[0], item[1]))
    else:
        global_best_worker, global_best = None, None

    # Review has already called the independent verifier for every best and
    # every claimed verified candidate.  Only parent-verified unique pairs now
    # reach the sole official L.txt writer.
    verified_by_pair: Dict[
        Tuple[Tuple[int, ...], Tuple[int, ...]], ReviewedCandidate
    ] = {}
    for report in reports:
        candidates = list(report.verified_candidates)
        if report.best is not None and report.best.verified:
            candidates.append(report.best)
        for candidate in candidates:
            verified_by_pair.setdefault(_canonical_pair(candidate), candidate)
    new_solution_count = premerged_solution_count
    for candidate in verified_by_pair.values():
        if append_verified_solution_if_new(
            config.L, candidate.a, candidate.b, Path(config.root)
        ):
            new_solution_count += 1

    summary_path = parallel_run_root(config) / "parent_summary.json"
    result = ParallelPipelineResult(
        L=config.L,
        base_seed=config.base_seed,
        workers=config.workers,
        seconds=config.seconds,
        elapsed=elapsed,
        interrupted=interrupted or any(report.interrupted for report in reports),
        reports=reports,
        global_best=global_best,
        global_best_worker=global_best_worker,
        new_solution_count=new_solution_count,
        summary_path=summary_path,
    )
    atomic_write_json(summary_path, _result_payload(result))
    return result


def _candidate_payload(candidate: Optional[ReviewedCandidate]) -> Optional[Dict[str, object]]:
    if candidate is None:
        return None
    return {
        "A": list(candidate.a), "B": list(candidate.b),
        "profile": list(candidate.profile), "score": candidate.score,
        "verified": candidate.verified, "source": candidate.source,
    }


def _result_payload(result: ParallelPipelineResult) -> Dict[str, object]:
    """Return the atomically persisted, JSON-safe parent summary."""
    return {
        "L": result.L,
        "base_seed": result.base_seed,
        "workers": result.workers,
        # JSON has no standard infinity literal; ``None`` plus the explicit
        # label keeps an until-Ctrl+C run portable across strict readers.
        "seconds": None if math.isinf(result.seconds) else result.seconds,
        "budget": "until_interrupt" if math.isinf(result.seconds) else "finite",
        "elapsed": result.elapsed,
        "interrupted": result.interrupted,
        "start_method": result.start_method,
        "global_best_worker": result.global_best_worker,
        "global_best": _candidate_payload(result.global_best),
        "new_solution_count": result.new_solution_count,
        "workers_results": [
            {
                "worker_index": report.worker_index,
                "seed": report.seed,
                "worker_root": str(report.worker_root),
                "ok": report.ok,
                "interrupted": report.interrupted,
                "elapsed": report.elapsed,
                "iteration": report.iteration,
                "restart_index": report.restart_index,
                "checkpoint_path": str(report.checkpoint_path) if report.checkpoint_path else None,
                "best": _candidate_payload(report.best),
                "verified_candidates": [
                    _candidate_payload(candidate)
                    for candidate in report.verified_candidates
                ],
                "error": report.error,
            }
            for report in result.reports
        ],
    }


__all__ = (
    "ParallelPipelineConfig", "ParallelPipelineResult", "ParallelWorkerError",
    "ReviewedCandidate", "SEED_STRIDE", "WorkerReport",
    "derived_worker_seed", "isolated_worker_root", "parallel_run_root",
    "run_parallel_pipeline",
)
