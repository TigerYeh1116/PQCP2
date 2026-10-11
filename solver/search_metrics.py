"""Observation-only session counters; no RNG, solver policy or candidate writes.

Timers are host wall-clock durations, NOT GPU utilization. Device polish timing
is collected after its existing CPU export synchronization. Concurrent stage
times overlap and must not be added to estimate total elapsed time. GF(2)
histograms count completed turns (including retries), not unique jobs. Unique
matched jobs are session-local canonical keys returned by the existing pool;
they exclude previously submitted/checkpoint-only work.
"""

from collections import Counter
from pathlib import Path
import threading
from time import perf_counter
from typing import Any, Dict
from uuid import uuid4
import warnings
import sqlite3

from .checkpoint import atomic_write_json
from .disk_index import DiskIndex


class SearchMetrics:
    """Thread-safe, rate-limited atomic snapshots; I/O failure disables logging."""

    def __init__(self, root: Path, length: int, seed: int, *, enabled: bool = True,
                 interval: float = 60.0, clock=perf_counter, parameters=None) -> None:
        if interval <= 0:
            raise ValueError("metrics interval must be positive")
        self.enabled, self.interval, self.clock = enabled, interval, clock
        self.started = self.last_save = clock()
        self.path = Path(root) / "results/search_metrics" / (
            "L{}_seed{}_{}.json".format(length, seed, uuid4().hex[:12]))
        self.length, self.seed = length, seed
        self.parameters = dict(parameters or {})
        self._lock, self._write_lock = threading.RLock(), threading.Lock()
        self._matched = DiskIndex()
        self._counts = Counter()
        self._seconds = Counter()
        self._statuses = Counter()
        self._ranks, self._free = Counter(), Counter()
        self._backlog = self._peak_backlog = 0

    def stage(self, name: str, seconds: float, **counts: int) -> None:
        """Record a completed interval, with explicitly named work counters."""
        if not self.enabled:
            return
        with self._lock:
            self._seconds[name] += max(0.0, seconds)
            for key, value in counts.items():
                self._counts[name + "." + key] += value
        self.flush()

    def matched(self, key: Any) -> None:
        """Canonical compressed match observed, once per session regardless of retry."""
        if self.enabled:
            with self._lock:
                try:
                    self._matched.add(key)
                except (OSError, sqlite3.Error) as error:
                    self.enabled = False
                    warnings.warn('Search metrics disabled after index failure: {}'.format(error))

    def backlog(self, queued: int, restored: int = 0) -> None:
        """Queue depth excludes the active turn; pending restored jobs are included."""
        if self.enabled:
            with self._lock:
                self._backlog = queued + restored
                self._peak_backlog = max(self._peak_backlog, self._backlog)

    def result(self, result: Any) -> None:
        """Report available exact backend counters; missing timings stay absent."""
        if not self.enabled:
            return
        if not isinstance(result, dict):
            result = {"status_counts": getattr(result, "status_counts", {})}
        with self._lock:
            self._counts["completion_turns"] += 1
            self._statuses[result.get("status", "unreported")] += 1
            self._counts["z3_calls"] += int(result.get(
                "z3_calls", sum(result.get("status_counts", {}).values())
                or bool(result.get("z3_executed", False))))
            self._counts["orientation_search_evaluations"] += int(result.get("orientation_search_evaluations", 0))
            self._counts["orientation_search_feasible"] += int(result.get("orientation_search_feasible", 0))
            for name in ("construction_time", "solve_time", "gf2_preprocessing_time", "orientation_search_time"):
                value = result.get(name)
                if isinstance(value, (int, float)) and value >= 0:
                    self._seconds[name] += value
                    self._counts[name + ".reported_turns"] += 1
            if result.get("gf2_applied"):
                self._ranks[str(result["gf2_rank"])] += 1
                self._free[str(result["gf2_free_bits"])] += 1
        self.flush()

    def snapshot(self) -> Dict[str, Any]:
        """Return a consistent serializable copy, never a search checkpoint."""
        with self._lock:
            elapsed = max(0.0, self.clock() - self.started)
            return {
                "schema_version": 1, "kind": "observation_only", "L": self.length,
                "seed": self.seed, "elapsed": elapsed, "counts": dict(self._counts),
                "parameters": dict(self.parameters),
                "seconds": dict(self._seconds), "statuses": dict(self._statuses),
                "unique_compressed_matches": len(self._matched),
                "unique_matches_per_minute": 60 * len(self._matched) / elapsed if elapsed else 0.0,
                "gf2_rank_turn_histogram": dict(self._ranks),
                "gf2_free_bits_turn_histogram": dict(self._free),
                "queued_backlog": self._backlog, "peak_queued_backlog": self._peak_backlog,
            }

    def flush(self, *, force: bool = False) -> None:
        """Write at coarse events, at most once per interval, or on shutdown."""
        if not self.enabled:
            return
        with self._write_lock:
            now = self.clock()
            if not force and now - self.last_save < self.interval:
                return
            try:
                atomic_write_json(self.path, self.snapshot())
            except OSError as error:
                self.enabled = False
                warnings.warn("Search metrics disabled after write failure: {}".format(error))
            self.last_save = now

    def close(self):
        """Release temporary exact-match metadata after the producer stops."""
        self.flush(force=True)
        self._matched.close()
