"""Compiled high-throughput backend for the Project 2 reference search.

The C subprocess owns only candidate generation and local-search navigation.
It never writes the official ``L.txt``.  Every machine-readable candidate is
recomputed by :mod:`solver.correlation`, checked by the independent verifier,
deduplicated (including A/B exchange), and only then appended by Python.

Compressed FKM seeds are generated once per ``(L, seed)`` and stored as a
reusable text bank.  The C process indexes that bank by ordinary and
alternating sums, so every accepted A initializer has the exact content
required by the selected target profile.  B uses the reference project's
exact-content spectral initializer.
"""

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import selectors
import signal
import subprocess
import sys
from time import perf_counter
from typing import Callable, Dict, Optional, Tuple

from .checkpoint import atomic_write_json
from .compressed_fkm import compressed_fkm_lift_candidates
from .correlation import full_correlation_profile
from .objective import pqcp_objective, pqcp_objective_breakdown
from .search_runner import append_verified_solution_if_new
from .target_profiles import pair_content, target_content_profiles
from .verifier import verify_pqcp


_CANDIDATE_RE = re.compile(
    r"^PQCP_CANDIDATE L=(?P<L>\d+) k=(?P<k>\d+) sign=(?P<sign>-?\d+) "
    r"energy=(?P<energy>-?\d+) "
    r"a=(?P<a>[01]+) b=(?P<b>[01]+)$"
)
_ELITE_RE = re.compile(
    r"^PQCP_ELITE L=(?P<L>\d+) k=(?P<k>\d+) sign=(?P<sign>-?\d+) "
    r"energy=(?P<energy>-?\d+)(?: score=(?P<score>\d+))? "
    r"a=(?P<a>[01]+) b=(?P<b>[01]+)$"
)
_STATS_RE = re.compile(
    r"^\u641c\u5c0b\u7d71\u8a08\uff1arestart=(?P<restarts>\d+), moves=(?P<moves>\d+), "
    r"swap evaluations=(?P<swaps>\d+), valid results=(?P<valid>\d+)$"
)
_FKM_STATS_RE = re.compile(
    r"^FKM seed statistics: applied=(?P<applied>\d+), misses=(?P<misses>\d+)$"
)


@dataclass(frozen=True)
class CSearchConfig:
    """Runtime controls for one compiled reference-search invocation."""

    L: int
    seed: int = 123
    seconds: float = float("inf")
    threads: Optional[int] = None
    candidate_count: int = 2
    seeds_per_content: int = 64
    root: Path = Path(".")
    source: Optional[Path] = None
    executable: Optional[Path] = None
    echo: bool = True
    emit_elites: bool = False
    direct_swap_sampling: bool = True
    quarter_frequency_pruning: bool = True
    half_shift_seeding: bool = True
    pair_seed_path: Optional[Path] = None
    pair_seed_percent: int = 100
    pair_seed_swaps: Optional[int] = None

    def __post_init__(self) -> None:
        if not isinstance(self.L, int) or isinstance(self.L, bool) or self.L < 4 or self.L % 2:
            raise ValueError("C backend requires an even integer L >= 4")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError("seed must be an integer")
        if math.isnan(self.seconds) or self.seconds <= 0:
            raise ValueError("seconds must be positive")
        if self.threads is not None and self.threads <= 0:
            raise ValueError("threads must be positive")
        if not 1 <= self.candidate_count <= 16:
            raise ValueError("candidate_count must be in 1..16")
        if self.seeds_per_content <= 0:
            raise ValueError("seeds_per_content must be positive")
        if not isinstance(self.direct_swap_sampling, bool):
            raise ValueError("direct_swap_sampling must be boolean")
        if not isinstance(self.quarter_frequency_pruning, bool):
            raise ValueError("quarter_frequency_pruning must be boolean")
        if not isinstance(self.half_shift_seeding, bool):
            raise ValueError("half_shift_seeding must be boolean")
        if self.pair_seed_path is not None and not isinstance(self.pair_seed_path, Path):
            raise ValueError("pair_seed_path must be a pathlib.Path or None")
        if not isinstance(self.pair_seed_percent, int) or isinstance(self.pair_seed_percent, bool):
            raise ValueError("pair_seed_percent must be an integer")
        if not 0 <= self.pair_seed_percent <= 100:
            raise ValueError("pair_seed_percent must be in 0..100")
        if self.pair_seed_swaps is not None and (
                not isinstance(self.pair_seed_swaps, int) or isinstance(self.pair_seed_swaps, bool)
                or not 0 <= self.pair_seed_swaps <= 16):
            raise ValueError("pair_seed_swaps must be None or an integer in 0..16")


@dataclass(frozen=True)
class CSearchResult:
    """Counters and independently reviewed discoveries from the C backend.

    ``valid_results`` retains the raw C hit count for diagnostics (duplicates
    included). User-facing statistics report ``new_solutions`` instead.
    """

    L: int
    elapsed: float
    interrupted: bool
    restarts: int
    moves: int
    swap_evaluations: int
    valid_results: int
    verified_candidates: int
    new_solutions: int
    fkm_seeds_applied: int
    fkm_seed_misses: int
    seed_bank_path: Path
    executable_path: Path
    log_path: Path

    @property
    def moves_per_second(self) -> float:
        return self.moves / self.elapsed if self.elapsed > 0 else 0.0

    @property
    def swap_evaluations_per_second(self) -> float:
        return self.swap_evaluations / self.elapsed if self.elapsed > 0 else 0.0


def project_root() -> Path:
    """Return the repository root independently of the process cwd."""
    return Path(__file__).resolve().parent.parent


def ensure_c_backend(
    source: Optional[Path] = None,
    executable: Optional[Path] = None,
) -> Path:
    """Compile the C backend atomically when missing or older than its source."""
    source_path = Path(source) if source is not None else project_root() / "csrc" / "pqcp_search.c"
    executable_path = Path(executable) if executable is not None else project_root() / "build" / "pqcp_search_c"
    if not source_path.is_file():
        raise FileNotFoundError("C backend source does not exist: {}".format(source_path))
    # A checked-in binary can be newer than the source while still being a
    # macOS Mach-O file. Google Colab is Linux and must rebuild that artifact.
    expected_magic = b"\x7fELF" if sys.platform.startswith("linux") else None
    compatible_binary = executable_path.is_file()
    if compatible_binary and expected_magic is not None:
        try:
            with executable_path.open("rb") as handle:
                compatible_binary = handle.read(4) == expected_magic
        except OSError:
            compatible_binary = False
    if (
        compatible_binary
        and executable_path.stat().st_mtime_ns >= source_path.stat().st_mtime_ns
    ):
        return executable_path
    executable_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = executable_path.with_name(".{}.tmp".format(executable_path.name))
    command = (
        "cc", "-O3", "-std=c11", "-pthread", str(source_path), "-lm",
        "-o", str(temporary),
    )
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            "C backend compilation failed:\n{}".format(
                (completed.stderr or completed.stdout).strip()
            )
        )
    temporary.replace(executable_path)
    return executable_path


def ensure_compressed_fkm_seed_bank(
    length: int,
    seed: int,
    root: Path = Path("."),
    seeds_per_content: int = 64,
) -> Path:
    """Create/reuse a complete factor-two FKM bank for all A contents.

    The metadata sidecar records the exact generation parameters.  A bank is
    reused only when those parameters and its nonempty line count agree.
    """
    root_path = Path(root)
    directory = root_path / "results" / "fkm_seeds"
    path = directory / "L{}_compressed_seed{}.txt".format(length, seed)
    metadata_path = path.with_suffix(".json")
    profiles = target_content_profiles(length, decimation_reduced=True)
    if not profiles:
        raise ValueError(
            "L={} has no target-content profile satisfying the necessary identities".format(length)
        )
    contents = tuple(sorted({
        (profile.a_even_ones, profile.a_odd_ones) for profile in profiles
    }))
    expected = {
        "format": 1,
        "L": length,
        "seed": seed,
        "seeds_per_content": seeds_per_content,
        "contents": [list(content) for content in contents],
    }
    try:
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        lines = path.read_text(encoding="ascii").splitlines()
        metadata_parameters = dict(existing)
        line_count = metadata_parameters.pop("line_count", None)
        if metadata_parameters == expected and line_count == len(lines) and lines and all(
            len(line) == length and set(line) <= {"0", "1"} for line in lines
        ):
            return path
    except (OSError, json.JSONDecodeError):
        pass

    words = []
    seen = set()
    for content_index, (even_ones, odd_ones) in enumerate(contents):
        candidates = compressed_fkm_lift_candidates(
            length,
            even_ones,
            odd_ones,
            seed=seed + (content_index + 1) * 1_000_003,
            limit=seeds_per_content,
            max_necklaces=max(256, 16 * seeds_per_content),
        )
        if not candidates:
            raise RuntimeError(
                "compressed FKM produced no seed for content ({},{})".format(
                    even_ones, odd_ones
                )
            )
        for candidate in candidates:
            # B is a dummy here; pair_content provides one established exact
            # implementation of the parity count convention.
            dummy = (0,) * length
            observed = pair_content(candidate, dummy)[:2]
            if observed != (even_ones, odd_ones):
                raise RuntimeError("compressed FKM seed has incorrect content")
            text = "".join(map(str, candidate))
            if text not in seen:
                seen.add(text)
                words.append(text)
    if not words:
        raise RuntimeError("compressed FKM seed bank would be empty")
    directory.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".{}.tmp".format(path.name))
    temporary.write_text("\n".join(words) + "\n", encoding="ascii")
    temporary.replace(path)
    atomic_write_json(metadata_path, {**expected, "line_count": len(words)})
    return path


def run_c_search(
    config: CSearchConfig,
    line_callback: Optional[Callable[[str], None]] = None,
    elite_callback: Optional[
        Callable[[Tuple[int, ...], Tuple[int, ...], Dict[str, int]], None]
    ] = None,
    stop_event=None,
) -> CSearchResult:
    """Run compiled search and independently process every exact candidate.

    ``stop_event`` is an optional thread-safe cancellation signal used by the
    concurrent MPS/C orchestrator.  It only requests the same graceful SIGINT
    path as Ctrl-C; verification and final C statistics are still drained.
    """
    root = Path(config.root)
    executable = ensure_c_backend(config.source, config.executable)
    seed_bank = ensure_compressed_fkm_seed_bank(
        config.L, config.seed, root, config.seeds_per_content
    )
    log_path = root / "logs" / "c_search_L{}_seed{}.log".format(config.L, config.seed)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PQCP_SEED"] = str(config.seed)
    environment["PQCP_CANDIDATES"] = str(config.candidate_count)
    environment["PQCP_FKM_SEEDS"] = str(seed_bank.resolve())
    if config.threads is not None:
        environment["PQCP_THREADS"] = str(config.threads)
    else:
        environment.pop("PQCP_THREADS", None)
    if config.pair_seed_path is not None:
        pair_seed_path = Path(config.pair_seed_path)
        if not pair_seed_path.is_file():
            raise FileNotFoundError("pair seed bank does not exist: {}".format(pair_seed_path))
        environment["PQCP_PAIR_SEEDS"] = str(pair_seed_path.resolve())
        environment["PQCP_PAIR_SEED_PERCENT"] = str(config.pair_seed_percent)
    else:
        environment.pop("PQCP_PAIR_SEEDS", None)
        environment.pop("PQCP_PAIR_SEED_PERCENT", None)
    if config.pair_seed_path is not None and config.pair_seed_swaps is not None:
        environment["PQCP_PAIR_SEED_SWAPS"] = str(config.pair_seed_swaps)
    else:
        environment.pop("PQCP_PAIR_SEED_SWAPS", None)
    environment.pop("PQCP_VERBOSE_CANDIDATES", None)
    if config.direct_swap_sampling:
        environment["PQCP_MOVE_POOL"] = "1"
    else:
        environment.pop("PQCP_MOVE_POOL", None)
    if config.quarter_frequency_pruning:
        environment["PQCP_QUARTER_PRUNING"] = "1"
    else:
        environment.pop("PQCP_QUARTER_PRUNING", None)
    if config.half_shift_seeding:
        environment["PQCP_HALF_SHIFT_SEED"] = "1"
    else:
        environment.pop("PQCP_HALF_SHIFT_SEED", None)
    if config.emit_elites or elite_callback is not None:
        environment["PQCP_EMIT_ELITES"] = "1"
    else:
        environment.pop("PQCP_EMIT_ELITES", None)

    process = subprocess.Popen(
        (str(executable), str(config.L)),
        cwd=root,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    if process.stdout is None:  # pragma: no cover - guaranteed by PIPE
        raise RuntimeError("failed to capture C backend output")
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    started = perf_counter()
    search_started_at: Optional[float] = None
    interrupted = False
    stop_sent = False
    stats = {"restarts": 0, "moves": 0, "swaps": 0, "valid": 0}
    fkm_stats = {"applied": 0, "misses": 0}
    verified_candidates = 0
    new_solutions = 0

    def display(line: str) -> None:
        if config.echo:
            print(line, flush=True)
        if line_callback is not None:
            line_callback(line)

    def new_solution_statistics() -> str:
        return (
            "搜尋統計：restart={}, moves={}, swap evaluations={}, valid results={}"
            .format(stats["restarts"], stats["moves"], stats["swaps"], new_solutions)
        )

    def emit(line: str, log_handle) -> None:
        nonlocal verified_candidates, new_solutions, stats, fkm_stats, search_started_at
        log_handle.write(line + "\n")
        log_handle.flush()
        if search_started_at is None and line.startswith("尚未找到不代表"):
            search_started_at = perf_counter()
        candidate_match = _CANDIDATE_RE.match(line)
        if candidate_match:
            values = candidate_match.groupdict()
            if int(values["L"]) != config.L:
                raise RuntimeError("C candidate length metadata mismatch")
            a = tuple(int(bit) for bit in values["a"])
            b = tuple(int(bit) for bit in values["b"])
            if len(a) != config.L or len(b) != config.L:
                raise RuntimeError("C candidate sequence length mismatch")
            profile = tuple(full_correlation_profile(a, b))
            verification = verify_pqcp(a, b)
            if verification.profile != profile or not verification.is_valid:
                raise RuntimeError("C candidate failed independent Python verifier")
            score = pqcp_objective(profile)
            if score != 0:
                raise RuntimeError("verified C candidate has nonzero objective")
            verified_candidates += 1
            is_new = append_verified_solution_if_new(config.L, a, b, root)
            if is_new:
                new_solutions += 1
            result_path = root / "results" / "verified" / (
                "L{}_c_seed{}_candidate{:06d}.json".format(
                    config.L, config.seed, verified_candidates
                )
            )
            atomic_write_json(result_path, {
                "L": config.L,
                "method": "c_compressed_fkm",
                "seed": config.seed,
                "A": list(a),
                "B": list(b),
                "profile": list(profile),
                "score": score,
                "objective_components": pqcp_objective_breakdown(profile),
                "verified": True,
                "new_solution_written": is_new,
            })
            message = (
                "\u627e\u5230\u4e00\u7d44\u65b0\u7684\u6709\u6548\u5019\u9078\uff08\u5df2\u8ffd\u52a0\u5beb\u5165 {}.txt\uff09".format(config.L)
                if is_new else
                "\u627e\u5230\u6709\u6548\u5019\u9078\uff0c\u4f46\u8207\u65e2\u6709\u5e8f\u5217\u5c0d\u91cd\u8907\uff08\u672a\u65b0\u589e\u5beb\u5165\uff09"
            )
            display(message)
            if is_new:
                # Do not wait for the next periodic C snapshot to acknowledge
                # a successfully persisted result. Other counters are the
                # latest snapshot; the new-result count is current.
                display(new_solution_statistics())
            return
        elite_match = _ELITE_RE.match(line)
        if elite_match:
            values = elite_match.groupdict()
            a = tuple(int(bit) for bit in values["a"])
            b = tuple(int(bit) for bit in values["b"])
            if int(values["L"]) != config.L or len(a) != config.L or len(b) != config.L:
                raise RuntimeError("C elite metadata mismatch")
            if elite_callback is not None:
                metadata = {
                    "L": int(values["L"]),
                    "k": int(values["k"]),
                    "sign": int(values["sign"]),
                    "energy": int(values["energy"]),
                }
                if values.get("score") is not None:
                    metadata["score"] = int(values["score"])
                elite_callback(a, b, metadata)
            return
        stats_match = _STATS_RE.match(line)
        if stats_match:
            stats = {key: int(value) for key, value in stats_match.groupdict().items()}
            # Raw C counters remain in the log and returned diagnostics only.
            # They include duplicates and can arrive after a newer candidate.
            line = new_solution_statistics()
        fkm_match = _FKM_STATS_RE.match(line)
        if fkm_match:
            fkm_stats = {
                key: int(value) for key, value in fkm_match.groupdict().items()
            }
        display(line)

    caught_error: Optional[BaseException] = None
    with log_path.open("w", encoding="utf-8") as log_handle:
        try:
            while True:
                now = perf_counter()
                if (stop_event is not None and stop_event.is_set() and not stop_sent
                        and process.poll() is None):
                    interrupted = True
                    process.send_signal(signal.SIGINT)
                    stop_sent = True
                if (
                    not stop_sent
                    and search_started_at is not None
                    and now - search_started_at >= config.seconds
                ):
                    process.send_signal(signal.SIGINT)
                    stop_sent = True
                events = selector.select(timeout=0.1)
                for key, _mask in events:
                    line = key.fileobj.readline()
                    if line:
                        emit(line.rstrip("\r\n"), log_handle)
                if process.poll() is not None:
                    for line in process.stdout:
                        emit(line.rstrip("\r\n"), log_handle)
                    break
        except KeyboardInterrupt:
            interrupted = True
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                stop_sent = True
        except BaseException as error:
            caught_error = error
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                stop_sent = True
        finally:
            if process.poll() is None:
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    process.wait(timeout=5)
            for line in process.stdout:
                emit(line.rstrip("\r\n"), log_handle)
            selector.close()
    elapsed = perf_counter() - started
    if caught_error is not None:
        raise caught_error
    if process.returncode != 0 and not (
        interrupted and process.returncode == -signal.SIGINT
    ):
        raise RuntimeError("C backend exited with status {}".format(process.returncode))
    return CSearchResult(
        L=config.L,
        elapsed=elapsed,
        interrupted=interrupted,
        restarts=stats["restarts"],
        moves=stats["moves"],
        swap_evaluations=stats["swaps"],
        valid_results=stats["valid"],
        verified_candidates=verified_candidates,
        new_solutions=new_solutions,
        fkm_seeds_applied=fkm_stats["applied"],
        fkm_seed_misses=fkm_stats["misses"],
        seed_bank_path=seed_bank,
        executable_path=executable,
        log_path=log_path,
    )


__all__ = (
    "CSearchConfig", "CSearchResult", "ensure_c_backend",
    "ensure_compressed_fkm_seed_bank", "run_c_search",
)
