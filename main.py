"""Run FKM -> SA -> guidance/beam repair -> optional bounded-Z3 fallback."""

import argparse
from pathlib import Path
import queue
import sys
import threading

from solver.pipeline import (
    DEFAULT_REPAIR_BEAM_WIDTH,
    DEFAULT_REPAIR_MAX_DEPTH,
    DEFAULT_Z3_COMPLETION_RADIUS,
    DEFAULT_Z3_TIMEOUT_SECONDS,
    DEFAULT_Z3_TRIGGER_SCORE,
    PROJECT_LENGTHS,
    PipelineConfig,
    _run_optional_z3,
    run_pipeline,
    validate_length,
)
from solver.parallel_pipeline import (
    ParallelPipelineConfig,
    ParallelWorkerError,
    run_parallel_pipeline,
)
from solver.search_runner import DEFAULT_GUIDED_PROPOSAL_SAMPLES
from solver.c_backend import CSearchConfig, run_c_search
from solver.correlation import full_correlation_profile
from solver.hybrid import SAElite
from solver.objective import pqcp_objective
from solver.target_profiles import pair_content, target_content_profiles
from solver.z3_guidance import correlation_hamming_lower_bound


class _CCompletionBridge:
    """Feed rare low-score C elites to existing completion without blocking C."""

    def __init__(self, args, length: int, root: Path, result_callback=None) -> None:
        self.args = args
        self.length = length
        self.root = root
        self.result_callback = result_callback
        self.enabled = bool(args.repair or args.z3)
        self._last_score = None
        self._queue = queue.Queue()
        self._closing = threading.Event()
        self._results = []
        self._error = None
        self._thread = None
        if self.enabled:
            self._thread = threading.Thread(
                target=self._worker,
                name="pqcp-c-z3-completion",
                daemon=False,
            )
            self._thread.start()

    @property
    def trigger_count(self) -> int:
        """Return completed completion-pipeline attempts."""
        return len(self._results)

    @property
    def z3_execution_count(self) -> int:
        """Return actual Z3 solver calls, excluding guidance/beam-only work."""
        count = 0
        for result in self._results:
            if isinstance(result, dict):
                count += int(bool(result.get("z3_executed", False)))
            else:
                status_counts = getattr(result, "status_counts", None)
                if isinstance(status_counts, dict):
                    count += sum(int(value) for value in status_counts.values())
                else:
                    # The broad portfolio object represents one completed Z3
                    # invocation even if an older implementation lacks its
                    # per-task status breakdown.
                    count += 1
        return count

    @property
    def valid_result_count(self) -> int:
        """Return independently verified solutions found by completion."""
        return sum(
            int(bool(
                result.get("solved", False)
                if isinstance(result, dict)
                else getattr(result, "solved", False)
            ))
            for result in self._results
        )

    def consider(self, a, b, metadata) -> None:
        """Queue only strictly improving elites satisfying existing guards."""
        if not self.enabled:
            return
        reported_score = metadata.get("score")
        if reported_score is not None:
            if reported_score > self.args.z3_trigger_score:
                return
            if self._last_score is not None and reported_score >= self._last_score:
                return
        profile = tuple(full_correlation_profile(a, b))
        score = pqcp_objective(profile)
        if reported_score is not None and score != reported_score:
            raise RuntimeError("C elite score failed independent Python consistency check")
        if score > self.args.z3_trigger_score:
            return
        if correlation_hamming_lower_bound(profile) > self.args.z3_profile_radius:
            return
        if self._last_score is not None and score >= self._last_score:
            return
        self._last_score = score
        content = pair_content(a, b)
        encoded = tuple(
            (
                candidate.k, candidate.eta,
                candidate.a_even_ones, candidate.a_odd_ones,
                candidate.b_even_ones, candidate.b_odd_ones,
            )
            for candidate in target_content_profiles(
                self.length, decimation_reduced=True
            )
            if candidate.k == metadata["k"]
            and candidate.eta == metadata["sign"]
            and (
                candidate.a_even_ones, candidate.a_odd_ones,
                candidate.b_even_ones, candidate.b_odd_ones,
            ) == content
        )
        self._queue.put((tuple(a), tuple(b), score, encoded))
        print(
            "[pipeline] C elite best_score={} 已排入 completion queue；"
            "C 搜尋不停頓。".format(score),
            flush=True,
        )

    def _worker(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            # C can report score 16 and score 8 only milliseconds apart.
            # Debounce briefly and retain the newest (strictly better) elite
            # so an obsolete completion does not occupy the solver first.
            stop_after_current = False
            while True:
                try:
                    newer = self._queue.get(timeout=0.5)
                except queue.Empty:
                    break
                if newer is None:
                    stop_after_current = True
                    break
                item = newer
            try:
                a, b, score, encoded = item
                completion_config = PipelineConfig(
                    L=self.length,
                    seed=self.args.seed,
                    seconds=0,
                    repair=True,
                    z3=self.args.z3,
                    z3_timeout=self.args.z3_timeout,
                    z3_trigger_score=self.args.z3_trigger_score,
                    z3_guidance_top_k=self.args.z3_guidance_top_k,
                    z3_profile_radius=self.args.z3_profile_radius,
                    z3_guided=not self.args.z3_broad,
                    repair_beam_width=self.args.repair_beam_width,
                    repair_max_depth=self.args.repair_max_depth,
                    z3_completion_radius=self.args.z3_completion_radius,
                    reference_port=True,
                    root=self.root,
                )
                elite = SAElite(a, b, score, self.args.seed, self.trigger_count)
                result = _run_optional_z3(
                    completion_config,
                    self.length,
                    (elite,),
                    self.root,
                    encoded or None,
                )
                self._results.append(result)
                if isinstance(result, dict):
                    print(
                        "[pipeline] completion：best_score={} → {} "
                        "(Z3 執行={})".format(
                            score,
                            result.get("status", "UNKNOWN"),
                            "是" if result.get("z3_executed", False) else "否",
                        ),
                        flush=True,
                    )
                if self.result_callback is not None:
                    self.result_callback(result)
            except BaseException as error:  # handed back to the main thread
                self._error = error
                print("[pipeline] completion error：{}".format(error), flush=True)
                return
            finally:
                if stop_after_current or self._closing.is_set():
                    return

    def close(self) -> None:
        """Finish already queued completion work before final reporting."""
        if self._thread is None:
            return
        # Any task already executing is allowed to finish so a SAT result
        # cannot be lost.  The worker then drops pending obsolete centres.
        self._closing.set()
        self._queue.put(None)
        self._thread.join()
        if self._error is not None:
            raise RuntimeError("background completion failed: {}".format(self._error))


def _group_count(length: int, root: Path = Path(".")) -> int:
    """Count saved sequence pairs using the established ``a=`` record line."""
    path = Path(root) / "{}.txt".format(length)
    if not path.exists():
        return 0
    return sum(
        1 for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()
        if line.startswith("a=")
    )


def _restart_count(state) -> int:
    """Translate zero-based SearchState restart indices to displayed runs."""
    return int(state.restart_index) + 1


def _swap_evaluations(state) -> int:
    """Return the exact sampled swap evaluations implied by completed moves."""
    parameters = getattr(state, "algorithm_parameters", None)
    samples = int(getattr(parameters, "proposal_samples", 1))
    return int(state.iteration) * samples


def _print_reference_start(length: object, workers: int, method: str,
                           initial_groups: int, reference_port: bool) -> None:
    """Print the reference project's Chinese start/progress layout."""
    print("[pipeline] 搜尋開始前既有結果：{} 組".format(initial_groups), flush=True)
    print(
        "[pipeline] 開始 FKM seed + Python local search；"
        "按 Ctrl-C 後進行獨立驗證",
        flush=True,
    )
    print(
        "必要條件通過；使用 {} 個執行緒開始搜尋 L={}。".format(
            workers, length
        ),
        flush=True,
    )
    if reference_port and isinstance(length, int):
        line = "壓縮引導：factor 2（{} -> {}）".format(length, length // 2)
        if length % 4 == 0:
            line += "，factor 4 多層路徑（{} -> {} -> {}）".format(
                length, length // 2, length // 4
            )
        print(line + "。", flush=True)
    else:
        print("搜尋引導：{}。".format(method), flush=True)
    print("尚未找到不代表不存在；可按 Ctrl-C 停止。", flush=True)


def _print_reference_progress(restarts: int, moves: int,
                              swap_evaluations: int, valid_results: int) -> None:
    """Print the same two counter lines as the reference search process."""
    print("搜尋進度：已測試 {} 組序列對".format(restarts), flush=True)
    _print_reference_stats(restarts, moves, swap_evaluations, valid_results)


def _print_reference_stats(restarts: int, moves: int,
                           swap_evaluations: int, valid_results: int) -> None:
    """Print the reference project's machine-readable counter line."""
    print(
        "搜尋統計：restart={}, moves={}, swap evaluations={}, "
        "valid results={}".format(
            restarts, moves, swap_evaluations, valid_results
        ),
        flush=True,
    )


def _with_new_valid_results(line: str, new_count: int) -> str:
    """Display persisted new pairs, not duplicate-inclusive solver hit counts."""
    marker = "valid results="
    if not line.startswith("搜尋統計：") or marker not in line:
        return line
    prefix, raw_count = line.rsplit(marker, 1)
    try:
        int(raw_count)
    except ValueError:
        return line
    return "{}{}{}".format(prefix, marker, new_count)


def _print_restarts_per_new_pqcp(restarts: int, new_solutions: int) -> None:
    """Print restarts per newly persisted, independently verified PQCP.

    ``new_solutions`` counts only sequence-pair groups newly added during this
    invocation.  Existing or duplicate solutions are therefore not included
    in the denominator.
    """
    if new_solutions > 0:
        print(
            "平均每找到 1 組新 PQCP 所需 restart：{:.3f} 次".format(
                restarts / new_solutions
            ),
            flush=True,
        )
    else:
        print(
            "平均每找到 1 組新 PQCP 所需 restart："
            "N/A（本輪新增 0 組）",
            flush=True,
        )


def _print_reference_candidate(length: int, candidate) -> None:
    """Print a verified discovery using the reference project's result shape."""
    shifts = [u for u in range(1, length) if candidate.profile[u] != 0]
    value = candidate.profile[shifts[0]] if shifts else 0
    if len(shifts) == 2:
        print(
            "找到 ({},4)-PQCP！非零位移為 {} 與 {}，數值為 {}。".format(
                length, shifts[0], shifts[1], value
            )
        )
    else:  # Defensive only: the independent verifier defines valid output.
        print("找到 ({},4)-PQCP！".format(length))
    print("a = {}".format("".join(map(str, candidate.a))))
    print("b = {}".format("".join(map(str, candidate.b))))


def parse_args(argv=None):
    """Parse a new run or a checkpoint resume without combining time units."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--L", type=int, help="Project 2 length (or positive experimental length)")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--workers", type=int, default=8,
        help="C search threads (the one-click default is 8)",
    )
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--minutes", type=float)
    parser.add_argument("--hours", type=float)
    parser.add_argument("--z3", action="store_true", help="run existing small-radius Z3 portfolio on near global bests")
    parser.add_argument("--repair", action="store_true", help="run guidance/beam repair without requiring Z3")
    parser.add_argument(
        "--z3-timeout",
        type=float,
        default=DEFAULT_Z3_TIMEOUT_SECONDS,
        help="seconds allowed for each Z3 task (default: 60)",
    )
    parser.add_argument(
        "--z3-trigger-score", type=int, default=DEFAULT_Z3_TRIGGER_SCORE,
        help="correlation-guidance score guard (default: 16)",
    )
    parser.add_argument("--z3-guidance-top-k", type=int, default=10)
    parser.add_argument("--z3-profile-radius", type=int, default=4)
    parser.add_argument("--repair-beam-width", type=int, default=DEFAULT_REPAIR_BEAM_WIDTH)
    parser.add_argument("--repair-max-depth", type=int, default=DEFAULT_REPAIR_MAX_DEPTH)
    parser.add_argument("--z3-completion-radius", type=int, default=DEFAULT_Z3_COMPLETION_RADIUS)
    parser.add_argument(
        "--z3-broad", action="store_true",
        help="rollback option: use the previous broad radius-three Z3 operation",
    )
    parser.add_argument("--enhanced", action="store_true", help="use sampled fixed-weight escape mode")
    parser.add_argument(
        "--gcp", action="store_true",
        help="initialize from a verified periodic or Turyn-adapted Golay seed",
    )
    parser.add_argument(
        "--reference-port", action="store_true", default=True,
        help="use the audited compressed-FKM/multiscale reference-project preset (default and fixed production mode)",
    )
    parser.add_argument(
        "--legacy-sampler", action="store_true",
        help="use the previous rejection sampler for C same-parity swaps",
    )
    parser.add_argument(
        "--python-backend", action="store_true",
        help="explicit rollback/debug path using the slower Python search loop",
    )
    parser.add_argument("--resume", type=Path, help="resume an existing baseline/enhanced checkpoint")
    parser.add_argument(
        "--resume-parallel", action="store_true",
        help="resume every private checkpoint for the selected --workers portfolio",
    )
    parser.add_argument("--checkpoint-interval", type=float, default=60.0)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def time_budget(args) -> float:
    """Return one supplied budget, or run until Ctrl+C when none is supplied."""
    supplied = [("--seconds", args.seconds, 1.0), ("--minutes", args.minutes, 60.0), ("--hours", args.hours, 3600.0)]
    active = [(name, value, factor) for name, value, factor in supplied if value is not None]
    if len(active) > 1:
        raise ValueError("specify only one of --seconds, --minutes, or --hours")
    if not active:
        return float("inf")
    name, value, factor = active[0]
    if value is None or value <= 0:
        raise ValueError("{} must be positive".format(name))
    return value * factor


def main(argv=None) -> int:
    """Print a concise run report; no non-solution is interpreted as nonexistence."""
    args = parse_args(argv)
    try:
        # Clicking VS Code's "Run Python File" starts this file with no
        # arguments.  Keep that path approachable while preserving strict CLI
        # validation when the caller did supply arguments.
        if args.L is None and args.resume is None and argv is None and len(sys.argv) == 1:
            raw_length = input("請輸入長度 L（偶數）：").strip()
            if not raw_length:
                raise ValueError("L 不可空白")
            try:
                args.L = int(raw_length)
            except ValueError as error:
                raise ValueError("L 必須是正整數") from error
            raw_seed = input("請輸入 seed（直接按 Enter 使用 123）：").strip()
            if raw_seed:
                try:
                    args.seed = int(raw_seed)
                except ValueError as error:
                    raise ValueError("seed 必須是整數") from error
            # Production search is deliberately fixed to the audited
            # compressed-FKM reference preset.  The old implementations stay
            # available as library-level rollback paths, but the one-click
            # entry point no longer asks the user to choose between them.
            args.reference_port = True
            args.gcp = False
            args.enhanced = False
            raw_z3 = input(
                "是否啟用 Z3 completion？（直接按 Enter = 否，輸入 y = 是）："
            ).strip().lower()
            if raw_z3 not in ("", "y", "yes"):
                raise ValueError("Z3 選擇只能留空或輸入 y")
            use_z3 = raw_z3 in ("y", "yes")
            # Z3 completion is fed by a background queue; it does not replace
            # or synchronously stop the eight compiled search threads.
            args.workers = 8
            args.repair = use_z3
            args.z3 = use_z3
        budget = time_budget(args)
        if args.L is None and args.resume is None:
            raise ValueError("--L is required unless --resume is supplied")
        if args.L is not None:
            listed = validate_length(args.L)
            if not listed:
                print("Experimental L={}; listed Project 2 lengths are {}.".format(args.L, sorted(PROJECT_LENGTHS)))
        if args.checkpoint_interval <= 0:
            raise ValueError("--checkpoint-interval must be positive")
        if args.workers <= 0:
            raise ValueError("--workers must be positive")
        if args.z3_timeout < 0:
            raise ValueError("--z3-timeout must be non-negative")
        if args.python_backend and args.workers > 1 and (args.z3 or args.repair):
            raise ValueError("parallel workers use nonblocking SA only; use --workers 1 for repair/Z3")
        if args.python_backend and args.workers > 1 and args.resume is not None:
            raise ValueError("the existing --resume checkpoint is single-worker; use --workers 1")
        if args.resume_parallel and args.workers <= 1:
            raise ValueError("--resume-parallel requires --workers greater than 1")
        if args.workers > 1 and args.enhanced:
            raise ValueError("parallel production mode currently uses the measured baseline SA")
        if args.reference_port and (args.gcp or args.enhanced):
            raise ValueError("正式入口已固定使用壓縮 FKM，不能與 --gcp 或 --enhanced 同時使用")
        requested_method = "from checkpoint" if args.resume is not None else (
            "compressed FKM + multiscale reference SA" if args.reference_port else (
            "FKM + enhanced SA" if args.enhanced else (
                "length-adapted GCP + guided fixed-weight SA" if args.gcp else (
                    "FKM + guided fixed-weight SA" if args.L in PROJECT_LENGTHS else "FKM + fixed-weight SA"
                )
            )
        ))
        display_length = args.L if args.L is not None else "from checkpoint"
        initial_groups = _group_count(args.L) if args.L is not None else 0
        if args.resume is None and not args.python_backend:
            if args.L is None:  # guarded above; keeps the type contract clear
                raise ValueError("L is required for the C backend")
            print("[pipeline] 搜尋開始前既有結果：{} 組".format(initial_groups), flush=True)
            print(
                "[pipeline] 開始 compressed FKM seed + C local search；"
                "按 Ctrl-C 後進行最終驗證",
                flush=True,
            )
            output_lock = threading.RLock()
            latest_statistics = (
                "搜尋統計：restart=0, moves=0, swap evaluations=0, valid results=0"
            )
            def c_output_notice(line):
                nonlocal latest_statistics
                with output_lock:
                    if line.startswith("搜尋統計："):
                        latest_statistics = line
                        # Both C and completion use the same verified writer.
                        # Count persisted new groups, never raw hits or SATs.
                        count = max(0, _group_count(args.L) - initial_groups)
                        line = _with_new_valid_results(line, count)
                    print(line, flush=True)
                    if args.z3 and line.startswith("搜尋統計："):
                        print(
                            "Z3 觸發使用次數：{}".format(bridge.z3_execution_count),
                            flush=True,
                        )

            def completion_notice(_result):
                # Completion can finish between C reports or after C stops.
                # Refresh from the same persisted count in either case.
                with output_lock:
                    c_output_notice(latest_statistics)

            bridge = _CCompletionBridge(
                args, args.L, Path("."), result_callback=completion_notice
            )
            try:
                c_result = run_c_search(
                    CSearchConfig(
                        L=args.L,
                        seed=args.seed,
                        seconds=budget,
                        threads=args.workers,
                        candidate_count=2,
                        root=Path("."),
                        # The parent owns the persisted-new-result display
                        # with or without asynchronous completion.
                        echo=False,
                        emit_elites=bool(args.repair or args.z3),
                        direct_swap_sampling=not args.legacy_sampler,
                    ),
                    line_callback=c_output_notice,
                    elite_callback=bridge.consider if bridge.enabled else None,
                )
            finally:
                bridge.close()
            current_groups = _group_count(args.L)
            new_groups = max(0, current_groups - initial_groups)
            print("\n========== 最終驗證摘要 ==========" , flush=True)
            print(
                "C 候選共 {} 組；{} 組通過 Python 獨立 verifier。".format(
                    c_result.valid_results, c_result.verified_candidates
                )
            )
            print(
                "本輪新增序列對：{} 組；檔案累計：{} 組。".format(
                    new_groups, current_groups
                )
            )
            print("\n========== 搜尋效率統計 ==========" , flush=True)
            print("總耗時：{:.3f} 秒".format(c_result.elapsed))
            print("moves/second：{:.0f}".format(c_result.moves_per_second))
            print(
                "swap evaluations/second：{:.0f}".format(
                    c_result.swap_evaluations_per_second
                )
            )
            print(
                "compressed FKM seeds：applied={}，misses={}".format(
                    c_result.fkm_seeds_applied, c_result.fkm_seed_misses
                )
            )
            print("壓縮 FKM seed bank：{}".format(c_result.seed_bank_path))
            print("完整 C 搜尋輸出已保存：{}".format(c_result.log_path))
            print(
                "最終 valid results（本輪新增，C + completion）：{}".format(
                    new_groups
                )
            )
            _print_restarts_per_new_pqcp(c_result.restarts, new_groups)
            if args.z3:
                print(
                    "Z3 觸發使用次數：{}".format(
                        bridge.z3_execution_count
                    )
                )
            return 0
        _print_reference_start(
            display_length, args.workers, requested_method,
            initial_groups, args.reference_port,
        )
        if args.repair or args.z3:
            print(
                "[pipeline] completion：{} | score <= {} | profile LB <= {} | "
                "top-K = {} | timeout = {}s".format(
                    "broad Z3 radius 3" if args.z3_broad else (
                        "guidance -> beam(depth={},width={}) -> Z3 radius {} fallback".format(
                            args.repair_max_depth, args.repair_beam_width, args.z3_completion_radius
                        )
                    ),
                    args.z3_trigger_score,
                    args.z3_profile_radius,
                    args.z3_guidance_top_k,
                    args.z3_timeout,
                )
            )
        if args.workers > 1:
            def parallel_progress(snapshot):
                _print_reference_progress(
                    snapshot["restarts"], snapshot["iterations"],
                    snapshot["swap_evaluations"],
                    max(0, _group_count(args.L) - initial_groups),
                )

            parallel_result = run_parallel_pipeline(
                ParallelPipelineConfig(
                    L=args.L,
                    workers=args.workers,
                    seconds=budget,
                    base_seed=args.seed,
                    gcp=args.gcp,
                    reference_port=args.reference_port,
                    checkpoint_interval=args.checkpoint_interval,
                    progress_interval=10.0 if args.verbose else 60.0,
                    resume=args.resume_parallel,
                ),
                progress_callback=parallel_progress,
            )
            candidate = parallel_result.global_best
            total_iterations = sum(report.iteration for report in parallel_result.reports)
            total_restarts = sum(report.restart_index + 1 for report in parallel_result.reports)
            proposal_samples = (
                2 if args.reference_port else DEFAULT_GUIDED_PROPOSAL_SAMPLES
            )
            total_swap_evaluations = total_iterations * proposal_samples
            if parallel_result.interrupted:
                print("\n[pipeline] 搜尋已停止", flush=True)
            if candidate is not None and candidate.verified:
                _print_reference_candidate(parallel_result.L, candidate)
                if parallel_result.new_solution_count:
                    print("新結果已追加寫入 {}.txt".format(parallel_result.L))
                else:
                    print("這組結果已存在於 {}.txt，不重複寫入。".format(parallel_result.L))
            else:
                print(
                    "\n搜尋已停止，共執行 {} 次 restart；"
                    "目前無法判定是否存在。".format(total_restarts)
                )
            valid_results = sum(
                len(report.verified_candidates) for report in parallel_result.reports
            )
            _print_reference_stats(
                total_restarts, total_iterations, total_swap_evaluations,
                valid_results,
            )
            current_groups = _group_count(parallel_result.L)
            new_groups = max(0, current_groups - initial_groups)
            print("\n========== 最終驗證摘要 ==========", flush=True)
            if candidate is not None and candidate.verified:
                print("最佳候選已通過獨立 verifier，並且符合 Project 2 第三題的額外條件。")
            else:
                print("最佳候選未通過獨立 verifier。")
            print(
                "本輪新增序列對：{} 組；檔案累計：{} 組。".format(
                    new_groups, current_groups
                )
            )
            print("\n========== 搜尋效率統計 ==========", flush=True)
            print("總耗時：{:.3f} 秒".format(parallel_result.elapsed))
            print(
                "搜尋前結果：{} 組；本輪新增：{} 組；累計：{} 組".format(
                    initial_groups, new_groups,
                    current_groups,
                )
            )
            print(
                "搜尋統計：restart={}, moves={}, swap evaluations={}, "
                "valid results={}".format(
                    total_restarts, total_iterations, total_swap_evaluations,
                    valid_results,
                )
            )
            _print_restarts_per_new_pqcp(total_restarts, new_groups)
            print("完整搜尋摘要已保存：{}".format(parallel_result.summary_path))
            if parallel_result.interrupted:
                print("搜尋 checkpoint 已保存於：{}".format(
                    parallel_result.summary_path.parent / "workers"
                ))
            return 0

        config = PipelineConfig(
            L=args.L, seed=args.seed, seconds=budget, enhanced=args.enhanced, z3=args.z3,
            repair=args.repair or args.z3,
            gcp=args.gcp,
            reference_port=args.reference_port,
            z3_timeout=args.z3_timeout, z3_trigger_score=args.z3_trigger_score, resume=args.resume,
            z3_guidance_top_k=args.z3_guidance_top_k,
            z3_profile_radius=args.z3_profile_radius,
            repair_beam_width=args.repair_beam_width,
            repair_max_depth=args.repair_max_depth,
            z3_completion_radius=args.z3_completion_radius,
            z3_guided=not args.z3_broad,
            checkpoint_interval=args.checkpoint_interval,
            progress_interval=10.0 if args.verbose else 60.0,
            verbose=args.verbose,
        )
        def progress(state, _elapsed):
            _print_reference_progress(
                _restart_count(state), state.iteration,
                _swap_evaluations(state),
                max(0, _group_count(state.L) - initial_groups),
            )
        def z3_notice(result, score):
            if isinstance(result, dict):
                print(
                    "[pipeline] completion：best_score={} → {}"
                    "（Z3 執行={}，下一步={}）".format(
                        score, result["status"], "是" if result.get("z3_executed", False) else "否",
                        "新的 SA restart" if result.get("restart_recommended", False) else "繼續",
                    ), flush=True,
                )
            else:
                print("[pipeline] Z3：best_score={} SAT={} UNSAT={} UNKNOWN={}".format(
                    score, result.status_counts["SAT"], result.status_counts["UNSAT"], result.status_counts["UNKNOWN"]
                ), flush=True)
        result = run_pipeline(config, progress=progress, z3_callback=z3_notice)
    except (ValueError, OSError, RuntimeError, ParallelWorkerError) as error:
        print("error: {}".format(error))
        return 2
    restarts = _restart_count(result.state)
    moves = int(result.state.iteration)
    swap_evaluations = _swap_evaluations(result.state)
    valid_results = len(result.verified_paths)
    if result.interrupted:
        print("\n[pipeline] 搜尋已停止", flush=True)
    if result.verification.is_valid:
        candidate = type("CandidateView", (), {
            "a": result.state.best_a,
            "b": result.state.best_b,
            "profile": result.verification.profile,
        })()
        _print_reference_candidate(result.L, candidate)
        if _group_count(result.L) > initial_groups:
            print("新結果已追加寫入 {}.txt".format(result.L))
        else:
            print("這組結果已存在於 {}.txt，不重複寫入。".format(result.L))
    else:
        print(
            "\n搜尋已停止，共執行 {} 次 restart；"
            "目前無法判定是否存在。".format(restarts)
        )
    _print_reference_stats(restarts, moves, swap_evaluations, valid_results)

    current_groups = _group_count(result.L)
    new_groups = max(0, current_groups - initial_groups)
    print("\n========== 最終驗證摘要 ==========", flush=True)
    if result.verification.is_valid:
        print("最佳候選已通過獨立 verifier，並且符合 Project 2 第三題的額外條件。")
    else:
        print("最佳候選未通過獨立 verifier。")
    print(
        "本輪新增序列對：{} 組；檔案累計：{} 組。".format(
            new_groups, current_groups
        )
    )

    print("\n========== 搜尋效率統計 ==========", flush=True)
    print("總耗時：{:.3f} 秒".format(result.elapsed))
    print(
        "搜尋前結果：{} 組；本輪新增：{} 組；累計：{} 組".format(
            initial_groups, new_groups,
            current_groups,
        )
    )
    print(
        "搜尋統計：restart={}, moves={}, swap evaluations={}, "
        "valid results={}".format(
            restarts, moves, swap_evaluations, valid_results
        )
    )
    _print_restarts_per_new_pqcp(restarts, new_groups)
    print("完整搜尋事件已保存：{}".format(result.log_path))
    print("搜尋 checkpoint 已保存：{}".format(result.checkpoint_path))
    print("最佳候選已保存：{}".format(result.best_path))
    if result.z3_result is not None:
        if isinstance(result.z3_result, dict):
            print("Completion：{} ({})".format(
                result.z3_result["status"], result.z3_result.get("reason") or "completed"
            ))
        else:
            print("Z3: SAT={} UNSAT={} UNKNOWN={}".format(
                result.z3_result.status_counts["SAT"], result.z3_result.status_counts["UNSAT"], result.z3_result.status_counts["UNKNOWN"]
            ))
        print("Z3 觸發次數：{}".format(result.z3_trigger_count))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
