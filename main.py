"""Run PyTorch continuous matrices -> exact swap completion -> verification."""

import argparse
from collections import deque
from dataclasses import asdict
import json
import math
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
    """Background exact compressed lifts plus the existing radius fallback."""

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
        self._observation_lock = threading.RLock()
        self._lift_pool = None
        self._multilevel_pool = None
        self._staged_store = None
        self._restored_work = deque()
        compressed_dedup = getattr(args, "z3_compressed_dedup", True)
        if args.z3 and getattr(args,"z3_staged",True):
            from solver.staged_completion import CompletionWorkStore, SlicePolicy
            policy=SlicePolicy(
                initial_ms=math.ceil(getattr(args,"z3_slice_seconds",1.0)*1000),
                maximum_ms=math.ceil(args.z3_timeout*1000),
                max_middle_models=getattr(args,"z3_multilevel_max_models",16),
                cross_pruning=getattr(args,"z3_cross_compression_pruning",True),
                rectangle_search=getattr(args,"z3_rectangle_search",True),
                rectangle_restarts=getattr(args,"z3_rectangle_restarts",4),
                rectangle_moves=getattr(args,"z3_rectangle_moves",1000),
                rectangle_samples=getattr(args,"z3_rectangle_samples",8),
                middle_mitm=getattr(args,"z3_middle_mitm",True),
                affine_lift=getattr(args,"z3_affine_lift",True),
                mitm_index_entries=getattr(args,"z3_mitm_index_entries",250000),
                affine_max_assignments=getattr(args,"z3_affine_max_assignments",1048576),
                affine_backend=getattr(args,"z3_affine_backend","auto"),
                mitm_batch_size=getattr(args,"z3_mitm_batch_size",4))
            self._staged_store=CompletionWorkStore(
                root,length,args.seed,policy,deduplicate_equivalents=compressed_dedup)
            for key in self._staged_store.recover():
                record = self._staged_store.read(key)
                if record["kind"] == "lift":
                    if not getattr(args, "z3_compressed_lift", True):
                        continue
                    if record["spec"]["factor"] == 4 and not getattr(args, "z3_multilevel_lift", True):
                        continue
                self._restored_work.append(key)
            print("[pipeline] Z3 分段求解已啟用：初始 {:.3f}s，逐次加倍；恢復 {} 個待解工作。".format(
                policy.clamp(policy.initial_ms)/1000,len(self._restored_work)),flush=True)
            if length%8 == 4 and getattr(args,"z3_compressed_lift",True):
                if policy.middle_mitm and getattr(args,"z3_multilevel_lift",True):
                    print("[pipeline] 中間層 MITM 已啟用：固定粗壓縮配對、子問題去重；超限交回 Z3。",flush=True)
                if policy.affine_lift:
                    print("[pipeline] 精確 GF(2) 方向縮減已啟用：小空間分段展開；較大空間交回 Z3。",flush=True)
            self._fill_restored_work()
            if compressed_dedup and self._staged_store.recovery_duplicates:
                print("[pipeline] 舊 Z3 checkpoint 去重：保留代表工作，略過 {} 個等價副本（原檔保留）。".format(
                    self._staged_store.recovery_duplicates), flush=True)
        if args.z3 and getattr(args, "z3_compressed_lift", True):
            from solver.compressed_completion import CompressedLiftPool
            self._lift_pool = CompressedLiftPool(length, deduplicate_equivalents=compressed_dedup)
            if compressed_dedup:
                print("[pipeline] 壓縮 Z3 工作等價去重已啟用（模式、target、content 共同變換）。", flush=True)
            print(
                "[pipeline] Z3：精確 factor-2 lift 已啟用（不限制 Hamming radius）；"
                "保留原本 radius completion。", flush=True,
            )
            if length % 4 == 0 and getattr(args, "z3_multilevel_lift", True):
                self._multilevel_pool = CompressedLiftPool(
                    length, factor=4, deduplicate_equivalents=compressed_dedup)
                print("[pipeline] Z3：多層精確壓縮 {} → {} → {} 已啟用。".format(
                    length // 4, length // 2, length), flush=True)
                if (length//4)%2 and getattr(args, "z3_cross_compression_pruning", True):
                    print("[pipeline] 精確 factor-4／factor-{} 雙向壓縮相容性剪枝已啟用。".format(
                        length//4), flush=True)
                    if length >= 12 and self._staged_store and self._staged_store.policy.rectangle_search:
                        print("[pipeline] C 矩形交換搜尋已啟用：保持兩種壓縮；有限 restart 後交回精確 Z3。", flush=True)
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
                count += int(result.get("z3_calls", bool(result.get("z3_executed", False))))
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
        """Exact compression matches bypass score/radius guards, not verification."""
        if not self.enabled:
            return
        with self._observation_lock:
            self._consider_locked(a, b, metadata)

    def _fill_restored_work(self):
        while self._restored_work and self._queue.qsize()<16:
            self._queue.put({"kind":"scheduled","key":self._restored_work.popleft()})

    def _enqueue_staged(self,kind,spec):
        key=self._staged_store.register(kind,spec)
        if key is None:
            return False
        self._queue.put({"kind":"scheduled","key":key})
        return True

    def _consider_locked(self, a, b, metadata) -> None:
        for pool in (self._multilevel_pool, self._lift_pool):
            if pool is None:
                continue
            for job in pool.offer(a, b, metadata["k"], metadata["sign"]):
                if self._queue.qsize() >= 16:
                    break  # bounded background work; not an impossibility claim
                queued=(self._enqueue_staged("lift",asdict(job)) if self._staged_store else True)
                if self._staged_store is None:
                    self._queue.put({"kind": "compressed_lift", "job": job})
                pool.submit(job)
                if not queued: continue
                print(
                    "[pipeline] 精確 factor-{} 配對 → Z3 lift queue："
                    "k={} sign={}；主搜尋繼續執行。".format(
                        job.factor, job.target.k, job.target.eta,
                    ), flush=True,
                )
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
        if self._staged_store is not None:
            self._enqueue_staged("radius",{
                "a":tuple(a),"b":tuple(b),"score":score,"encoded":encoded,
                "parameters":{
                    "z3_trigger_score":self.args.z3_trigger_score,
                    "z3_guidance_top_k":self.args.z3_guidance_top_k,
                    "z3_profile_radius":self.args.z3_profile_radius,
                    "z3_guided":not self.args.z3_broad,
                    "repair_beam_width":self.args.repair_beam_width,
                    "repair_max_depth":self.args.repair_max_depth,
                    "z3_completion_radius":self.args.z3_completion_radius}})
        else:
            self._queue.put((tuple(a), tuple(b), score, encoded))
        print(
            "[pipeline] elite best_score={} 已排入 completion queue；"
            "主搜尋繼續執行。".format(score),
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
            # Preserve the old debounce in rollback mode. With lift jobs,
            # never discard a different compressed pattern in favour of a
            # newer low score: these are distinct exact subproblems.
            while self._lift_pool is None and self._staged_store is None:
                try:
                    newer = self._queue.get(timeout=0.5)
                except queue.Empty:
                    break
                if newer is None:
                    stop_after_current = True
                    break
                item = newer
            try:
                if isinstance(item,dict) and item.get("kind")=="scheduled":
                    self._run_staged_turn(item["key"])
                    continue
                if isinstance(item, dict) and item.get("kind") == "compressed_lift":
                    from solver.compressed_completion import run_compressed_lift_job
                    options = ({"max_middle_models": getattr(self.args, "z3_multilevel_max_models", 16),
                                "cross_compression_pruning": getattr(self.args, "z3_cross_compression_pruning", True)}
                               if getattr(item["job"], "factor", 2) == 4 else {})
                    result = run_compressed_lift_job(
                        item["job"], int(self.args.z3_timeout * 1000), self.root, self.args.seed,
                        **options)
                    self._results.append(result)
                    if result.get("factor") == 4:
                        print("[pipeline] Z3 multilevel lift：{} middle_models={} "
                              "Z3 checks={} cross_pairs={} cross_pruned={} construction={:.3f}s solve={:.3f}s".format(
                                  result["status"], result["middle_models"], result["z3_calls"],
                                  result.get("compatible_cross_pairs",0), result.get("middle_cross_pruned",0),
                                  result["construction_time"], result["solve_time"]), flush=True)
                    else:
                        print(
                            "[pipeline] Z3 compressed lift：{} variables={} "
                            "construction={:.3f}s solve={:.3f}s".format(
                                result["status"], result["orientation_variables"],
                                result["construction_time"], result["solve_time"],
                            ), flush=True,
                        )
                    if self.result_callback is not None:
                        self.result_callback(result)
                    continue
                a, b, score, encoded = item
                if self._lift_pool is not None and self._last_score is not None and score > self._last_score:
                    continue  # obsolete radius center, not a dropped lift
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

    def _run_staged_turn(self,key):
        """Run one quantum, then put unresolved work behind other jobs."""
        store=self._staged_store
        record=store.read(key)
        if record["kind"]=="lift":
            result=store.advance_lift(key)
        else:
            spec=record["spec"]
            budget=store.policy.clamp(record["state"]["work"]["budget_ms"])
            record["state"]["work"]["budget_ms"]=budget
            config=PipelineConfig(L=self.length,seed=self.args.seed,seconds=0,
                repair=True,z3=True,z3_timeout=budget/1000,reference_port=True,root=self.root,
                **spec["parameters"])
            elite=SAElite(tuple(spec["a"]),tuple(spec["b"]),spec["score"],self.args.seed,self.trigger_count)
            raw=_run_optional_z3(config,self.length,(elite,),self.root,
                                 tuple(tuple(x) for x in spec["encoded"]) or None)
            if isinstance(raw,dict):
                result=raw
                unknown="UNKNOWN" in str(raw.get("status","")) and not raw.get("solved",False)
            else:
                counts=getattr(raw,"status_counts",{})
                solved=bool(getattr(raw,"solved",False))
                if getattr(raw, "implementation_error", False) or (counts.get("SAT", 0) and not solved):
                    raise RuntimeError("Z3 portfolio SAT failed independent verification")
                # The existing broad fallback schedules exactly one radius-3
                # task. A budget may expire before it starts; no task is NOT
                # a proof of UNSAT and must remain recoverable.
                unknown=not solved and (bool(counts.get("UNKNOWN",0)) or not counts.get("UNSAT",0))
                result={"status":"Z3_SAT" if solved else "Z3_UNKNOWN" if unknown else "Z3_UNSAT",
                        "solved":solved,"z3_executed":bool(sum(counts.values())),
                        "z3_calls":sum(counts.values()),"status_counts":counts}
            store.finish_radius(key,record,unknown,result.get("status","UNKNOWN"))
            result.update(staged=True,phase=record["phase"],slice_ms=budget,work_kind="radius",
                          checkpoint=str(store.path(key)))
        self._results.append(result)
        phase=result["phase"]
        if result["work_kind"] == "rectangle":
            print("[pipeline] 雙向壓縮矩形搜尋：{} E_full={} restart={}/{} "
                  "moves={} rectangle evaluations={}（Z3 執行=否；進度已保存）".format(
                      result["status"],result["rectangle_energy"],result["rectangle_restart"],
                      result["rectangle_restart_limit"],result["rectangle_moves"],
                      result["rectangle_evaluations"]),flush=True)
        elif result.get("binary_backend") == "affine":
            print("[pipeline] GF(2) 精確展開：child={} parent={} backend={} free_bits={} checked={}/{} next={} "
                  "（Z3 執行=否；進度已保存）".format(
                      result["affine_status"],result["status"],result.get("affine_backend","python"),
                      result["affine_free_bits"],result["affine_next_index"],
                      result["affine_assignments"],
                      "queue" if phase=="READY" else "saved pending" if phase=="WAITING" else "finished"),flush=True)
        elif result.get("middle_backend") == "mitm":
            print("[pipeline] 中間層 MITM：{} middle_models={} duplicates={} next={} "
                  "（僅壓縮候選；Z3 執行=否；進度已保存）".format(
                      result["mitm_status"],result["middle_models"],result["mitm_duplicates"],
                      "queue" if phase=="READY" else "saved pending" if phase=="WAITING" else "finished"),flush=True)
        else:
            print("[pipeline] Z3 分段求解：{} work={} slice={:.3f}s next={}（進度已保存）".format(
                result["status"],result["work_kind"],result["slice_ms"]/1000,
                "queue" if phase=="READY" else "saved pending" if phase=="WAITING" else "finished"),flush=True)
        if self.result_callback is not None:
            self.result_callback(result)
        with self._observation_lock:
            # A queued task is already durable, including on Ctrl+C.
            # Give restored backlog a slot before putting this task back.
            self._fill_restored_work()
            if phase=="READY": self._queue.put({"kind":"scheduled","key":key})

    def close(self) -> None:
        """Finish the active turn; staged pending jobs remain durable on disk."""
        if self._thread is None:
            return
        # Any task already executing is allowed to finish so a SAT result
        # cannot be lost. Staged queued centres have already been saved;
        # rollback mode retains its original discard-on-shutdown behaviour.
        self._closing.set()
        self._queue.put(None)
        self._thread.join()
        if self._staged_store is not None:
            print("[pipeline] Z3 待解進度：{}（下次相同 L／seed 自動接續）".format(
                self._staged_store.directory),flush=True)
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
        "--backend", choices=("torch", "mps", "cuda", "adam", "c"), default="torch",
        help=("search method; torch is the portable default, mps/cuda are "
              "device shortcuts, and adam/c are rollback ablations"),
    )
    parser.add_argument(
        "--torch-device", choices=("auto", "mps", "cuda", "cpu"), default="auto",
        help="PyTorch device (auto: CUDA, then MPS, then CPU)",
    )
    parser.add_argument(
        "--torch-batch-size", type=int,
        help="continuous matrix lanes (default: 2048 through L=58, then 1024)",
    )
    parser.add_argument("--torch-steps-per-restart", type=int, default=2000)
    parser.add_argument("--torch-observation-interval", type=int, default=25)
    parser.add_argument("--torch-kernel", choices=("direct", "dft"), default="dft",
                        help="continuous PACF kernel; discrete verification always stays exact")
    parser.add_argument("--torch-loss", choices=("legacy", "balanced", "projected"), default="balanced",
                        help="continuous loss only; legacy restores old fold weights")
    parser.add_argument("--torch-loss-backend", choices=("profile", "spectral"), default="profile",
                        help="equivalent loss evaluation; spectral requires DFT")
    parser.add_argument("--torch-optimization-mode", choices=("relaxed", "straight_through", "douglas_rachford"),
                        default="relaxed", help="continuous method; feasibility projection is experimental")
    parser.add_argument("--torch-observation-backend", choices=("torch", "metal"), default="torch",
                        help="exact observation implementation; metal requires MPS and L<=94")
    parser.add_argument("--torch-fast-observation", action="store_true",
                        help="reuse bounded independent proofs for identical nonzero candidates")
    parser.add_argument(
        "--torch-frequency-pruning",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=("exact L/4 content-profile and certified all-frequency PSD "
              "candidate pruning (production default: enabled)"),
    )
    parser.add_argument(
        "--torch-symmetry-pruning",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="keep one complement/rotation/A-B-swap content representative (default: enabled)",
    )
    parser.add_argument(
        "--torch-half-shift-seeding",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="construct seeds with the exact necessary S(L/2)=0 condition (default: enabled)",
    )
    parser.add_argument(
        "--torch-compression-pairing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="exact factor-2/factor-4 compressed PACF hash pairing (default: enabled)",
    )
    parser.add_argument("--torch-mathematical-loss",
                        choices=("none", "psd_cap", "divisor_lift", "lattice", "variance", "combined",
                                 "lattice_bootstrap"),
                        default="lattice",
                        help="extra mathematical guidance (default: lattice; none disables it); exact verifier is unchanged")
    parser.add_argument("--mps-initial-seconds", type=float,
                        help="optional earlier first seed handoff; later cycles keep their normal duration")
    parser.add_argument("--mps-pair-seed-percent", type=int, default=100,
                        help="C restarts from PyTorch elites, 0..100 (legacy option name)")
    parser.add_argument(
        "--mps-seconds-per-cycle", type=float, default=45.0,
        help="continuous PyTorch formation time (legacy option name; default: 45)",
    )
    parser.add_argument(
        "--cpu-seconds-per-cycle", type=float, default=15.0,
        help="compiled annealing completion time in each cycle (default: 15)",
    )
    parser.add_argument(
        "--legacy-kick", action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--adaptive-kick", action="store_true",
        help="experimental mild reheating after C stagnation kicks",
    )
    parser.add_argument(
        "--c-quench", action=argparse.BooleanOptionalAction, default=True,
        help="exact energy + PQCP-score one-swap descent (default: enabled)",
    )
    parser.add_argument("--torch-full-only", action="store_true",
                        help="ablation: omit folded residual energy from PyTorch loss")
    parser.add_argument(
        "--workers", type=int, default=8,
        help="C search threads (the one-click default is 8)",
    )
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--minutes", type=float)
    parser.add_argument("--hours", type=float)
    parser.add_argument("--z3", action=argparse.BooleanOptionalAction, default=True,
                        help="enable Z3 completion (default: enabled; --no-z3 disables it)")
    parser.add_argument(
        "--z3-compressed-lift", action=argparse.BooleanOptionalAction, default=True,
        help="with --z3, solve exact factor-2 matches without a Hamming radius; "
             "--no-z3-compressed-lift restores the previous completion path",
    )
    parser.add_argument("--repair", action="store_true", help="run guidance/beam repair without requiring Z3")
    parser.add_argument("--z3-multilevel-lift", action=argparse.BooleanOptionalAction, default=True,
                        help="with compressed Z3 completion and L divisible by 4, lift L/4 -> L/2 -> L")
    parser.add_argument("--z3-multilevel-max-models", type=int, default=16,
                        help="maximum intermediate models per job; hitting the cap means UNKNOWN")
    parser.add_argument("--z3-cross-compression-pruning", action=argparse.BooleanOptionalAction, default=True,
                        help="exact factor-4/factor-(L/4) compatibility for L/4 odd (default: enabled)")
    parser.add_argument("--z3-staged",action=argparse.BooleanOptionalAction,default=True,
                        help="durable round-robin Z3 slices (default: enabled; --no-z3-staged restores one-shot checks)")
    parser.add_argument("--z3-compressed-dedup", action=argparse.BooleanOptionalAction, default=True,
                        help="deduplicate equivalent fixed compression/target/content jobs (default: enabled)")
    parser.add_argument("--z3-rectangle-search", action=argparse.BooleanOptionalAction, default=True,
                        help="bounded fixed-dual-compression C search before staged factor-4 Z3 (default: enabled)")
    parser.add_argument("--z3-rectangle-restarts", type=int, default=4)
    parser.add_argument("--z3-rectangle-moves", type=int, default=1000)
    parser.add_argument("--z3-rectangle-samples", type=int, default=8)
    parser.add_argument("--z3-middle-mitm",action=argparse.BooleanOptionalAction,default=True,
                        help="staged factor-4 middle MITM for L %% 8 = 4; bounded, with old Z3 fallback")
    parser.add_argument("--z3-affine-lift",action=argparse.BooleanOptionalAction,default=True,
                        help="staged exact GF(2)/affine binary lift for L %% 8 = 4; otherwise retain Z3")
    parser.add_argument("--z3-mitm-index-entries",type=int,default=250000,
                        help="middle MITM index cap; exceeding it defers to Z3, not UNSAT")
    parser.add_argument("--z3-affine-max-assignments",type=int,default=1048576,
                        help="maximum complete affine space for enumeration; larger spaces use Z3")
    parser.add_argument("--z3-affine-backend",choices=("auto","python","native"),default="auto",
                        help="exact affine scan implementation: auto uses GIL-free C with Python fallback")
    parser.add_argument("--z3-mitm-batch-size",type=int,default=4,
                        help="durably queue 1..16 middle models per index build (default: 4; 1 restores one-at-a-time)")
    parser.add_argument("--z3-slice-seconds",type=float,default=1.0,
                        help="first per-check Z3 budget, doubled on UNKNOWN (default: 1 second)")
    parser.add_argument(
        "--z3-timeout",
        type=float,
        default=DEFAULT_Z3_TIMEOUT_SECONDS,
        help="maximum staged check budget, or one-shot timeout (default: 60 seconds; 0: no ceiling)",
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


def _effective_torch_batch_size(length: int, requested) -> int:
    """Balance lane parallelism against the wider matrices at larger L."""
    if requested is not None:
        if requested <= 0:
            raise ValueError("--torch-batch-size must be positive")
        return requested
    return 2048 if length <= 58 else 1024


def _requested_torch_device_name(args) -> str:
    """Combine the portable device flag with legacy backend shortcuts."""
    shortcut = args.backend if args.backend in ("mps", "cuda") else None
    if shortcut is not None and args.torch_device not in ("auto", shortcut):
        raise ValueError(
            "--backend {} 與 --torch-device {} 衝突".format(
                shortcut, args.torch_device
            )
        )
    return shortcut or args.torch_device


def _resolved_torch_device_name(args) -> str:
    """Resolve and validate the actual accelerator before starting workers."""
    try:
        from solver.torch_device import resolve_torch_device
    except ImportError as error:
        raise RuntimeError("請先在當前 Python 環境安裝 PyTorch。") from error
    return resolve_torch_device(_requested_torch_device_name(args)).type


def _run_torch_adam_backend(args, budget: float) -> int:
    """Retain the former raw-real Adam search as an explicit ablation/resume path."""
    try:
        from solver.torch_search import TorchSearch, TorchSearchConfig
    except ImportError as error:
        raise RuntimeError("請在專案 .venv 安裝 PyTorch：python -m pip install torch") from error
    if args.resume is not None:
        search = TorchSearch.resume(args.resume)
        requested_device = _requested_torch_device_name(args)
        if requested_device != "auto" and search.device.type != requested_device:
            raise ValueError(
                "checkpoint 使用 {}，但本次指定 {}".format(
                    search.device.type, requested_device
                )
            )
        if args.L is not None and args.L != search.config.L:
            raise ValueError("--L 與 checkpoint 長度不符")
        args.L, args.seed = search.config.L, search.config.seed
    else:
        device = _resolved_torch_device_name(args)
        search = TorchSearch(TorchSearchConfig(
            L=args.L, seed=args.seed,
            device=device,
            batch_size=_effective_torch_batch_size(args.L, args.torch_batch_size),
            steps_per_restart=args.torch_steps_per_restart,
            observation_interval=args.torch_observation_interval,
            compression=not args.torch_full_only,
            continuous_kernel=args.torch_kernel,
            loss_mode=args.torch_loss,
            loss_backend=args.torch_loss_backend,
            optimization_mode=args.torch_optimization_mode,
            observation_backend=args.torch_observation_backend,
            fast_observation=args.torch_fast_observation,
            frequency_pruning=args.torch_frequency_pruning,
            symmetry_pruning=args.torch_symmetry_pruning,
            half_shift_seeding=args.torch_half_shift_seeding,
            compression_signature_pairing=args.torch_compression_pairing,
            mathematical_loss=args.torch_mathematical_loss,
        ))
    initial_groups = _group_count(args.L)
    prior_restarts = search.restarts - search.config.batch_size if args.resume is not None else 0
    print("[pipeline] PyTorch 搜尋 | L={} | seed={} | batch={} | device={}".format(
        args.L, args.seed, search.config.batch_size, search.device), flush=True)
    method = "Douglas–Rachford" if search.config.optimization_mode == "douglas_rachford" else "Adam"
    print("壓縮 FKM 初始化 → 批次 {} → 固定奇偶 content 量化 → 獨立驗證 → L.txt".format(method), flush=True)
    print("找到新解後持續搜尋；按 Ctrl+C 儲存並停止。", flush=True)

    progress_lock = threading.RLock()
    def progress(state):
        with progress_lock:
            count = max(0, _group_count(args.L) - initial_groups)
            print("搜尋統計：restart={}, epochs={}, projected evaluations={}, valid results={}".format(
                state.restarts, state.epoch, state.candidate_evaluations, count), flush=True)
            print("elapsed={:.1f}s best_score={} device={}".format(
                state.elapsed, state.best["score"] if state.best else "pending", state.device), flush=True)
            if args.z3:
                print("Z3 觸發使用次數：{}".format(bridge.z3_execution_count), flush=True)

    bridge = _CCompletionBridge(args, args.L, Path("."), result_callback=lambda _r: progress(search))
    try:
        result = search.run(
            seconds=budget, checkpoint_path=args.resume,
            checkpoint_interval=args.checkpoint_interval, progress=progress,
            elite_callback=bridge.consider if bridge.enabled else None,
        )
    finally:
        bridge.close()
    progress(search)
    new_groups = max(0, _group_count(args.L) - initial_groups)
    _print_restarts_per_new_pqcp(search.restarts - prior_restarts, new_groups)
    print("PyTorch 搜尋{}；checkpoint：{}".format(
        "已安全中斷" if result["interrupted"] else "已結束", result["checkpoint"]))
    print("最佳候選：{}".format(search.best_path))
    print("本輪新增序列對：{} 組。".format(new_groups))
    return 0


def _run_torch_backend(args, budget: float) -> int:
    """Use the same unseeded PyTorch refinement on MPS, CUDA, or CPU."""
    if args.resume is not None or args.backend == "adam":
        return _run_torch_adam_backend(args, budget)
    from solver.torch_hybrid_runner import TorchHybridConfig, run_torch_hybrid_search
    device = _resolved_torch_device_name(args)
    root = Path(".")
    initial_groups = _group_count(args.L)
    print("[pipeline] 搜尋開始前既有結果：{} 組".format(initial_groups), flush=True)
    print(
        ("[pipeline] PyTorch {} 連續矩陣→固定 content 量化/GPU polish "
         "∥ CPU C swap completion → independent verifier").format(device.upper()),
        flush=True,
    )
    print("找到新解後持續搜尋；按 Ctrl-C 安全停止。", flush=True)
    bridge = _CCompletionBridge(args, args.L, root)

    def output(line):
        if line.startswith("搜尋統計："):
            line = _with_new_valid_results(line, max(0, _group_count(args.L) - initial_groups))
        print(line, flush=True)
        if args.z3 and line.startswith("搜尋統計："):
            print("Z3 觸發使用次數：{}".format(bridge.z3_execution_count), flush=True)

    try:
        result = run_torch_hybrid_search(TorchHybridConfig(
            L=args.L, seed=args.seed, seconds=budget, threads=args.workers,
            device=device,
            cycle_seconds=args.cpu_seconds_per_cycle,
            formation_seconds=args.mps_seconds_per_cycle,
            continuous_batch_size=_effective_torch_batch_size(
                args.L, args.torch_batch_size
            ),
            continuous_steps_per_restart=args.torch_steps_per_restart,
            continuous_observation_interval=args.torch_observation_interval,
            continuous_loss_mode=args.torch_loss,
            continuous_optimization_mode=args.torch_optimization_mode,
            continuous_loss_backend=args.torch_loss_backend,
            continuous_frequency_pruning=args.torch_frequency_pruning,
            continuous_symmetry_pruning=args.torch_symmetry_pruning,
            continuous_half_shift_seeding=args.torch_half_shift_seeding,
            continuous_compression_pairing=args.torch_compression_pairing,
            continuous_mathematical_loss=args.torch_mathematical_loss,
            observation_backend=args.torch_observation_backend,
            fast_observation=args.torch_fast_observation,
            initial_formation_seconds=args.mps_initial_seconds,
            completion_pair_seed_percent=args.mps_pair_seed_percent,
            adaptive_kick=args.adaptive_kick and not args.legacy_kick,
            c_quench=args.c_quench,
            root=root,
        ), line_callback=output, elite_callback=bridge.consider if bridge.enabled else None)
    finally:
        bridge.close()
    new_groups = max(0, _group_count(args.L) - initial_groups)
    print("[pipeline] {} refresh cycles={} formation={:.3f}s expanded seeds={}".format(
        device.upper(), result.cycles, result.mps_formation_seconds, result.expanded_seeds), flush=True)
    print("[pipeline] {} seed strategy={}".format(device.upper(), result.seed_strategy), flush=True)
    _print_reference_stats(result.restarts, result.moves, result.swap_evaluations, new_groups)
    _print_restarts_per_new_pqcp(result.restarts, new_groups)
    if args.z3:
        print("Z3 觸發使用次數：{}".format(bridge.z3_execution_count), flush=True)
    print("本輪新增序列對：{} 組。".format(new_groups), flush=True)
    print("PyTorch seed bank：{}".format(result.last_seed_bank), flush=True)
    return 0


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
            # One-click runs use the verified fast swap-SA path, with compressed FKM.
            args.reference_port = True
            args.gcp = False
            args.enhanced = False
            raw_z3 = input(
                "是否啟用 Z3 completion？（直接按 Enter = 是，輸入 n = 否）："
            ).strip().lower()
            if raw_z3 not in ("", "y", "yes", "n", "no"):
                raise ValueError("Z3 選擇只能留空或輸入 y/n")
            use_z3 = raw_z3 not in ("n", "no")
            raw_mathematical_loss = input(
                "是否啟用新的 PACF 數學 loss？（直接按 Enter = 是，輸入 n = 否）："
            ).strip().lower()
            if raw_mathematical_loss not in ("", "y", "yes", "n", "no"):
                raise ValueError("數學 loss 選擇只能留空或輸入 y/n")
            args.torch_mathematical_loss = (
                "none" if raw_mathematical_loss in ("n", "no") else "lattice"
            )
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
        if not math.isfinite(args.z3_timeout):
            raise ValueError("--z3-timeout must be finite")
        if not math.isfinite(args.z3_slice_seconds) or args.z3_slice_seconds<=0:
            raise ValueError("--z3-slice-seconds must be positive and finite")
        if args.z3_multilevel_max_models <= 0:
            raise ValueError("--z3-multilevel-max-models must be positive")
        if args.z3_mitm_index_entries <= 0 or args.z3_affine_max_assignments <= 0:
            raise ValueError("MITM index and affine assignment caps must be positive")
        if not 1<=args.z3_mitm_batch_size<=16:
            raise ValueError("--z3-mitm-batch-size must be in 1..16")
        if args.z3_rectangle_restarts <= 0 or not 1 <= args.z3_rectangle_moves <= 2**31-1 or not 1 <= args.z3_rectangle_samples <= 1024:
            raise ValueError("invalid --z3-rectangle-restarts/moves/samples")
        torch_resume = False
        if args.resume is not None:
            with args.resume.open(encoding="utf-8") as handle:
                checkpoint_info = json.load(handle)
            if not isinstance(checkpoint_info, dict):
                raise ValueError("checkpoint must contain a JSON object")
            torch_resume = checkpoint_info.get("format") == "pqcp-torch-search"
        if torch_resume or (args.resume is None and args.backend in ("torch", "mps", "cuda", "adam") and not args.python_backend):
            if args.enhanced or args.gcp or args.resume_parallel:
                raise ValueError("PyTorch 使用批次梯度搜尋；不適用舊 SA 專用參數")
            if torch_resume:
                print("[pipeline] 繼續原 PyTorch/Adam 斷點。", flush=True)
            return _run_torch_backend(args, budget)
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
            print("[pipeline] 顯式 --backend c 回退模式。", flush=True)
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
                        adaptive_kick=args.adaptive_kick and not args.legacy_kick,
                        quench=args.c_quench,
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
