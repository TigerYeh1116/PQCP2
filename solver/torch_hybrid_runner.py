"""Persistent CUDA continuous-matrix formation plus exact C completion.

No L.txt or verified inventory is used to initialize or teach the population.
Only the existing result writer reads L.txt, for exact/A-B-swap deduplication.
"""

from dataclasses import dataclass
import math
import json
from pathlib import Path
import re
import threading
from time import perf_counter
from typing import Callable, List, Optional

from .c_backend import CSearchConfig, run_c_search
from .checkpoint import atomic_write_json
from .compressed_pairing import pair_by_compressed_signature
from .correlation import full_correlation_profile
from .objective import pqcp_objective, pqcp_objective_breakdown
from .search_runner import append_verified_solution_if_new
from .structured_energy import structured_energy_breakdown
from .target_profiles import TargetContentProfile, pair_content
from .torch_hybrid import write_elite_pair_seed_bank
from .torch_population import PopulationSearchConfig, TorchPopulationSearch
from .torch_polish import discrete_energy, polish_stream
from .torch_search import BatchedPQCP, TorchSearch, TorchSearchConfig
from .verifier import verify_pqcp


_C_STATS = re.compile(
    r"^搜尋統計：restart=(\d+), moves=(\d+), "
    r"swap evaluations=(\d+), valid results=(\d+)$"
)
_C_PROGRESS = re.compile(r"^搜尋進度：已測試 (\d+) 組序列對$")


@dataclass(frozen=True)
class TorchHybridConfig:
    """Controls for continuous CUDA formation and compiled completion."""

    L: int
    seed: int = 123
    seconds: float = float("inf")
    cycle_seconds: float = 15.0
    threads: int = 8
    target_seed_count: int = 128
    bootstrap_pair_seed_percent: int = 25
    bootstrap_islands_per_profile: int = 2
    bootstrap_population_size: int = 128
    generations_per_cycle: int = 128
    formation_seconds: float = 45.0
    continuous_batch_size: int = 2048
    continuous_steps_per_restart: int = 2000
    continuous_observation_interval: int = 25
    continuous_archive_size: int = 512
    continuous_loss_mode: str = "balanced"
    continuous_optimization_mode: str = "relaxed"
    continuous_loss_backend: str = "profile"
    continuous_frequency_pruning: bool = True
    continuous_symmetry_pruning: bool = True
    continuous_half_shift_seeding: bool = True
    continuous_compression_pairing: bool = True
    continuous_mathematical_loss: str = "none"
    initial_formation_seconds: Optional[float] = None
    fast_observation: bool = False
    observation_backend: str = "torch"
    completion_pair_seed_percent: int = 100
    polish_elites: int = 128
    polish_steps: int = 250
    polish_candidates: int = 8
    device: str = "cuda"
    root: Path = Path(".")

    def __post_init__(self) -> None:
        if not isinstance(self.L, int) or isinstance(self.L, bool) or self.L < 4 or self.L % 2:
            raise ValueError("hybrid CUDA search requires even L >= 4")
        if math.isnan(self.seconds) or self.seconds <= 0:
            raise ValueError("seconds must be positive")
        if not math.isfinite(self.cycle_seconds) or self.cycle_seconds <= 0:
            raise ValueError("cycle_seconds must be finite and positive")
        if self.threads <= 0 or self.target_seed_count <= 0:
            raise ValueError("thread and seed controls must be positive")
        if not 0 <= self.bootstrap_pair_seed_percent <= 100:
            raise ValueError("bootstrap_pair_seed_percent must be in 0..100")
        if self.bootstrap_islands_per_profile <= 0 or self.bootstrap_population_size <= 0:
            raise ValueError("bootstrap population controls must be positive")
        if self.device not in ("cuda", "cpu"):
            raise ValueError("device must be cuda or cpu")
        if self.continuous_loss_mode not in ("legacy", "balanced", "projected"):
            raise ValueError("continuous_loss_mode must be legacy, balanced or projected")
        if self.continuous_optimization_mode not in ("relaxed", "straight_through", "douglas_rachford"):
            raise ValueError("invalid continuous_optimization_mode")
        if self.continuous_loss_backend not in ("profile", "spectral"):
            raise ValueError("continuous_loss_backend must be profile or spectral")
        if not isinstance(self.continuous_frequency_pruning, bool):
            raise ValueError("continuous_frequency_pruning must be boolean")
        if not isinstance(self.continuous_symmetry_pruning, bool):
            raise ValueError("continuous_symmetry_pruning must be boolean")
        if not isinstance(self.continuous_half_shift_seeding, bool):
            raise ValueError("continuous_half_shift_seeding must be boolean")
        if not isinstance(self.continuous_compression_pairing, bool):
            raise ValueError("continuous_compression_pairing must be boolean")
        if self.continuous_mathematical_loss not in ("none", "psd_cap", "divisor_lift", "lattice", "variance", "combined", "lattice_bootstrap"):
            raise ValueError("invalid continuous_mathematical_loss")
        if not isinstance(self.fast_observation, bool):
            raise ValueError("fast_observation must be boolean")
        if self.observation_backend not in ("torch", "metal"):
            raise ValueError("observation_backend must be torch or metal")
        if self.observation_backend == "metal":
            raise ValueError("metal observation requires MPS and L<=94")
        if (not isinstance(self.completion_pair_seed_percent, int) or
                isinstance(self.completion_pair_seed_percent, bool) or
                not 0 <= self.completion_pair_seed_percent <= 100):
            raise ValueError("completion_pair_seed_percent must be an integer in 0..100")
        if self.initial_formation_seconds is not None and (
                not math.isfinite(self.initial_formation_seconds) or self.initial_formation_seconds <= 0):
            raise ValueError("initial_formation_seconds must be finite and positive")
        if not isinstance(self.generations_per_cycle, int) or self.generations_per_cycle <= 0:
            raise ValueError("generations_per_cycle must be a positive integer")
        if not math.isfinite(self.formation_seconds) or self.formation_seconds <= 0:
            raise ValueError("formation_seconds must be finite and positive")
        for name in (
            "continuous_batch_size", "continuous_steps_per_restart",
            "continuous_observation_interval", "continuous_archive_size",
            "polish_elites", "polish_steps", "polish_candidates",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(name + " must be a positive integer")


@dataclass(frozen=True)
class TorchHybridResult:
    """Aggregate exact counters across periodic MPS refresh cycles."""

    elapsed: float
    interrupted: bool
    cycles: int
    mps_formation_seconds: float
    expanded_seeds: int
    restarts: int
    moves: int
    swap_evaluations: int
    verified_candidates: int
    new_solutions: int
    last_seed_bank: Optional[Path]
    seed_strategy: str


@dataclass(frozen=True)
class PreparedSeedBank:
    """One MPS-produced seed bank and its safe C mixing policy."""

    path: Optional[Path]
    count: int
    elapsed: float
    strategy: str
    pair_seed_percent: int
    verified: int = 0
    new_solutions: int = 0
    generations: int = 0
    best_score: Optional[int] = None
    compression_matches: int = 0


def _new_population(config: TorchHybridConfig) -> TorchPopulationSearch:
    return TorchPopulationSearch(PopulationSearchConfig(
        config.L,
        seed=config.seed,
        device=config.device,
        islands_per_profile=config.bootstrap_islands_per_profile,
        population_size=config.bootstrap_population_size,
        elite_count=min(16, config.bootstrap_population_size),
    ))


def _new_continuous_search(config: TorchHybridConfig) -> TorchSearch:
    """Create the production CUDA stage without consulting known solutions."""
    return TorchSearch(TorchSearchConfig(
        config.L,
        seed=config.seed,
        device=config.device,
        batch_size=config.continuous_batch_size,
        steps_per_restart=config.continuous_steps_per_restart,
        observation_interval=config.continuous_observation_interval,
        stagnation_steps=max(
            config.continuous_observation_interval * 8,
            config.continuous_steps_per_restart // 2,
        ),
        archive_size=config.continuous_archive_size,
        elite_count=min(16, config.continuous_archive_size),
        continuous_kernel="dft",
        optimization_mode=config.continuous_optimization_mode,
        loss_mode=config.continuous_loss_mode,
        loss_backend=config.continuous_loss_backend,
        frequency_pruning=config.continuous_frequency_pruning,
        symmetry_pruning=config.continuous_symmetry_pruning,
        half_shift_seeding=config.continuous_half_shift_seeding,
        compression_signature_pairing=config.continuous_compression_pairing,
        mathematical_loss=config.continuous_mathematical_loss,
        fast_observation=config.fast_observation,
        observation_backend=config.observation_backend,
    ), Path(config.root))


def _compression_signature_records(
    search: TorchSearch, candidates: List[dict], limit: int,
) -> List[dict]:
    """Recombine archive sides only when factor-2/4 PACFs match exactly."""
    if not search.config.compression_signature_pairing or not candidates:
        return []
    pools = {}
    for record in candidates:
        ae, ao, be, bo = pair_content(record["A"], record["B"])
        pools.setdefault((ae, ao), []).append(record["A"])
        pools.setdefault((be, bo), []).append(record["B"])

    output = []
    # An individual sequence's PACF does not remember whether it originally
    # occupied side A or B, nor which target another archived side used.
    # Re-indexing by exact individual content safely enables cross-lane and
    # cross-target recombination without changing either sequence.
    for target in search.available:
        if len(output) >= limit:
            break
        a_pool = pools.get((target.a_even_ones, target.a_odd_ones), ())
        b_pool = pools.get((target.b_even_ones, target.b_odd_ones), ())
        if not a_pool or not b_pool:
            continue
        matches = pair_by_compressed_signature(
            a_pool, b_pool, target,
            limit=limit - len(output),
            general_psd_pruning=getattr(search.config, "frequency_pruning", False),
        )
        for match_index, match in enumerate(matches):
            exact = tuple(full_correlation_profile(match.a, match.b))
            score = pqcp_objective(exact)
            check = verify_pqcp(match.a, match.b)
            if tuple(check.profile) != exact:
                raise RuntimeError(
                    "compression pairing disagrees with independent verifier"
                )
            weights = {
                factor: weight for factor, weight in ((2, 2), (4, 4))
                if factor in match.factors
            }
            target_energy = structured_energy_breakdown(
                exact, target.k, target.eta, match.factors
            ).weighted_total(weights)
            payload = {
                "L": search.config.L,
                "A": list(match.a),
                "B": list(match.b),
                "profile": list(exact),
                "score": score,
                "iteration": search.epoch,
                "restart": search.generation,
                "k": target.k,
                "sign": target.eta,
                "target_energy": target_energy,
                "verified": check.is_valid,
                "method": "exact_compressed_signature_pairing",
                "compression_factors": list(match.factors),
                "compression_match_index": match_index,
                "objective_components": pqcp_objective_breakdown(exact),
            }
            if check.is_valid:
                search.verified_hits += 1
                is_new = append_verified_solution_if_new(
                    search.config.L, match.a, match.b, search.root
                )
                if is_new:
                    search.new_solutions += 1
                    path = search.root / "results/verified" / (
                        "L{}_compressed_seed{}_epoch{}_match{}.json".format(
                            search.config.L, search.config.seed,
                            search.epoch, match_index,
                        )
                    )
                    suffix = 0
                    original = path
                    while path.exists():
                        suffix += 1
                        path = original.with_name(
                            original.stem + "_{}".format(suffix) + original.suffix
                        )
                    atomic_write_json(path, payload)
                    print(
                        "找到新的 ({},4)-PQCP，已寫入 {}.txt".format(
                            search.config.L, search.config.L
                        ),
                        flush=True,
                    )
            else:
                output.append(payload)
    return output


def _continuous_seed_records(search: TorchSearch, limit: int) -> List[dict]:
    """Return exact-signature recombinations and close original MPS elites."""
    candidates = [payload for payload in search.archive.values()
                  if not payload["verified"] and payload["score"] > 0]
    candidates.extend(_compression_signature_records(search, candidates, limit))
    deduplicated = {}
    for payload in candidates:
        key = (tuple(payload["A"]), tuple(payload["B"]),
               int(payload["k"]), int(payload["sign"]))
        previous = deduplicated.get(key)
        if previous is None or (
            payload["target_energy"], payload["score"]
        ) < (previous["target_energy"], previous["score"]):
            deduplicated[key] = payload
    candidates = list(deduplicated.values())
    candidates.sort(key=lambda payload: (
        payload["target_energy"], payload["score"], payload["iteration"],
    ))
    if not candidates:
        return []
    minimum = min(32, limit, len(candidates))
    cutoff = max(
        candidates[0]["target_energy"] + search.config.L,
        candidates[minimum - 1]["target_energy"],
    )
    return [item for item in candidates if item["target_energy"] <= cutoff][:limit]


def _polish_continuous_records(
    config: TorchHybridConfig, search: TorchSearch, records: List[dict], cycle: int,
) -> tuple:
    """Run the existing exact same-parity MPS annealer after quantization."""
    if not records:
        return [], 0, 0
    import torch

    selected = records[:min(config.polish_elites, len(records))]
    targets = tuple(TargetContentProfile(
        config.L, int(record["k"]), int(record["sign"]),
        *pair_content(record["A"], record["B"]),
    ) for record in selected)
    model = BatchedPQCP(targets, search.device)
    bits = torch.tensor(
        [[record["A"], record["B"]] for record in selected],
        dtype=torch.float32, device=search.device,
    )
    polished = polish_stream(
        model, 1 - 2 * bits,
        steps=config.polish_steps,
        candidates=config.polish_candidates,
        seed=config.seed + (cycle + 1) * 8_000_003,
        block_steps=config.polish_steps,
        proposal_policy="opposite",
        kick_interval=min(250, config.polish_steps),
        compression=True,
    )
    output = list(records)
    verified = new = 0
    polished_bits = ((1 - polished.signs) / 2).to(torch.int32).cpu().tolist()
    gpu_profiles = polished.profile.to(torch.int32).cpu().tolist()
    gpu_scores = polished.scores.to(torch.int64).cpu().tolist()
    gpu_energies = discrete_energy(
        polished.profile, model.targets, compression=True
    ).to(torch.int64).cpu().tolist()
    for lane, ((a, b), profile, score, energy) in enumerate(zip(
            polished_bits, gpu_profiles, gpu_scores, gpu_energies)):
        exact = full_correlation_profile(a, b)
        if exact != profile or pqcp_objective(exact) != score:
            raise RuntimeError("GPU polish disagrees with full correlation recomputation")
        check = verify_pqcp(a, b)
        if score == 0 and not check.is_valid:
            raise RuntimeError("GPU polish score zero failed independent verifier")
        factors = tuple(factor for factor in (2, 4) if config.L % factor == 0)
        weights = {
            factor: weight for factor, weight in ((2, 2), (4, 4))
            if factor in factors
        }
        exact_energy = structured_energy_breakdown(
            exact, targets[lane].k, targets[lane].eta, factors
        ).weighted_total(weights)
        if exact_energy != energy:
            raise RuntimeError("GPU polish target energy failed exact recomputation")
        if check.is_valid:
            verified += 1
            new += append_verified_solution_if_new(config.L, a, b, Path(config.root))
            continue
        output.append({
            "L": config.L, "A": list(a), "B": list(b),
            "profile": list(exact), "score": int(score),
            "iteration": search.epoch, "k": targets[lane].k,
            "sign": targets[lane].eta, "target_energy": int(exact_energy),
            "verified": False, "method": "torch_mps_polish",
        })
    deduplicated = {}
    for record in output:
        key = tuple(sorted((tuple(record["A"]), tuple(record["B"]))))
        previous = deduplicated.get(key)
        if previous is None or (record["target_energy"], record["score"]) < (
            previous["target_energy"], previous["score"]
        ):
            deduplicated[key] = record
    ranked = sorted(deduplicated.values(), key=lambda record: (
        record["target_energy"], record["score"], record.get("iteration", 0),
    ))
    minimum = min(32, config.target_seed_count, len(ranked))
    cutoff = max(
        ranked[0]["target_energy"] + config.L,
        ranked[minimum - 1]["target_energy"],
    )
    return ([record for record in ranked if record["target_energy"] <= cutoff][
        :config.target_seed_count
    ], verified, new)


def _persist_continuous_best(
    config: TorchHybridConfig, records: List[dict], cycle: int,
) -> Optional[int]:
    """Atomically retain the best post-quantization/post-polish candidate."""
    if not records:
        return None
    best = min(records, key=lambda record: (record["score"], record["target_energy"]))
    check = verify_pqcp(best["A"], best["B"])
    exact = list(check.profile)
    if exact != best["profile"] or pqcp_objective(exact) != best["score"]:
        raise RuntimeError("continuous best failed independent recomputation")
    nonzero = [u for u in range(1, config.L) if exact[u] != 0]
    payload = {
        **best,
        "cycle": cycle,
        "verified": check.is_valid,
        "nonzero_shifts": nonzero,
        "nonzero_values": [exact[u] for u in nonzero],
        "objective_components": pqcp_objective_breakdown(exact),
    }
    path = Path(config.root) / "results" / "best" / (
        "L{}_mps_continuous_seed{}.json".format(config.L, config.seed)
    )
    replace = True
    previous_score = None
    if path.exists():
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
            previous_score = int(previous["score"])
            replace = (payload["score"], payload["target_energy"]) < (
                previous["score"], previous.get("target_energy", 10 ** 30)
            )
        except (OSError, ValueError, KeyError, TypeError):
            replace = True
    if replace:
        atomic_write_json(path, payload)
        return int(payload["score"])
    return previous_score


def _population_seed_records(search: TorchPopulationSearch, limit: int) -> List[dict]:
    """Export energy-ranked elites round-robin across all target islands."""
    import torch

    batch = search.elites
    if batch is None:
        return []
    bits = batch.bits.to(torch.int32).cpu().tolist()
    profiles = batch.profiles.to(torch.int32).cpu().tolist()
    scores = batch.scores.to(torch.int32).cpu().tolist()
    records = []
    for member in range(search.config.elite_count):
        for island, target in enumerate(search.profiles):
            if scores[island][member] == 0:
                continue
            records.append({
                "A": bits[island][member][0], "B": bits[island][member][1],
                "profile": profiles[island][member], "score": scores[island][member],
                "iteration": search.generation, "k": target.k, "sign": target.eta,
            })
            if len(records) >= limit:
                return records
    return records


def _persist_population_best(config: TorchHybridConfig, search: TorchPopulationSearch,
                             cycle: int, elapsed: float, old_score: Optional[int]) -> None:
    """Write a fully recomputed best immediately, before the next generation."""
    best = dict(search.best)
    best.pop("profile_case", None)
    check = verify_pqcp(best["A"], best["B"])
    exact = list(check.profile)
    if exact != best["profile"] or pqcp_objective(exact) != best["score"]:
        raise RuntimeError("GPU best failed full correlation recomputation")
    record = {
        **best, "L": config.L, "seed": config.seed, "cycle": cycle,
        "method": "unseeded-refinement", "verified": check.is_valid,
        "old_best_score": old_score, "new_best_score": best["score"],
        "formation_elapsed": elapsed, "objective_components": pqcp_objective_breakdown(exact),
    }
    root = Path(config.root)
    atomic_write_json(root / "results" / "best" / "L{}_mps_unseeded_seed{}.json".format(config.L, config.seed), record)
    history = root / "logs" / "mps_unseeded_L{}_seed{}.jsonl".format(config.L, config.seed)
    history.parent.mkdir(parents=True, exist_ok=True)
    with history.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def prepare_mps_seed_bank(config: TorchHybridConfig, cycle: int = 0, *,
                          search: Optional[TorchPopulationSearch] = None,
                          seconds: Optional[float] = None) -> PreparedSeedBank:
    """Advance a persistent unseeded population, immediately saving each zero.

    Every new zero is fully recomputed and independently verified before the
    existing atomic/deduplicating writer appends it. Solutions are excluded
    from seed export; the population excludes them from parent selection and
    probability learning. Scores do not replace verification.
    """
    import torch

    root = Path(config.root)
    started = perf_counter()
    budget = config.formation_seconds if seconds is None else min(seconds, config.formation_seconds)
    search = _new_population(config) if search is None else search
    verified = new = 0
    seen = set()
    for _ in range(config.generations_per_cycle):
        if perf_counter() - started >= budget:
            break
        old_score = search.best["score"] if search.best else None
        batch = search.step()
        for row, col in (batch.scores == 0).nonzero().cpu().tolist():
            a, b = (tuple(bits) for bits in batch.bits[row, col].to(torch.int32).cpu().tolist())
            key = tuple(sorted((a, b)))
            if key in seen:
                continue
            check = verify_pqcp(a, b)
            if not check.is_valid or list(check.profile) != batch.profiles[row, col].cpu().tolist():
                raise RuntimeError("GPU zero failed independent exact verifier")
            new += append_verified_solution_if_new(config.L, a, b, root)
            verified += 1
            seen.add(key)
        if search.best is not None and search.best["score"] != old_score:
            _persist_population_best(config, search, cycle, perf_counter() - started, old_score)
    expanded = _population_seed_records(search, config.target_seed_count)
    destination = root / "results" / "mps_seeds" / (
        "L{}_seed{}_current.txt".format(config.L, config.seed)
    )
    records = (write_elite_pair_seed_bank(expanded, destination, length=config.L, retarget=False)
               if expanded else ())
    return PreparedSeedBank(
        destination if records else None, len(records), perf_counter() - started,
        "unseeded-refinement", config.bootstrap_pair_seed_percent,
        verified, new, search.generation,
        search.best["score"] if search.best else None,
    )


def prepare_continuous_seed_bank(
    config: TorchHybridConfig,
    cycle: int = 0,
    *,
    search: Optional[TorchSearch] = None,
    seconds: Optional[float] = None,
    elite_callback=None,
    stop_event=None,
) -> PreparedSeedBank:
    """Advance many continuous matrices on CUDA, then export close discrete lanes.

    Adam operates only on relaxed matrices.  At observation boundaries every
    lane is rank-projected to an exact fixed-content binary pair.  Export is
    ranked by exact integer distance to that lane's assigned ``(k, eta)``
    target, and the existing C annealer receives only this bounded elite bank.
    Neither the continuous loss nor accelerator arithmetic can publish a
    solution: score-zero pairs are recomputed and independently verified by
    :class:`TorchSearch` before the normal deduplicating writer is called.
    """
    started = perf_counter()
    budget = config.formation_seconds if seconds is None else min(
        seconds, config.formation_seconds
    )
    search = _new_continuous_search(config) if search is None else search
    initial_verified = search.verified_hits
    initial_new = search.new_solutions
    initial_epoch = search.epoch
    initial_elapsed = search.elapsed
    try:
        if search.last_observed_epoch != search.epoch:
            search.observe()
        while (perf_counter() - started < budget
               and (stop_event is None or not stop_event.is_set())):
            stale = all(
                search.epoch - last >= search.config.stagnation_steps
                for last in search.last_improved
            )
            if search.round_step >= search.config.steps_per_restart or stale:
                search.generation += 1
                search._initialize_batch()
                search.observe()
            search.step()
            search.elapsed = initial_elapsed + perf_counter() - started
            if search.epoch % search.config.observation_interval == 0:
                search.observe()
    finally:
        search.elapsed = initial_elapsed + perf_counter() - started
        if search.last_observed_epoch != search.epoch:
            search.observe(update_stagnation=False)

    elites = _continuous_seed_records(search, config.target_seed_count)
    elites, polish_verified, polish_new = _polish_continuous_records(
        config, search, elites, cycle
    )
    polished_best_score = _persist_continuous_best(config, elites, cycle)
    if elite_callback is not None:
        for elite in elites:
            elite_callback(
                tuple(elite["A"]), tuple(elite["B"]),
                {"k": elite["k"], "sign": elite["sign"], "score": elite["score"]},
            )
    destination = Path(config.root) / "results" / "mps_seeds" / (
        "L{}_seed{}_continuous_cycle{:06d}.txt".format(
            config.L, config.seed, cycle
        )
    )
    records = (
        write_elite_pair_seed_bank(
            elites, destination, length=config.L, retarget=False
        )
        if elites else ()
    )
    return PreparedSeedBank(
        destination if records else None,
        len(records),
        perf_counter() - started,
        "continuous-matrix-dft+gpu-polish",
        config.completion_pair_seed_percent,
        search.verified_hits - initial_verified + polish_verified,
        search.new_solutions - initial_new + polish_new,
        search.epoch - initial_epoch,
        min(
            score for score in (
                search.best["score"] if search.best else None,
                polished_best_score,
            ) if score is not None
        ),
        sum(
            record.get("method") == "exact_compressed_signature_pairing"
            for record in elites
        ),
    )


def run_torch_hybrid_search(
    config: TorchHybridConfig,
    *,
    line_callback: Optional[Callable[[str], None]] = None,
    elite_callback=None,
) -> TorchHybridResult:
    """Run persistent CUDA formation and compiled C annealing concurrently."""
    started = perf_counter()
    totals = {"formation": 0.0, "expanded": 0, "restarts": 0, "moves": 0,
              "swaps": 0, "verified": 0, "new": 0}
    state_lock = threading.RLock()
    output_lock = threading.RLock()
    stop_event = threading.Event()
    latest = {"bank": None, "pair_seed_percent": 100}
    worker_errors = []
    cpu_cycles = [0]
    mps_cycles = 0
    interrupted = False
    seed_strategy = "pending"

    def emit(line: str) -> None:
        if line_callback is not None:
            with output_lock:
                line_callback(line)

    def cpu_worker() -> None:
        while not stop_event.is_set():
            remaining = config.seconds - (perf_counter() - started)
            if remaining <= 0:
                break
            with state_lock:
                bank = latest["bank"]
                pair_percent = latest["pair_seed_percent"]
                base = {name: totals[name] for name in ("restarts", "moves", "swaps", "new")}

            def cumulative_output(line: str) -> None:
                """Report the active C process on top of completed counters."""
                match = _C_STATS.match(line)
                if match:
                    values = tuple(int(value) for value in match.groups())
                    line = (
                        "搜尋統計：restart={}, moves={}, swap evaluations={}, "
                        "valid results={}".format(
                            base["restarts"] + values[0], base["moves"] + values[1],
                            base["swaps"] + values[2], base["new"] + values[3],
                        )
                    )
                else:
                    progress = _C_PROGRESS.match(line)
                    if progress:
                        line = "搜尋進度：已測試 {} 組序列對".format(
                            base["restarts"] + int(progress.group(1))
                        )
                emit(line)

            try:
                result = run_c_search(CSearchConfig(
                    L=config.L, seed=config.seed + cpu_cycles[0] * 1_000_003,
                    seconds=min(config.cycle_seconds, remaining),
                    # Four C workers left enough host capacity to feed MPS and
                    # matched eight-worker aggregate C throughput on the
                    # target Mac; explicit smaller values remain respected.
                    threads=min(config.threads, 4),
                    root=config.root, echo=False, pair_seed_path=bank,
                    pair_seed_percent=pair_percent,
                    pair_seed_swaps=0 if bank is not None else None,
                    quarter_frequency_pruning=config.continuous_frequency_pruning,
                    half_shift_seeding=config.continuous_half_shift_seeding,
                    emit_elites=elite_callback is not None,
                ), line_callback=cumulative_output, elite_callback=elite_callback,
                   stop_event=stop_event)
            except BaseException as error:
                with state_lock:
                    worker_errors.append(error)
                stop_event.set()
                break
            with state_lock:
                totals["restarts"] += result.restarts
                totals["moves"] += result.moves
                totals["swaps"] += result.swap_evaluations
                totals["verified"] += result.verified_candidates
                totals["new"] += result.new_solutions
                cpu_cycles[0] += 1

    worker = threading.Thread(target=cpu_worker, name="pqcp-c-completion")
    worker.start()
    try:
        # Accelerator initialization can fail too; never orphan C.
        search = _new_continuous_search(config)
        while not stop_event.is_set():
            remaining = config.seconds - (perf_counter() - started)
            if remaining <= 0:
                break
            if mps_cycles == 0 and config.initial_formation_seconds is not None:
                remaining = min(remaining, config.initial_formation_seconds)
            prepared = prepare_continuous_seed_bank(
                config, mps_cycles, search=search, seconds=remaining,
                elite_callback=elite_callback, stop_event=stop_event,
            )
            with state_lock:
                if prepared.path is not None:
                    latest["bank"] = prepared.path
                    latest["pair_seed_percent"] = prepared.pair_seed_percent
                totals["formation"] += prepared.elapsed
                totals["expanded"] += prepared.count
                totals["verified"] += prepared.verified
                totals["new"] += prepared.new_solutions
                error = worker_errors[0] if worker_errors else None
            mps_cycles += 1
            seed_strategy = prepared.strategy
            emit(
                "[CUDA] cycle={} strategy={} matrices={} seeds={} formation={:.3f}s "
                "C=concurrent epochs={} best_score={} compressed_pairs={} new={}".format(
                    mps_cycles, prepared.strategy, config.continuous_batch_size,
                    prepared.count, prepared.elapsed, prepared.generations,
                    prepared.best_score, prepared.compression_matches,
                    prepared.new_solutions,
                )
            )
            if error is not None:
                break
    except KeyboardInterrupt:
        interrupted = True
    finally:
        stop_event.set()
        worker.join(timeout=35.0)
    if worker.is_alive():
        raise RuntimeError("C completion worker did not stop gracefully")
    if worker_errors:
        raise worker_errors[0]
    with state_lock:
        snapshot = dict(totals)
        last_bank = latest["bank"]
    return TorchHybridResult(
        elapsed=perf_counter() - started, interrupted=interrupted, cycles=mps_cycles,
        mps_formation_seconds=snapshot["formation"], expanded_seeds=snapshot["expanded"],
        restarts=snapshot["restarts"], moves=snapshot["moves"],
        swap_evaluations=snapshot["swaps"], verified_candidates=snapshot["verified"],
        new_solutions=snapshot["new"], last_seed_bank=last_bank,
        seed_strategy=seed_strategy,
    )


__all__ = (
    "PreparedSeedBank", "TorchHybridConfig", "TorchHybridResult",
    "prepare_continuous_seed_bank", "prepare_mps_seed_bank",
    "run_torch_hybrid_search",
)
