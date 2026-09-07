"""Executable orchestration for the existing Project 2 PQCP components.

This module selects the FKM-initialized fixed-weight swap runner, records
independently recomputed best candidates, and optionally passes recorded SA
elites through guidance, deterministic beam repair, and a positive-radius Z3
fallback for still-unsolved but improved endpoints.
"""

from dataclasses import dataclass
import json
from pathlib import Path
import random
from time import perf_counter
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .checkpoint import CheckpointError, SearchParameters, SearchState, atomic_write_json, load_checkpoint
from .beam_repair import beam_repair
from .correlation import full_correlation_profile
from .enhanced_search import EnhancedParameters, EnhancedSearch, load_enhanced_checkpoint, save_enhanced_checkpoint
from .golay import complete_one_flip_each, project_length_golay_seed
from .objective import pqcp_objective, pqcp_objective_breakdown
from .reference_search import reference_search_parameters
from .search_runner import (
    DEFAULT_GUIDED_OBJECTIVE_ENERGY_WEIGHT,
    DEFAULT_GUIDED_PROPOSAL_SAMPLES,
    SearchRunner,
    append_verified_solution_if_new,
)
from .verifier import verify_pqcp
from .weight_constraints import admissible_weight_pairs, canonical_weight_pairs
from .target_profiles import TargetContentProfile, canonical_target_content_profiles, pair_content
from .z3_guidance import correlation_hamming_lower_bound, guided_completion

try:  # Keep non-Z3 SA runs usable when the optional package is unavailable.
    from .hybrid import PortfolioConfig, SAElite, run_hybrid_portfolio
except ImportError:  # pragma: no cover - exercised only on environments without Z3
    PortfolioConfig = SAElite = run_hybrid_portfolio = None


PROJECT_LENGTHS = frozenset((44, 46, 58, 68, 86, 90, 94))
DEFAULT_Z3_TIMEOUT_SECONDS = 60.0
DEFAULT_Z3_TRIGGER_SCORE = 16
DEFAULT_Z3_GUIDANCE_TOP_K = 10
DEFAULT_Z3_PROFILE_RADIUS = 4
DEFAULT_REPAIR_BEAM_WIDTH = 30
DEFAULT_REPAIR_MAX_DEPTH = 4
DEFAULT_Z3_COMPLETION_RADIUS = 4
# Exact/near Golay profiles are sharp fixed-weight local minima.  A temporary
# high initial temperature lets the existing SA leave that centre; ordinary
# FKM restarts still use SearchParameters.initial_temperature == 8.0.
GCP_INITIAL_TEMPERATURE = 64.0


@dataclass(frozen=True)
class PipelineConfig:
    """Runtime choices only; all search settings retain their existing defaults."""

    L: Optional[int] = None
    seed: int = 123
    seconds: float = 60.0
    enhanced: bool = False
    z3: bool = False
    repair: bool = False
    z3_timeout: float = DEFAULT_Z3_TIMEOUT_SECONDS
    z3_trigger_score: int = DEFAULT_Z3_TRIGGER_SCORE
    z3_guidance_top_k: int = DEFAULT_Z3_GUIDANCE_TOP_K
    z3_profile_radius: int = DEFAULT_Z3_PROFILE_RADIUS
    z3_guided: bool = True
    repair_beam_width: int = DEFAULT_REPAIR_BEAM_WIDTH
    repair_max_depth: int = DEFAULT_REPAIR_MAX_DEPTH
    z3_completion_radius: int = DEFAULT_Z3_COMPLETION_RADIUS
    fkm_seed_policy: str = "legacy"
    fkm_candidate_count: int = 16
    fkm_elite_count: int = 4
    gcp: bool = False
    reference_port: bool = False
    target_profile_offset: int = 0
    resume: Optional[Path] = None
    checkpoint_interval: float = 60.0
    progress_interval: float = 60.0
    verbose: bool = False
    root: Path = Path(".")


@dataclass(frozen=True)
class PipelineResult:
    """Final immutable summary of one pipeline invocation."""

    L: int
    method: str
    state: object
    elapsed: float
    checkpoint_path: Path
    best_path: Path
    log_path: Path
    interrupted: bool
    verification: object
    verified_paths: Tuple[Path, ...]
    z3_result: Optional[object]
    z3_trigger_count: int


def validate_length(length: int) -> bool:
    """Validate a positive integer and return whether it is a listed Project 2 length."""
    if not isinstance(length, int) or isinstance(length, bool) or length <= 0:
        raise ValueError("L must be a positive integer")
    return length in PROJECT_LENGTHS


def run_pipeline(
    config: PipelineConfig,
    progress: Optional[Callable[[object, float], None]] = None,
    z3_callback: Optional[Callable[[object, int], None]] = None,
) -> PipelineResult:
    """Run existing SA, persist all best candidates, then optionally run bounded Z3.

    A resume file determines both L and search mode; no FKM initialization is
    performed in that case.  Ctrl+C is caught so current state, rich best
    snapshot, and checkpoint are retained before returning.
    """
    if config.seconds < 0:
        raise ValueError("search time must be non-negative")
    if config.z3_timeout < 0:
        raise ValueError("z3 timeout must be non-negative")
    if not isinstance(config.z3_trigger_score, int) or isinstance(config.z3_trigger_score, bool) or config.z3_trigger_score < 0:
        raise ValueError("z3 trigger score must be a non-negative integer")
    if not isinstance(config.z3_guidance_top_k, int) or isinstance(config.z3_guidance_top_k, bool) or config.z3_guidance_top_k <= 0:
        raise ValueError("z3 guidance top-k must be a positive integer")
    if not isinstance(config.z3_profile_radius, int) or isinstance(config.z3_profile_radius, bool) or config.z3_profile_radius < 0:
        raise ValueError("z3 profile radius must be a non-negative integer")
    if not isinstance(config.repair_beam_width, int) or isinstance(config.repair_beam_width, bool) or config.repair_beam_width <= 0:
        raise ValueError("repair beam width must be a positive integer")
    if not isinstance(config.repair_max_depth, int) or isinstance(config.repair_max_depth, bool) or config.repair_max_depth < 0:
        raise ValueError("repair max depth must be a non-negative integer")
    if not isinstance(config.z3_completion_radius, int) or isinstance(config.z3_completion_radius, bool) or config.z3_completion_radius <= 0:
        raise ValueError("z3 completion radius must be a positive integer")
    if config.fkm_seed_policy not in (
        "legacy", "phase_random", "phase_top_q", "compressed_top_q",
        "compressed_a_random_b_top_q",
    ):
        raise ValueError("unsupported FKM seed policy")
    if (
        not isinstance(config.fkm_candidate_count, int)
        or isinstance(config.fkm_candidate_count, bool)
        or config.fkm_candidate_count <= 0
    ):
        raise ValueError("FKM candidate count must be a positive integer")
    if (
        not isinstance(config.fkm_elite_count, int)
        or isinstance(config.fkm_elite_count, bool)
        or config.fkm_elite_count <= 0
    ):
        raise ValueError("FKM elite count must be a positive integer")
    if (
        config.fkm_seed_policy in (
            "phase_top_q", "compressed_top_q",
            "compressed_a_random_b_top_q",
        )
        and config.fkm_elite_count > config.fkm_candidate_count
    ):
        raise ValueError("FKM elite count must not exceed candidate count")
    if config.reference_port and (config.gcp or config.enhanced):
        raise ValueError("reference_port cannot be combined with GCP or enhanced mode")
    root = Path(config.root)
    mode, state = _load_or_initialize(config)
    length = state.L
    is_project_length = validate_length(length)
    if config.L is not None and config.resume is not None and config.L != length:
        raise ValueError("--L does not match the resumed checkpoint length")
    method = mode
    checkpoint_path = Path(config.resume) if config.resume is not None else root / "checkpoints" / _checkpoint_name(length, method)
    best_path = root / "results" / "best" / "L{}_{}_seed{}.json".format(length, method, state.seed)
    log_path = root / "logs" / "pipeline_L{}_{}_seed{}.jsonl".format(length, method, state.seed)
    observer = _BestObserver(root, method, state.seed, best_path, log_path)
    initial_event = observer.record(state, elapsed=getattr(state, "elapsed_seconds", 0.0), old_best_score=None)
    z3_results: List[object] = []
    last_z3_score: Optional[int] = None

    def trigger_z3(event: Dict[str, object]) -> bool:
        """Run bounded completion and report whether SA should change basin."""
        nonlocal last_z3_score
        score = int(event["score"])
        # One fixed threshold is used for the entire run.  It is not tightened
        # according to L or according to how many times Z3 has already run.
        if (not (config.repair or config.z3) or event["verified"] or score > config.z3_trigger_score
                or (last_z3_score is not None and score >= last_z3_score)):
            return False
        profile = event.get("profile")
        if profile is None or len(profile) < 3:
            return False
        if config.z3_guided and correlation_hamming_lower_bound(profile) > config.z3_profile_radius:
            return False
        last_z3_score = score
        encoded_profiles = event.get("target_content_profiles")
        if encoded_profiles:
            result = _run_optional_z3(
                config, length, observer.elites[-1:], root, encoded_profiles,
            )
        else:
            result = _run_optional_z3(config, length, observer.elites[-1:], root)
        z3_results.append(result)
        if z3_callback is not None:
            z3_callback(result, event["score"])
        return bool(
            isinstance(result, dict)
            and result.get("z3_executed", False)
            and not result.get("solved", False)
        )

    restart_after_initial_completion = trigger_z3(initial_event)

    verified_paths: List[Path] = []
    def record_verified(a: Sequence[int], b: Sequence[int], source_state: object) -> None:
        verification = verify_pqcp(a, b)
        if not verification.is_valid:
            raise RuntimeError("score-zero candidate failed independent verification")
        append_verified_solution_if_new(length, tuple(a), tuple(b), root)
        verified_paths.append(_save_verified_result(root, method, source_state, a, b, verification.profile))

    if mode in ("baseline", "reference"):
        runner = SearchRunner(state)
        if restart_after_initial_completion:
            runner.restart_after_completion_miss()

        def baseline_event(outcome, current):
            should_restart = _record_baseline_event(
                observer, outcome, current, trigger_z3
            )
            if should_restart and not current.finished:
                runner.restart_after_completion_miss()

        summary = runner.run(
            seconds=config.seconds,
            checkpoint_path=checkpoint_path,
            # The observer writes a richer atomic best snapshot than the
            # legacy runner payload, including every global-best event.
            best_path=None,
            checkpoint_interval=config.checkpoint_interval,
            progress_interval=config.progress_interval,
            progress_callback=progress,
            on_verified_solution=record_verified,
            event_callback=baseline_event,
        )
        final_state, interrupted, elapsed = summary.state, summary.interrupted, summary.state.elapsed_seconds
    else:
        enhanced_search = EnhancedSearch(state)
        if restart_after_initial_completion:
            enhanced_search._restart()
        final_state, interrupted, elapsed = _run_enhanced(
            enhanced_search, config, checkpoint_path, observer, progress, record_verified, trigger_z3
        )

    observer.record(final_state, getattr(final_state, "elapsed_seconds", elapsed), old_best_score=None)
    verification = verify_pqcp(final_state.best_a, final_state.best_b)
    z3_result = _select_z3_result(z3_results)
    return PipelineResult(
        length, method, final_state, elapsed, checkpoint_path, best_path, log_path,
        interrupted, verification, tuple(verified_paths), z3_result, len(z3_results),
    )


class _BestObserver:
    """Observation-only atomic best snapshot and append-only event history writer."""

    def __init__(self, root: Path, method: str, seed: int, best_path: Path, log_path: Path) -> None:
        self.root, self.method, self.seed = root, method, seed
        self.best_path, self.log_path = best_path, log_path
        self.elites: List[object] = []

    def record(self, state: object, elapsed: float, old_best_score: Optional[int]) -> Dict[str, object]:
        a, b = tuple(state.best_a), tuple(state.best_b)
        profile = full_correlation_profile(a, b)
        score = pqcp_objective(profile)
        if score != state.best_score:
            raise RuntimeError("best score disagrees with full correlation recomputation")
        verification = verify_pqcp(a, b)
        if tuple(profile) != verification.profile:
            raise RuntimeError("verifier profile disagrees with full correlation recomputation")
        if score == 0 and not verification.is_valid:
            raise RuntimeError("score-zero candidate failed independent verification")
        nonzero = [shift for shift in range(1, state.L) if profile[shift] != 0]
        event = {
            "L": state.L, "method": self.method, "seed": self.seed,
            "iteration": state.iteration, "restart_index": state.restart_index,
            "elapsed": elapsed, "old_best_score": old_best_score,
            "new_best_score": score, "score": score,
            "A": list(a), "B": list(b), "profile": profile,
            "nonzero_shifts": nonzero, "nonzero_values": [profile[shift] for shift in nonzero],
            "objective_components": pqcp_objective_breakdown(profile),
            "verified": verification.is_valid,
        }
        parameters = getattr(state, "algorithm_parameters", None)
        encoded_profiles = getattr(parameters, "target_content_profiles", None)
        if getattr(parameters, "preserve_alternating_content", False) and encoded_profiles:
            content = pair_content(a, b)
            event["target_content_profiles"] = [
                list(values) for values in encoded_profiles if tuple(values[2:]) == content
            ]
        atomic_write_json(self.best_path, event)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
            handle.flush()
        if SAElite is not None:
            elite = SAElite(a, b, score, self.seed, len(self.elites))
            if not self.elites or (elite.a, elite.b, elite.score) != (self.elites[-1].a, self.elites[-1].b, self.elites[-1].score):
                self.elites.append(elite)
        return event


def _load_or_initialize(config: PipelineConfig) -> Tuple[str, object]:
    if config.resume is not None:
        path = Path(config.resume)
        try:
            state = load_checkpoint(path)
            parameters = state.algorithm_parameters
            mode = "reference" if (
                parameters.temperature_schedule == "linear_restart"
                and parameters.fkm_seed_policy == "compressed_a_random_b_top_q"
                and parameters.acceptance_mode == "fixed_target_multiscale"
            ) else "baseline"
            return mode, state
        except CheckpointError:
            return "enhanced", load_enhanced_checkpoint(path)
    if config.L is None:
        raise ValueError("--L is required unless --resume is supplied")
    is_project_length = validate_length(config.L)
    if config.reference_port:
        parameters = reference_search_parameters(
            config.L, target_profile_offset=config.target_profile_offset,
        )
        return "reference", SearchRunner.new(
            config.L, config.seed, parameters
        ).state
    if config.gcp:
        if config.enhanced:
            raise ValueError("GCP candidate initialization currently supports baseline SearchRunner only")
        # Preserve the exact small power-of-two lift used by the original GCP
        # experiment.  Official Project 2 lengths use the verified periodic or
        # Turyn-adapted construction registered in ``solver.golay``.
        solution = (
            complete_one_flip_each(config.L, config.seed)
            if config.L & (config.L - 1) == 0 else None
        )
        if solution is not None:
            a, b = solution
        else:
            golay_seed = project_length_golay_seed(config.L, config.seed)
            a, b = golay_seed.a, golay_seed.b
        weight_pairs = canonical_weight_pairs(config.L) if is_project_length else None
        content_profiles = canonical_target_content_profiles(config.L) if is_project_length else ()
        if is_project_length:
            ordered_weight_pairs = admissible_weight_pairs(config.L)
            if not ordered_weight_pairs:
                raise ValueError(
                    "L={} has no integer Hamming-weight pair satisfying the necessary (L,4)-PQCP identity".format(config.L)
                )
            if not content_profiles:
                raise ValueError(
                    "L={} has no ordinary/alternating target-content profile".format(config.L)
                )
            a, b, selected = _project_to_target_content_profile(
                a, b, content_profiles, config.seed
            )
            content_profiles = (selected,) + tuple(
                profile for profile in content_profiles if profile != selected
            )
        parameters = SearchParameters(
            acceptance_mode="fixed_target_full" if is_project_length else "objective_plus_target_pair",
            initial_temperature=8.0,
            proposal_samples=DEFAULT_GUIDED_PROPOSAL_SAMPLES,
            objective_energy_weight=DEFAULT_GUIDED_OBJECTIVE_ENERGY_WEIGHT,
            # After the initial GCP-derived basin, retain the existing safe
            # fixed-weight FKM restart schedule whenever the necessary
            # identity has solutions.  Empty means no mathematically valid
            # schedule is available, not that the GCP construction failed.
            weight_pairs=weight_pairs or None,
            fkm_seed_policy=config.fkm_seed_policy,
            fkm_candidate_count=config.fkm_candidate_count,
            fkm_elite_count=config.fkm_elite_count,
            target_content_profiles=_encode_content_profiles(content_profiles) or None,
            preserve_alternating_content=is_project_length,
        )
        runner = SearchRunner.from_candidate(a, b, config.seed, parameters)
        runner.state.temperature = GCP_INITIAL_TEMPERATURE
        return "baseline", runner.state
    weight_pairs = None
    if is_project_length:
        content_profiles = canonical_target_content_profiles(config.L)
        if not content_profiles:
            raise ValueError(
                "L={} has no ordinary/alternating target-content profile satisfying the necessary identities".format(config.L)
            )
    if config.enhanced:
        return "enhanced", EnhancedSearch.new(config.L, config.seed, EnhancedParameters(weight_pairs=weight_pairs)).state
    parameters = SearchParameters(weight_pairs=weight_pairs)
    if is_project_length:
        parameters = SearchParameters(
            weight_pairs=weight_pairs,
            fkm_seed_policy=config.fkm_seed_policy,
            fkm_candidate_count=config.fkm_candidate_count,
            fkm_elite_count=config.fkm_elite_count,
            target_content_profiles=_encode_content_profiles(content_profiles),
            preserve_alternating_content=True,
            acceptance_mode="fixed_target_full",
            initial_temperature=8.0,
            proposal_samples=DEFAULT_GUIDED_PROPOSAL_SAMPLES,
            objective_energy_weight=DEFAULT_GUIDED_OBJECTIVE_ENERGY_WEIGHT,
        )
    return "baseline", SearchRunner.new(config.L, config.seed, parameters).state


def _encode_content_profiles(profiles: Sequence[TargetContentProfile]):
    """Convert mathematical profiles into the checkpoint's compact tuples."""
    return tuple(
        (p.k, p.eta, p.a_even_ones, p.a_odd_ones,
         p.b_even_ones, p.b_odd_ones)
        for p in profiles
    )


def _project_to_target_content_profile(
    a: Sequence[int],
    b: Sequence[int],
    profiles: Sequence[TargetContentProfile],
    seed: int,
) -> Tuple[Tuple[int, ...], Tuple[int, ...], TargetContentProfile]:
    """Minimally flip a seed into one compatible even/odd content profile."""
    pair = (tuple(a), tuple(b))
    current = pair_content(*pair)
    distances = {
        profile: sum(abs(left - right) for left, right in zip(
            current,
            (profile.a_even_ones, profile.a_odd_ones,
             profile.b_even_ones, profile.b_odd_ones),
        ))
        for profile in profiles
    }
    minimum = min(distances.values())
    closest = sorted(profile for profile, distance in distances.items() if distance == minimum)
    rng = random.Random(seed)
    selected = closest[rng.randrange(len(closest))]

    def project(values: Tuple[int, ...], even_target: int, odd_target: int) -> Tuple[int, ...]:
        result = list(values)
        for parity, target in ((0, even_target), (1, odd_target)):
            positions = list(range(parity, len(result), 2))
            current_count = sum(result[index] for index in positions)
            source = 0 if current_count < target else 1
            choices = [index for index in positions if result[index] == source]
            rng.shuffle(choices)
            for index in choices[:abs(target - current_count)]:
                result[index] ^= 1
        return tuple(result)

    projected_a = project(pair[0], selected.a_even_ones, selected.a_odd_ones)
    projected_b = project(pair[1], selected.b_even_ones, selected.b_odd_ones)
    expected = (selected.a_even_ones, selected.a_odd_ones,
                selected.b_even_ones, selected.b_odd_ones)
    if pair_content(projected_a, projected_b) != expected:
        raise RuntimeError("failed to project seed to target content profile")
    return projected_a, projected_b, selected


def _project_to_admissible_weights(
    a: Sequence[int],
    b: Sequence[int],
    weight_pairs: Sequence[Tuple[int, int]],
    seed: int,
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Minimally perturb a constructed seed onto one exact legal weight pair.

    This is a one-time initialization operation.  Once projected, every SA
    proposal preserves both selected weights.  Ties are resolved
    deterministically from ``seed`` so independent runs retain diversity.
    """
    pair = (tuple(a), tuple(b))
    current = (sum(pair[0]), sum(pair[1]))
    distances = {
        target: abs(current[0] - target[0]) + abs(current[1] - target[1])
        for target in weight_pairs
    }
    minimum = min(distances.values())
    closest = sorted(target for target, distance in distances.items() if distance == minimum)
    rng = random.Random(seed)
    target = closest[rng.randrange(len(closest))]

    def project(values: Tuple[int, ...], target_weight: int) -> Tuple[int, ...]:
        result = list(values)
        current_weight = sum(result)
        source_bit = 0 if current_weight < target_weight else 1
        positions = [index for index, bit in enumerate(result) if bit == source_bit]
        rng.shuffle(positions)
        for position in positions[:abs(target_weight - current_weight)]:
            result[position] ^= 1
        if sum(result) != target_weight:
            raise RuntimeError("failed to project Golay seed to requested Hamming weight")
        return tuple(result)

    return project(pair[0], target[0]), project(pair[1], target[1])


def _record_baseline_event(observer: _BestObserver, outcome: object, state: object,
                           trigger_z3: Callable[[Dict[str, object]], bool]) -> bool:
    """Record an observed baseline improvement, then optionally trigger exact Z3."""
    if outcome.improved_best:
        return trigger_z3(observer.record(state, state.elapsed_seconds, outcome.old_best_score))
    return False


def _run_enhanced(search: EnhancedSearch, config: PipelineConfig, checkpoint_path: Path, observer: _BestObserver,
                  progress: Optional[Callable[[object, float], None]],
                  record_verified: Callable[[Sequence[int], Sequence[int], object], None],
                  trigger_z3: Callable[[Dict[str, object]], bool]) -> Tuple[object, bool, float]:
    """Time/checkpoint wrapper around existing enhanced ``step``; no new move logic.

    Unlike the original bounded enhanced helper, a verified score-zero state
    is recorded and then restarted so a long run can continue seeking another
    new PQCP during its remaining user-provided budget.
    """
    started = last = last_checkpoint = last_progress = perf_counter()
    interrupted = False
    try:
        while perf_counter() - started < config.seconds:
            if search.state.finished:
                if search.state.current_score != 0:
                    break
                record_verified(search.state.current_a, search.state.current_b, search.state)
                _restart_after_enhanced_solution(search)
                continue
            if search.state.current_score == 0:
                record_verified(search.state.current_a, search.state.current_b, search.state)
                _restart_after_enhanced_solution(search)
                continue
            old_best = search.state.best_score
            search.step()
            now = perf_counter()
            search.state.elapsed_seconds += max(0.0, now - last)
            last = now
            if search.state.best_score < old_best:
                if trigger_z3(observer.record(search.state, search.state.elapsed_seconds, old_best)):
                    search._restart()
            if search.state.current_score == 0:
                record_verified(search.state.current_a, search.state.current_b, search.state)
                _restart_after_enhanced_solution(search)
                continue
            if now - last_checkpoint >= config.checkpoint_interval:
                save_enhanced_checkpoint(checkpoint_path, search.state)
                last_checkpoint = now
            if progress is not None and now - last_progress >= config.progress_interval:
                progress(search.state, now - started)
                last_progress = now
    except KeyboardInterrupt:
        interrupted = True
    finally:
        now = perf_counter()
        search.state.elapsed_seconds += max(0.0, now - last)
        save_enhanced_checkpoint(checkpoint_path, search.state)
    return search.state, interrupted, perf_counter() - started


def _restart_after_enhanced_solution(search: EnhancedSearch) -> None:
    """Continue an existing enhanced run after its independently verified solution.

    ``EnhancedSearch.step`` deliberately marks score zero as finished.  The
    pipeline has already verified and persisted that candidate, so it invokes
    the searcher's existing restart routine rather than adding a move or
    changing acceptance.  Synchronizing the serialized RNG state keeps a
    checkpoint written immediately after this restart resumable.
    """
    search.state.finished = False
    search._restart()
    search.state.rng_state = search._rng.getstate()


def _run_optional_z3(
    config: PipelineConfig,
    length: int,
    elites: Sequence[SAElite],
    root: Path,
    encoded_target_profiles=None,
):
    """Run guidance -> beam repair -> positive-radius Z3 fallback.

    Guidance/beam solutions are verified and saved directly.  Only an
    unsolved but objectively improved beam endpoint reaches Z3, at a positive
    Hamming radius.  Bounded misses are never promoted to global UNSAT.
    """
    if not elites:
        return None
    center = elites[-1]
    if config.z3_guided:
        guidance = guided_completion(
            center.a,
            center.b,
            top_k=config.z3_guidance_top_k,
            max_profile_lower_bound=config.z3_profile_radius,
        )
        if guidance.solved:
            if guidance.a is None or guidance.b is None:  # pragma: no cover - solved contract
                raise RuntimeError("guided completion reported solved without sequences")
            return _persist_algorithm_solution(
                root, length, "guidance", guidance.a, guidance.b,
                {
                    "first_moves_examined": guidance.first_moves_examined,
                    "second_moves_examined": guidance.second_moves_examined,
                },
            )

        beam = beam_repair(
            center.a,
            center.b,
            max_depth=config.repair_max_depth,
            beam_width=config.repair_beam_width,
        )
        if beam.solved:
            return _persist_algorithm_solution(
                root, length, "beam", beam.a, beam.b,
                {
                    "depth": beam.depth,
                    "states_examined": beam.states_examined,
                    "elapsed": beam.elapsed_time,
                },
            )

        if not config.z3:
            return {
                "status": "REPAIR_MISS",
                "solved": False,
                "z3_executed": False,
                "reason": "guidance and beam missed; exact Z3 fallback is disabled",
                "beam_initial_score": beam.initial_score,
                "beam_best_score": beam.best_score,
                "beam_states_examined": beam.states_examined,
                "restart_recommended": False,
            }

        try:
            from .z3_solver import solve_with_z3
        except ImportError as error:  # pragma: no cover - optional dependency environment
            return {"status": "UNAVAILABLE", "solved": False, "z3_executed": False, "reason": str(error)}
        solve_kwargs = {
            "radius": config.z3_completion_radius,
            "timeout_ms": int(config.z3_timeout * 1000),
        }
        if encoded_target_profiles:
            solve_kwargs["allowed_target_content_profiles"] = tuple(
                TargetContentProfile(length, *values) for values in encoded_target_profiles
            )
        completion = solve_with_z3(length, beam.a, beam.b, **solve_kwargs)
        new_solution = False
        if completion.status == "SAT":
            if not completion.verified or completion.a is None or completion.b is None or completion.profile is None:
                raise RuntimeError("Z3 SAT completion failed independent verification")
            new_solution = append_verified_solution_if_new(
                length, tuple(completion.a), tuple(completion.b), root
            )
            _save_z3_verified_result(root, completion, completion.profile)
        return {
            "status": "Z3_{}".format(completion.status),
            "solved": completion.status == "SAT" and completion.verified,
            "z3_executed": True,
            "reason": completion.reason,
            "radius": completion.radius,
            "elapsed": completion.elapsed_time,
            "beam_initial_score": beam.initial_score,
            "beam_best_score": beam.best_score,
            "beam_states_examined": beam.states_examined,
            "new_solution_written": new_solution,
            "restart_recommended": completion.status != "SAT",
        }

    # Explicit rollback mode retains the previous broad-radius Z3 portfolio.
    if run_hybrid_portfolio is None or PortfolioConfig is None:
        return {"status": "UNAVAILABLE", "reason": "optional z3 package is unavailable"}
    selected_elite = center
    selected_radius = 3
    timeout_ms = int(config.z3_timeout * 1000)
    config_z3 = PortfolioConfig(
        timeout_radius_0_ms=timeout_ms, timeout_radius_1_ms=timeout_ms,
        timeout_radius_2_ms=timeout_ms, timeout_radius_3_ms=timeout_ms,
        radius_0_elite_count=1 if selected_radius == 0 else 0,
        radius_1_elite_count=0, radius_2_elite_count=0,
        radius_3_elite_count=1 if selected_radius == 3 else 0,
        total_timeout_seconds=config.z3_timeout,
    )
    try:
        result = run_hybrid_portfolio(length, (selected_elite,), 1, config_z3)
        if result.solved and result.solution is not None:
            solution = result.solution
            if solution.a is None or solution.b is None:
                raise RuntimeError("Z3 reported a solved result without sequences")
            verification = verify_pqcp(solution.a, solution.b)
            if not verification.is_valid:
                raise RuntimeError("Z3 SAT candidate failed independent verification")
            append_verified_solution_if_new(length, tuple(solution.a), tuple(solution.b), root)
            _save_z3_verified_result(root, solution, verification.profile)
        return result
    except ImportError as error:
        # Keep an unavailable optional Z3 dependency distinct from UNSAT.
        return {"status": "UNAVAILABLE", "reason": str(error)}


def _persist_algorithm_solution(
    root: Path,
    length: int,
    method: str,
    a: Sequence[int],
    b: Sequence[int],
    diagnostics: Dict[str, object],
) -> Dict[str, object]:
    """Independently verify and persist a non-Z3 final-repair solution."""
    verification = verify_pqcp(a, b)
    if not verification.is_valid:
        raise RuntimeError("{} repair candidate failed independent verification".format(method))
    new_solution = append_verified_solution_if_new(length, tuple(a), tuple(b), root)
    path = _save_algorithm_verified_result(root, length, method, a, b, verification.profile)
    return {
        "status": "{}_SAT".format(method.upper()),
        "solved": True,
        "z3_executed": False,
        "new_solution_written": new_solution,
        "verified_path": str(path),
        **diagnostics,
    }


def _checkpoint_name(length: int, method: str) -> str:
    if method == "baseline":
        return "L{}.json".format(length)
    return "L{}_{}.json".format(length, method)


def _select_z3_result(results: Sequence[object]) -> Optional[object]:
    """Retain an earlier solved completion instead of hiding it with a later miss."""
    for result in results:
        solved = result.get("solved", False) if isinstance(result, dict) else getattr(result, "solved", False)
        if bool(solved):
            return result
    return results[-1] if results else None


def _save_verified_result(root: Path, method: str, state: object, a: Sequence[int], b: Sequence[int], profile: Sequence[int]) -> Path:
    """Save a verified run-specific JSON record without overwriting prior solutions."""
    directory = root / "results" / "verified"
    base = directory / "L{}_{}_seed{}_iter{}.json".format(state.L, method, state.seed, state.iteration)
    destination = base
    suffix = 1
    while destination.exists():
        destination = base.with_name("{}_{}.json".format(base.stem, suffix))
        suffix += 1
    nonzero = [shift for shift in range(1, state.L) if profile[shift] != 0]
    atomic_write_json(destination, {
        "L": state.L, "method": method, "seed": state.seed, "iteration": state.iteration,
        "restart_index": state.restart_index, "A": list(a), "B": list(b), "profile": list(profile),
        "nonzero_shifts": nonzero, "nonzero_values": [profile[shift] for shift in nonzero], "verified": True,
    })
    return destination


def _save_z3_verified_result(root: Path, solution: object, profile: Sequence[int]) -> Path:
    """Persist a verified bounded-Z3 result without replacing any prior solution file."""
    directory = root / "results" / "verified"
    base = directory / "L{}_z3_radius{}.json".format(solution.L, solution.radius)
    destination = base
    suffix = 1
    while destination.exists():
        destination = base.with_name("{}_{}.json".format(base.stem, suffix))
        suffix += 1
    nonzero = [shift for shift in range(1, solution.L) if profile[shift] != 0]
    atomic_write_json(destination, {
        "L": solution.L, "method": "bounded_z3", "radius": solution.radius,
        "elapsed": solution.elapsed_time, "A": list(solution.a), "B": list(solution.b),
        "profile": list(profile), "nonzero_shifts": nonzero,
        "nonzero_values": [profile[shift] for shift in nonzero], "verified": True,
    })
    return destination


def _save_algorithm_verified_result(
    root: Path,
    length: int,
    method: str,
    a: Sequence[int],
    b: Sequence[int],
    profile: Sequence[int],
) -> Path:
    """Persist a verified guidance/beam solution without overwriting files."""
    directory = root / "results" / "verified"
    base = directory / "L{}_{}_repair.json".format(length, method)
    destination = base
    suffix = 1
    while destination.exists():
        destination = base.with_name("{}_{}.json".format(base.stem, suffix))
        suffix += 1
    nonzero = [shift for shift in range(1, length) if profile[shift] != 0]
    atomic_write_json(destination, {
        "L": length, "method": "{}_repair".format(method),
        "A": list(a), "B": list(b), "profile": list(profile),
        "nonzero_shifts": nonzero,
        "nonzero_values": [profile[shift] for shift in nonzero],
        "verified": True,
    })
    return destination
