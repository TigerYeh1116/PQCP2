"""Reliable long-run wrapper around FKM-initialized fixed-weight swap SA."""

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import re
import threading
from time import perf_counter
from typing import Callable, Optional, Sequence, Tuple

from .annealing import (
    initialize_from_fkm, initialize_from_fkm_content_profile,
    initialize_from_fkm_weights,
)
from .checkpoint import SearchParameters, SearchState, atomic_write_json, load_checkpoint, save_checkpoint
from .compression import CorrelationState
from .correlation import full_correlation_profile
from .moves import (
    SwapEvaluation,
    apply_weight_preserving_swap,
    hamming_weights,
    rollback_weight_preserving_swap,
    sample_weight_preserving_swaps,
    trial_fixed_target_swap_energy,
    trial_fixed_target_swap_energy_key,
    trial_multiscale_fixed_target_swap_energy,
    trial_weight_preserving_swap,
)
from .verifier import verify_pqcp
from .structured_energy import multiscale_error_energy
from .target_profiles import TargetContentProfile, pair_content


# Empirically selected by the paired score<=8 benchmark.  Keeping the
# historical one-proposal behavior in SearchParameters' default makes old
# checkpoints and explicit baselines reproducible, while production entry
# points opt into this measured search-width setting.
DEFAULT_GUIDED_PROPOSAL_SAMPLES = 10
DEFAULT_GUIDED_OBJECTIVE_ENERGY_WEIGHT = 3
# Swap construction preserves weight/content algebraically.  Rechecking the
# complete sequences periodically retains a runtime guard without spending an
# additional O(L) scan on every hot-path iteration.
INVARIANT_CHECK_INTERVAL = 1024


class SearchImplementationError(RuntimeError):
    """Raised when a score-zero candidate fails independent verification."""


@dataclass(frozen=True)
class StepOutcome:
    """The observable result of one weight-preserving SA iteration."""

    improved_best: bool
    verified_solution: Optional[Tuple[Tuple[int, ...], Tuple[int, ...]]]
    old_best_score: Optional[int] = None
    new_best_score: Optional[int] = None
    restart_info: Optional["RestartInfo"] = None
    kick_performed: bool = False


@dataclass(frozen=True)
class RestartInfo:
    """Read-only metadata emitted when the existing restart mechanism runs."""

    restart_index: int
    start_score: int
    local_best_score: int
    start_iteration: int
    end_iteration: int
    temperature: float
    reason: str


@dataclass(frozen=True)
class RunSummary:
    """Controlled stop information for a time-budgeted long search invocation."""

    state: SearchState
    checkpoint_path: Optional[Path]
    best_path: Optional[Path]
    interrupted: bool
    verified_solution_found: bool


class SearchRunner:
    """Mutable runtime that can be reconstructed exactly from ``SearchState``.

    Every proposal exchanges one zero and one one inside A or inside B, so
    both Hamming weights remain fixed throughout a restart.  This runner
    stores all mutable trajectory state after every step.  Resume
    therefore restores the same next random draw, state, temperature, and
    restart condition rather than reinitializing FKM.
    """

    def __init__(self, state: SearchState) -> None:
        """Restore a runner from validated state and its serialized RNG state."""
        self.state = state
        self._correlation = CorrelationState(state.current_a, state.current_b)
        if self._correlation.profile != tuple(full_correlation_profile(state.current_a, state.current_b)):
            raise ValueError("checkpoint current profile is internally inconsistent")
        if self._correlation.score != state.current_score:
            raise ValueError("checkpoint current objective is internally inconsistent")
        self._acceptance_energy = self._current_acceptance_energy()
        if state.restart_best_energy is None:
            state.restart_best_energy = self._acceptance_energy
        if state.restart_last_energy_improvement_iteration is None:
            state.restart_last_energy_improvement_iteration = state.iteration
        if state.restart_best_energy > self._acceptance_energy:
            raise ValueError("checkpoint restart best energy exceeds current energy")
        self._rng = random.Random()
        self._rng.setstate(state.rng_state)
        self.state._rng_provider = self._rng

    @classmethod
    def new(cls, L: int, seed: int, parameters: SearchParameters) -> "SearchRunner":
        """Create deterministic FKM-initialized state for restart index zero."""
        if not isinstance(L, int) or isinstance(L, bool) or L <= 0:
            raise ValueError("L must be a positive integer")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("seed must be an integer")
        a, b, rng = _restart_initialization(L, seed, 0, parameters)
        correlation = CorrelationState(a, b)
        score = correlation.score
        return cls(SearchState(
            L=L,
            current_a=a,
            current_b=b,
            current_score=score,
            best_a=a,
            best_b=b,
            best_score=score,
            iteration=0,
            restart_index=0,
            restart_start_iteration=0,
            restart_start_score=score,
            restart_local_best_score=score,
            temperature=parameters.initial_temperature,
            rng_state=rng.getstate(),
            seed=seed,
            elapsed_seconds=0.0,
            last_improvement_iteration=0,
            stagnation_count=0,
            algorithm_parameters=parameters,
        ))

    @classmethod
    def from_candidate(
        cls,
        a: Sequence[int],
        b: Sequence[int],
        seed: int,
        parameters: SearchParameters,
    ) -> "SearchRunner":
        """Start a reproducible trajectory from an explicit complete A/B candidate."""
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("seed must be an integer")
        correlation = CorrelationState(a, b)
        candidate_a, candidate_b = correlation.a, correlation.b
        score = correlation.score
        rng = random.Random(seed)
        return cls(SearchState(
            L=correlation.L,
            current_a=candidate_a,
            current_b=candidate_b,
            current_score=score,
            best_a=candidate_a,
            best_b=candidate_b,
            best_score=score,
            iteration=0,
            restart_index=0,
            restart_start_iteration=0,
            restart_start_score=score,
            restart_local_best_score=score,
            temperature=parameters.initial_temperature,
            rng_state=rng.getstate(),
            seed=seed,
            elapsed_seconds=0.0,
            last_improvement_iteration=0,
            stagnation_count=0,
            algorithm_parameters=parameters,
        ))

    @classmethod
    def resume(cls, checkpoint_path: Path) -> "SearchRunner":
        """Restore exactly from a JSON checkpoint without FKM reinitialization."""
        return cls(load_checkpoint(checkpoint_path))

    def step(self) -> StepOutcome:
        """Execute one exact incremental weight-preserving swap proposal.

        ``proposal_samples`` legal swaps are evaluated without mutating the
        correlation state.  The proposal with the lowest configured SA energy
        is then applied once and passed through the unchanged Metropolis
        acceptance rule.  A value of one is the historical single-proposal
        baseline.
        """
        if self.state.finished:
            return StepOutcome(False, None, self.state.best_score, self.state.best_score)
        old_best_score = self.state.best_score
        check_invariants = self.state.iteration % INVARIANT_CHECK_INTERVAL == 0
        weights_before = (
            hamming_weights(self._correlation.a, self._correlation.b)
            if check_invariants else None
        )
        move = None
        evaluation = None
        best_energy = None
        best_rank = None
        target = self._current_target_content_profile()
        use_fixed_target_fast_path = (
            self.state.algorithm_parameters.acceptance_mode == "fixed_target_full"
        )
        multiscale_weights = self._compression_weights()
        use_multiscale_fast_path = multiscale_weights is not None
        use_compressed_tiebreak = (
            self.state.algorithm_parameters.acceptance_mode
            == "fixed_target_full_compressed_tiebreak"
        )
        candidates = sample_weight_preserving_swaps(
            self._correlation.a,
            self._correlation.b,
            self._rng,
            self.state.algorithm_parameters.proposal_samples,
            same_parity=self.state.algorithm_parameters.preserve_alternating_content,
        )
        evaluation_cache = {}
        if use_compressed_tiebreak:
            if target is None:
                raise SearchImplementationError(
                    "compressed tie-break requires a target content profile"
                )
            # Compression is needed only when two sampled moves have the same
            # best complete energy.  Evaluate the cheap full component first,
            # then spend compression work solely on the actual tie set.
            tied_moves = []
            for candidate in candidates:
                if candidate in evaluation_cache:
                    candidate_energy = evaluation_cache[candidate]
                else:
                    candidate_energy = trial_fixed_target_swap_energy(
                        self._correlation, candidate, target.k, 4 * target.eta,
                    )
                    evaluation_cache[candidate] = candidate_energy
                if best_energy is None or candidate_energy < best_energy:
                    best_energy = candidate_energy
                    tied_moves = [candidate]
                elif candidate_energy == best_energy and candidate not in tied_moves:
                    tied_moves.append(candidate)
            if len(tied_moves) == 1:
                move = tied_moves[0]
            elif tied_moves:
                factors = (2, 4) if self.state.L % 4 == 0 else (2,)
                move = min(
                    tied_moves,
                    key=lambda candidate: trial_fixed_target_swap_energy_key(
                        self._correlation, candidate, target.k,
                        4 * target.eta, factors,
                    ),
                )
        else:
            for candidate in candidates:
                cached = evaluation_cache.get(candidate)
                if cached is None:
                    if use_fixed_target_fast_path:
                        if target is None:
                            raise SearchImplementationError(
                                "fixed-target energy requires a target content profile"
                            )
                        candidate_evaluation = None
                        candidate_energy = trial_fixed_target_swap_energy(
                            self._correlation, candidate, target.k, 4 * target.eta,
                        )
                        candidate_rank = (candidate_energy,)
                        cached = (candidate_evaluation, candidate_energy, candidate_rank)
                    elif use_multiscale_fast_path:
                        if target is None or multiscale_weights is None:
                            raise SearchImplementationError(
                                "compressed fixed-target energy requires a target content profile"
                            )
                        candidate_evaluation = None
                        candidate_energy = trial_multiscale_fixed_target_swap_energy(
                            self._correlation, candidate, target.k, 4 * target.eta,
                            multiscale_weights,
                        )
                        candidate_rank = (candidate_energy,)
                        cached = (candidate_evaluation, candidate_energy, candidate_rank)
                    else:
                        candidate_evaluation = trial_weight_preserving_swap(
                            self._correlation, candidate
                        )
                        candidate_energy = self._evaluation_acceptance_energy(
                            candidate_evaluation
                        )
                        candidate_rank = (candidate_energy,)
                        cached = (candidate_evaluation, candidate_energy, candidate_rank)
                    evaluation_cache[candidate] = cached
                else:
                    candidate_evaluation, candidate_energy, candidate_rank = cached
                if best_rank is None or candidate_rank < best_rank:
                    move = candidate
                    evaluation = candidate_evaluation
                    best_energy = candidate_energy
                    best_rank = candidate_rank
        if move is not None:
            apply_weight_preserving_swap(self._correlation, move)
            if best_energy is None:
                raise SearchImplementationError("selected swap lacks an exact energy")
            proposed_energy = best_energy
            if (use_fixed_target_fast_path or use_multiscale_fast_path
                    or use_compressed_tiebreak):
                proposed_score = self._correlation.score
                if check_invariants and self._current_acceptance_energy() != proposed_energy:
                    raise SearchImplementationError(
                        "fast fixed-target evaluation disagrees with exact incremental state"
                    )
            else:
                if evaluation is None:
                    raise SearchImplementationError("selected swap lacks an exact evaluation")
                proposed_score = evaluation.score
                if (
                    self._correlation.score != proposed_score
                    or self._correlation.target_pair_energy != evaluation.target_pair_energy
                ):
                    raise SearchImplementationError("trial swap evaluation disagrees with exact incremental state")
            delta = proposed_energy - self._acceptance_energy
            denominator = (
                self.state.algorithm_parameters.metropolis_scale
                * self.state.temperature
            )
            accepted = delta <= 0 or self._rng.random() < math.exp(-delta / denominator)
            if accepted:
                self.state.current_a = self._correlation.a
                self.state.current_b = self._correlation.b
                self.state.current_score = proposed_score
                self._acceptance_energy = proposed_energy
                if (
                    self.state.restart_best_energy is None
                    or proposed_energy < self.state.restart_best_energy
                ):
                    self.state.restart_best_energy = proposed_energy
                    self.state.restart_last_energy_improvement_iteration = (
                        self.state.iteration + 1
                    )
                if proposed_score < self.state.restart_local_best_score:
                    self.state.restart_local_best_score = proposed_score
            else:
                rollback_weight_preserving_swap(self._correlation, move)
        if (
            check_invariants
            and hamming_weights(self._correlation.a, self._correlation.b) != weights_before
        ):
            raise SearchImplementationError("weight-preserving SA move changed a Hamming weight")
        target = self._current_target_content_profile()
        if (
            check_invariants
            and target is not None
            and self.state.algorithm_parameters.preserve_alternating_content
        ):
            expected = (target.a_even_ones, target.a_odd_ones,
                        target.b_even_ones, target.b_odd_ones)
            if pair_content(self._correlation.a, self._correlation.b) != expected:
                raise SearchImplementationError("same-parity move changed target content")

        self.state.iteration += 1
        kick_performed = self._maybe_kick()
        improved = False
        if self.state.current_score < self.state.best_score:
            self.state.best_a = self.state.current_a
            self.state.best_b = self.state.current_b
            self.state.best_score = self.state.current_score
            self.state.last_improvement_iteration = self.state.iteration
            improved = True
        self.state.temperature = self._next_temperature()
        verified_solution = None
        restart_info = None
        if self.state.current_score == 0:
            verification = verify_pqcp(self.state.current_a, self.state.current_b)
            if not verification.is_valid:
                raise SearchImplementationError("score-zero SA candidate failed independent verification")
            verified_solution = (self.state.current_a, self.state.current_b)
            restart_info = self._restart("verified solution")
        elif self._should_restart():
            restart_info = self._restart("stagnation or restart iteration limit")
        improved = self.state.best_score < old_best_score
        return StepOutcome(
            improved, verified_solution, old_best_score, self.state.best_score,
            restart_info, kick_performed,
        )

    def _maybe_kick(self) -> bool:
        """Apply exact same-parity swaps after restart-local energy stagnation.

        A kick is a search heuristic, never a pruning condition.  It leaves
        all fixed content invariants unchanged and updates PACF exactly.
        """
        parameters = self.state.algorithm_parameters
        threshold = parameters.kick_stagnation_iterations
        last = self.state.restart_last_energy_improvement_iteration
        if threshold is None or last is None or self.state.iteration - last < threshold:
            return False
        changed = False
        for _ in range(parameters.kick_swaps):
            moves = sample_weight_preserving_swaps(
                self._correlation.a,
                self._correlation.b,
                self._rng,
                1,
                same_parity=parameters.preserve_alternating_content,
            )
            if not moves:
                break
            apply_weight_preserving_swap(self._correlation, moves[0])
            changed = True
        self.state.current_a = self._correlation.a
        self.state.current_b = self._correlation.b
        self.state.current_score = self._correlation.score
        self._acceptance_energy = self._current_acceptance_energy()
        if self.state.current_score < self.state.restart_local_best_score:
            self.state.restart_local_best_score = self.state.current_score
        if (
            self.state.restart_best_energy is None
            or self._acceptance_energy < self.state.restart_best_energy
        ):
            self.state.restart_best_energy = self._acceptance_energy
        self.state.restart_last_energy_improvement_iteration = self.state.iteration
        return changed

    def _next_temperature(self) -> float:
        """Advance the legacy geometric or restart-local linear schedule."""
        parameters = self.state.algorithm_parameters
        if parameters.temperature_schedule == "geometric":
            return max(
                parameters.min_temperature,
                self.state.temperature * parameters.cooling_rate,
            )
        if parameters.max_iterations_per_restart is None:  # validated earlier
            raise SearchImplementationError(
                "linear restart schedule lacks a restart iteration limit"
            )
        progress = min(
            1.0,
            (self.state.iteration - self.state.restart_start_iteration)
            / parameters.max_iterations_per_restart,
        )
        return max(
            parameters.min_temperature,
            parameters.initial_temperature
            - (parameters.initial_temperature - parameters.min_temperature) * progress,
        )

    def run(
        self,
        seconds: float,
        checkpoint_path: Optional[Path] = None,
        best_path: Optional[Path] = None,
        checkpoint_interval: Optional[float] = 60.0,
        progress_interval: Optional[float] = 60.0,
        progress_callback: Optional[Callable[[SearchState, float], None]] = None,
        on_verified_solution: Optional[Callable[[Tuple[int, ...], Tuple[int, ...], SearchState], None]] = None,
        event_callback: Optional[Callable[[StepOutcome, SearchState], None]] = None,
        clock: Callable[[], float] = perf_counter,
    ) -> RunSummary:
        """Run until time budget, completion, or Ctrl+C while safely persisting state."""
        if seconds < 0:
            raise ValueError("seconds must be non-negative")
        started = clock()
        last_clock = started
        last_checkpoint = started
        last_progress = started
        interrupted = False
        verified_solution_found = False
        if best_path is not None:
            save_best_candidate(best_path, self.state)
        try:
            if not self.state.finished and self.state.current_score == 0:
                verification = verify_pqcp(self.state.current_a, self.state.current_b)
                if not verification.is_valid:
                    raise SearchImplementationError("score-zero initial candidate failed independent verification")
                verified_solution_found = True
                if on_verified_solution is not None:
                    on_verified_solution(self.state.current_a, self.state.current_b, self.state)
                self._restart("verified initial solution")
            while not self.state.finished and clock() - started < seconds:
                outcome = self.step()
                if outcome.improved_best and best_path is not None:
                    save_best_candidate(best_path, self.state)
                if outcome.verified_solution is not None:
                    verified_solution_found = True
                    if on_verified_solution is not None:
                        on_verified_solution(outcome.verified_solution[0], outcome.verified_solution[1], self.state)
                now = clock()
                self.state.elapsed_seconds += max(0.0, now - last_clock)
                last_clock = now
                if event_callback is not None and (outcome.improved_best or outcome.restart_info is not None):
                    event_callback(outcome, self.state)
                if checkpoint_path is not None and checkpoint_interval is not None and now - last_checkpoint >= checkpoint_interval:
                    save_checkpoint(checkpoint_path, self.state)
                    last_checkpoint = now
                if progress_callback is not None and progress_interval is not None and now - last_progress >= progress_interval:
                    progress_callback(self.state, now - started)
                    last_progress = now
        except KeyboardInterrupt:
            interrupted = True
        finally:
            now = clock()
            self.state.elapsed_seconds += max(0.0, now - last_clock)
            if checkpoint_path is not None:
                save_checkpoint(checkpoint_path, self.state)
            if best_path is not None:
                save_best_candidate(best_path, self.state)
        return RunSummary(self.state, checkpoint_path, best_path, interrupted, verified_solution_found)

    def _should_restart(self) -> bool:
        """Apply only configured deterministic iteration-based restart triggers."""
        parameters = self.state.algorithm_parameters
        since_improvement = self.state.iteration - self.state.last_improvement_iteration
        in_restart = self.state.iteration - self.state.restart_start_iteration
        return (
            parameters.stagnation_iterations is not None and since_improvement >= parameters.stagnation_iterations
        ) or (
            parameters.max_iterations_per_restart is not None and in_restart >= parameters.max_iterations_per_restart
        )

    def restart_after_completion_miss(self) -> RestartInfo:
        """Start a fresh deterministic trajectory after bounded completion fails.

        Completion is allowed only inside its configured Hamming radius.  A
        miss therefore changes the SA basin instead of silently enlarging the
        exact neighborhood.  This public wrapper preserves the existing
        restart seed schedule and checkpoint/resume state.
        """
        return self._restart("bounded Z3 completion miss")

    def _restart(self, reason: str) -> RestartInfo:
        """Start the next deterministic FKM trajectory or mark a finite run complete."""
        parameters = self.state.algorithm_parameters
        info = RestartInfo(
            restart_index=self.state.restart_index,
            start_score=self.state.restart_start_score,
            local_best_score=self.state.restart_local_best_score,
            start_iteration=self.state.restart_start_iteration,
            end_iteration=self.state.iteration,
            temperature=self.state.temperature,
            reason=reason,
        )
        if parameters.max_restarts is not None and self.state.restart_index + 1 >= parameters.max_restarts:
            self.state.finished = True
            return info
        self.state.restart_index += 1
        a, b, rng = _restart_initialization(self.state.L, self.state.seed, self.state.restart_index, parameters)
        self._rng = rng
        self.state._rng_provider = self._rng
        self._correlation = CorrelationState(a, b)
        self._acceptance_energy = self._current_acceptance_energy()
        self.state.current_a = a
        self.state.current_b = b
        self.state.current_score = self._correlation.score
        self.state.restart_start_iteration = self.state.iteration
        self.state.restart_start_score = self.state.current_score
        self.state.restart_local_best_score = self.state.current_score
        self.state.last_improvement_iteration = self.state.iteration
        self.state.restart_best_energy = self._acceptance_energy
        self.state.restart_last_energy_improvement_iteration = self.state.iteration
        self.state.temperature = parameters.initial_temperature
        self.state.stagnation_count += 1
        if self.state.current_score < self.state.best_score:
            self.state.best_a = a
            self.state.best_b = b
            self.state.best_score = self.state.current_score
        return info

    def _current_acceptance_energy(self) -> int:
        """Return the configured exact SA navigation energy for current A/B."""
        mode = self.state.algorithm_parameters.acceptance_mode
        target = self._current_target_content_profile()
        if mode.startswith("fixed_target_"):
            if target is None:
                raise SearchImplementationError("fixed-target energy requires a target content profile")
            return self._structured_energy(self._correlation.profile, target)
        if mode == "target_pair_squared":
            return self._correlation.target_pair_energy
        if self.state.algorithm_parameters.acceptance_mode == "objective_plus_target_pair":
            # Preserve direct pressure on the reported Project objective while
            # adding a normalized smoother sidelobe-error gradient.
            return (
                self.state.algorithm_parameters.objective_energy_weight
                * self._correlation.score
                + self._correlation.target_pair_energy // 16
            )
        return self._correlation.score

    def _evaluation_acceptance_energy(self, evaluation: SwapEvaluation) -> int:
        """Map a non-mutating exact swap evaluation to configured SA energy."""
        mode = self.state.algorithm_parameters.acceptance_mode
        target = self._current_target_content_profile()
        if mode.startswith("fixed_target_"):
            if target is None:
                raise SearchImplementationError("fixed-target energy requires a target content profile")
            return self._structured_energy(evaluation.profile, target)
        if mode == "target_pair_squared":
            return evaluation.target_pair_energy
        if mode == "objective_plus_target_pair":
            return (
                self.state.algorithm_parameters.objective_energy_weight
                * evaluation.score
                + evaluation.target_pair_energy // 16
            )
        return evaluation.score

    def _current_target_content_profile(self) -> Optional[TargetContentProfile]:
        """Reconstruct the deterministic target case assigned to this restart."""
        profiles = self.state.algorithm_parameters.target_content_profiles
        if profiles is None:
            return None
        index = _target_profile_index(
            self.state.seed,
            self.state.restart_index,
            len(profiles),
            self.state.algorithm_parameters.randomize_target_profiles,
            self.state.algorithm_parameters.target_profile_offset,
        )
        values = profiles[index]
        return TargetContentProfile(self.state.L, *values)

    def _structured_energy(self, profile: Sequence[int], target: TargetContentProfile) -> int:
        """Evaluate one of the explicit fixed-target ablation energies."""
        mode = self.state.algorithm_parameters.acceptance_mode
        if mode in ("fixed_target_full", "fixed_target_full_compressed_tiebreak"):
            target_value = 4 * target.eta
            return sum(
                (profile[shift] - (target_value if shift == target.k else 0)) ** 2
                for shift in range(1, self.state.L // 2 + 1)
            ) // 16
        if mode == "fixed_target_full_e2":
            return multiscale_error_energy(profile, target.k, target.eta, {2: 1}) // 16
        if mode == "fixed_target_multiscale":
            weights = {2: 2}
            if self.state.L % 4 == 0:
                weights[4] = 4
            return multiscale_error_energy(profile, target.k, target.eta, weights) // 16
        raise SearchImplementationError("unknown fixed-target energy mode")

    def _compression_weights(self):
        """Return explicit weights for a configured compressed navigation mode."""
        mode = self.state.algorithm_parameters.acceptance_mode
        if mode == "fixed_target_full_e2":
            return {2: 1}
        if mode == "fixed_target_multiscale":
            weights = {2: 2}
            if self.state.L % 4 == 0:
                weights[4] = 4
            return weights
        return None


def save_best_candidate(path: Path, state: SearchState) -> Path:
    """Atomically persist the global best profile immediately after improvement."""
    profile = full_correlation_profile(state.best_a, state.best_b)
    payload = {
        "L": state.L,
        "a": list(state.best_a),
        "b": list(state.best_b),
        "score": state.best_score,
        "seed": state.seed,
        "iteration": state.iteration,
        "restart_index": state.restart_index,
        "elapsed_seconds": state.elapsed_seconds,
        "correlation_profile": profile,
    }
    atomic_write_json(Path(path), payload)
    return Path(path)


_SOLUTION_WRITE_LOCK = threading.RLock()


def append_verified_solution_if_new(
    L: int,
    a: Tuple[int, ...],
    b: Tuple[int, ...],
    project_root: Path,
) -> bool:
    """Serialize verification/deduplication/append across C and completion threads.

    Atomic replacement alone does not protect the read-modify-write operation:
    two threads could read the same old file and overwrite each other's result.
    This lock protects writers in this process, not independent search processes.
    """
    with _SOLUTION_WRITE_LOCK:
        return _append_verified_solution_if_new(L, a, b, project_root)


def _append_verified_solution_if_new(
    L: int,
    a: Tuple[int, ...],
    b: Tuple[int, ...],
    project_root: Path,
) -> bool:
    """Atomically append a new independently verified PQCP to the project's L.txt.

    Existing recorded A/B pairs are compared exactly before appending.  This
    preserves the project's established text format and never overwrites prior
    discoveries; the caller may then continue searching from a fresh restart.
    """
    verification = verify_pqcp(a, b)
    if not verification.is_valid:
        raise SearchImplementationError("refusing to record a verifier-failing solution")
    path = Path(project_root) / "{}.txt".format(L)
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    recorded = set(re.findall(r"^a=([01]+)\nb=([01]+)$", existing, flags=re.MULTILINE))
    pair = ("".join(map(str, a)), "".join(map(str, b)))
    # The Project 2 pair autocorrelation sum is unchanged when A and B are
    # exchanged, so (A, B) and (B, A) represent the same recorded PQCP.
    if pair in recorded or (pair[1], pair[0]) in recorded:
        return False
    nonzero = tuple(shift for shift in range(1, L) if verification.profile[shift] != 0)
    # Project files record the one value shared by the symmetric pair u,L-u,
    # rather than repeating it once for each actual nonzero shift.
    value = verification.profile[nonzero[0]]
    block = (
        "\nL={}\nnonzero shifts={}\nnonzero PACS={}\na={}\nb={}\n".format(
            L, ",".join(map(str, nonzero)), value, pair[0], pair[1]
        )
    )
    _atomic_write_text(path, existing + block)
    return True


def _restart_initialization(L: int, seed: int, restart_index: int, parameters: SearchParameters):
    """Derive deterministic restart seed, choose bounded FKM pair, and create move RNG."""
    restart_seed = seed + restart_index
    if parameters.target_content_profiles is not None:
        index = _target_profile_index(
            seed,
            restart_index,
            len(parameters.target_content_profiles),
            parameters.randomize_target_profiles,
            parameters.target_profile_offset,
        )
        values = parameters.target_content_profiles[index]
        profile = TargetContentProfile(L, *values)
        a, b = initialize_from_fkm_content_profile(
            profile,
            seed=restart_seed,
            pool_size=parameters.fkm_pool_size,
            policy=parameters.fkm_seed_policy,
            candidate_count=parameters.fkm_candidate_count,
            elite_count=parameters.fkm_elite_count,
        )
    elif parameters.weight_pairs is None:
        a, b = initialize_from_fkm(L, weight=parameters.weight, seed=restart_seed, pool_size=parameters.fkm_pool_size)
    else:
        weight_a, weight_b = parameters.weight_pairs[restart_index % len(parameters.weight_pairs)]
        a, b = initialize_from_fkm_weights(L, weight_a, weight_b, seed=restart_seed, pool_size=parameters.fkm_pool_size)
    return a, b, random.Random(restart_seed)


def _target_profile_index(
    seed: int,
    restart_index: int,
    profile_count: int,
    randomized: bool,
    offset: int = 0,
) -> int:
    """Choose a reproducible profile, distributing independent seed streams.

    The legacy path cycles from index zero.  The randomized path makes one
    stable seed-specific permutation and cycles through it, so every profile
    is visited exactly once per block without coupling this choice to SA RNG.
    """
    if not randomized:
        return (restart_index + offset) % profile_count
    digest = hashlib.sha256(
        "pqcp2-target-profile-order-v1|{}|{}".format(seed, profile_count).encode("ascii")
    ).digest()
    order = list(range(profile_count))
    random.Random(int.from_bytes(digest, "big")).shuffle(order)
    return order[(restart_index + offset) % profile_count]


def _atomic_write_text(path: Path, content: str) -> None:
    """Atomically replace a text result file without leaving partial output."""
    payload_path = Path(path)
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = payload_path.with_name(".{}.tmp".format(payload_path.name))
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(payload_path)
