"""Exact sampled fixed-weight escapes layered on weight-preserving swap SA."""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
from time import perf_counter
from typing import Any, Dict, Optional, Tuple

from .annealing import initialize_from_fkm, initialize_from_fkm_weights
from .checkpoint import CheckpointError, atomic_write_json
from .compression import CorrelationState
from .moves import (
    WeightPreservingSwap,
    apply_weight_preserving_swap,
    choose_weight_preserving_swap,
    hamming_weights,
    rollback_weight_preserving_swap,
)
from .verifier import verify_pqcp


@dataclass(frozen=True)
class EnhancedParameters:
    """Controls for exact fixed-weight SA plus sampled stagnation escapes."""

    initial_temperature: float = 8.0
    cooling_rate: float = 0.999
    min_temperature: float = 0.05
    fkm_pool_size: int = 128
    weight: Optional[int] = None
    weight_pairs: Optional[Tuple[Tuple[int, int], ...]] = None
    stagnation_iterations: int = 25_000
    two_bit_samples: int = 128
    escape_policy: str = "best_sampled"
    max_escape_fraction: float = 0.25
    max_escapes_per_restart: int = 4
    random_kick_strength: int = 0

    def __post_init__(self) -> None:
        """Validate bounded, deterministic escape controls."""
        if self.initial_temperature <= 0 or self.min_temperature <= 0 or self.min_temperature > self.initial_temperature:
            raise ValueError("temperatures must be positive and ordered")
        if not 0 < self.cooling_rate <= 1:
            raise ValueError("cooling_rate must be in (0, 1]")
        if self.stagnation_iterations <= 0 or self.two_bit_samples < 0:
            raise ValueError("stagnation_iterations must be positive and samples non-negative")
        if self.weight_pairs is not None and (
            not self.weight_pairs or any(
                not isinstance(pair, tuple) or len(pair) != 2 or
                any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in pair)
                for pair in self.weight_pairs
            )
        ):
            raise ValueError("weight_pairs must be non-empty pairs of non-negative integers or None")
        if self.escape_policy not in ("best_sampled", "annealed_sampled"):
            raise ValueError("escape_policy must be best_sampled or annealed_sampled")
        if not 0 <= self.max_escape_fraction <= 1:
            raise ValueError("max_escape_fraction must be in [0, 1]")
        if self.max_escapes_per_restart <= 0 or self.random_kick_strength not in (0, 2):
            raise ValueError("max_escapes_per_restart must be positive and kick strength 0 or 2")


@dataclass(frozen=True)
class TwoBitMove:
    """One same-sequence two-bit move used as a zero/one exchange."""

    kind: str
    p: int
    q: int

    def __post_init__(self) -> None:
        if self.kind not in ("AA", "BB"):
            raise ValueError("weight-preserving two-bit kind must be AA or BB")
        if self.p == self.q:
            raise ValueError("AA and BB moves require distinct positions")


@dataclass
class EnhancedState:
    """Full deterministic state, including escape counters, for JSON resume."""

    L: int
    current_a: Tuple[int, ...]
    current_b: Tuple[int, ...]
    current_score: int
    best_a: Tuple[int, ...]
    best_b: Tuple[int, ...]
    best_score: int
    iteration: int
    restart_index: int
    temperature: float
    rng_state: Tuple[Any, ...]
    seed: int
    elapsed_seconds: float
    last_improvement_iteration: int
    last_escape_iteration: int
    escape_triggers: int
    sampled_moves: int
    accepted_escapes: int
    rejected_escapes: int
    escape_evaluation_seconds: float
    escapes_followed_by_global_best: int
    pending_escape_improvement: bool
    escapes_since_restart: int
    algorithm_parameters: EnhancedParameters
    finished: bool = False

    def to_dict(self) -> Dict[str, Any]:
        rng_provider = getattr(self, "_rng_provider", None)
        if rng_provider is not None:
            self.rng_state = rng_provider.getstate()
        data = asdict(self)
        data["rng_state"] = _jsonify(self.rng_state)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EnhancedState":
        try:
            data = dict(data)
            data["current_a"] = tuple(data["current_a"])
            data["current_b"] = tuple(data["current_b"])
            data["best_a"] = tuple(data["best_a"])
            data["best_b"] = tuple(data["best_b"])
            data["rng_state"] = _tupleify(data["rng_state"])
            parameters = dict(data["algorithm_parameters"])
            if parameters.get("weight_pairs") is not None:
                parameters["weight_pairs"] = tuple(tuple(pair) for pair in parameters["weight_pairs"])
            data["algorithm_parameters"] = EnhancedParameters(**parameters)
            state = cls(**data)
        except (KeyError, TypeError, ValueError) as error:
            raise CheckpointError("invalid enhanced checkpoint state: {}".format(error)) from error
        if state.L <= 0 or len(state.current_a) != state.L or len(state.current_b) != state.L:
            raise CheckpointError("enhanced checkpoint sequence length mismatch")
        return state


@dataclass(frozen=True)
class EnhancedRunResult:
    """Comparable bounded-run result and exact escape accounting."""

    state: EnhancedState
    elapsed_time: float
    verified: bool


class EnhancedSearch:
    """Weight-preserving swap SA with sampled fixed-weight plateau escapes."""

    def __init__(self, state: EnhancedState) -> None:
        self.state = state
        self._correlation = CorrelationState(state.current_a, state.current_b)
        if self._correlation.score != state.current_score:
            raise ValueError("enhanced checkpoint objective is inconsistent")
        self._rng = random.Random()
        self._rng.setstate(state.rng_state)
        self.state._rng_provider = self._rng

    @classmethod
    def new(cls, L: int, seed: int, parameters: EnhancedParameters) -> "EnhancedSearch":
        a, b = _weighted_fkm_initialization(L, seed, 0, parameters)
        correlation = CorrelationState(a, b)
        score = correlation.score
        rng = random.Random(seed)
        return cls(EnhancedState(
            L, a, b, score, a, b, score, 0, 0, parameters.initial_temperature,
            rng.getstate(), seed, 0.0, 0, -parameters.stagnation_iterations,
            0, 0, 0, 0, 0.0, 0, False, 0, parameters,
        ))

    @classmethod
    def resume(cls, path: Path) -> "EnhancedSearch":
        return cls(load_enhanced_checkpoint(path))

    def trial_two_bit_move(self, move: TwoBitMove) -> int:
        """Evaluate a two-bit move exactly, then restore bits/profile/objective exactly."""
        original_a, original_b, original_profile, original_score = (
            self._correlation.a, self._correlation.b, self._correlation.profile, self.state.current_score
        )
        self._apply_move(move)
        score = self._correlation.score
        self._rollback_move(move)
        assert (self._correlation.a, self._correlation.b, self._correlation.profile, self.state.current_score) == (
            original_a, original_b, original_profile, original_score
        )
        return score

    def step(self) -> bool:
        """Perform one regular fixed-weight swap and possibly one sampled escape."""
        if self.state.finished:
            return False
        old_best = self.state.best_score
        weights_before = hamming_weights(self._correlation.a, self._correlation.b)
        move = choose_weight_preserving_swap(self._correlation.a, self._correlation.b, self._rng)
        if move is not None:
            apply_weight_preserving_swap(self._correlation, move)
            proposed_score = self._correlation.score
            delta = proposed_score - self.state.current_score
            accepted = delta <= 0 or self._rng.random() < math.exp(-delta / self.state.temperature)
            if accepted:
                self._set_current_from_correlation(proposed_score)
            else:
                rollback_weight_preserving_swap(self._correlation, move)
        if hamming_weights(self._correlation.a, self._correlation.b) != weights_before:
            raise RuntimeError("weight-preserving enhanced move changed a Hamming weight")
        self.state.iteration += 1
        self._record_best_if_improved()
        self.state.temperature = max(self.state.algorithm_parameters.min_temperature,
                                     self.state.temperature * self.state.algorithm_parameters.cooling_rate)
        if self._is_stagnant():
            self._handle_stagnation()
        if self.state.current_score == 0:
            verification = verify_pqcp(self.state.current_a, self.state.current_b)
            if not verification.is_valid:
                raise RuntimeError("score-zero enhanced candidate failed independent verification")
            self.state.finished = True
        return self.state.best_score < old_best

    def run_steps(self, steps: int) -> None:
        """Run a deterministic count of steps, used for resume-equivalence testing."""
        for _ in range(steps):
            self.step()

    def run(self, seconds: float, clock=perf_counter) -> EnhancedRunResult:
        """Run to a wall-clock budget; decisions remain RNG/iteration deterministic."""
        started = clock()
        last = started
        verified = False
        if self.state.current_score == 0:
            verification = verify_pqcp(self.state.current_a, self.state.current_b)
            if not verification.is_valid:
                raise RuntimeError("score-zero enhanced initial candidate failed independent verification")
            self.state.best_a, self.state.best_b, self.state.best_score = (
                self.state.current_a, self.state.current_b, 0
            )
            self.state.finished = True
            return EnhancedRunResult(self.state, 0.0, True)
        while not self.state.finished and clock() - started < seconds:
            self.step()
            now = clock()
            self.state.elapsed_seconds += max(0.0, now - last)
            last = now
            if self.state.best_score == 0:
                verification = verify_pqcp(self.state.best_a, self.state.best_b)
                if not verification.is_valid:
                    raise RuntimeError("score-zero enhanced candidate failed independent verification")
                verified = True
                self.state.finished = True
        return EnhancedRunResult(self.state, clock() - started, verified)

    def _handle_stagnation(self) -> None:
        parameters = self.state.algorithm_parameters
        if parameters.two_bit_samples == 0:
            self._restart()
            return
        self.state.escape_triggers += 1
        self.state.last_escape_iteration = self.state.iteration
        if not self._within_escape_budget(parameters.two_bit_samples):
            self.state.rejected_escapes += 1
            return
        started = perf_counter()
        best_move, best_score = self._sample_best_move(parameters.two_bit_samples)
        self.state.escape_evaluation_seconds += perf_counter() - started
        self.state.sampled_moves += parameters.two_bit_samples
        accepted = best_score <= self.state.current_score
        if not accepted and parameters.escape_policy == "best_sampled":
            accepted = True
        elif not accepted and parameters.escape_policy == "annealed_sampled":
            accepted = self._rng.random() < math.exp(-(best_score - self.state.current_score) / self.state.temperature)
        if accepted:
            self._apply_move(best_move)
            self._set_current_from_correlation(best_score)
            self.state.accepted_escapes += 1
            self.state.escapes_since_restart += 1
            self.state.pending_escape_improvement = True
            self._record_best_if_improved()
        else:
            self.state.rejected_escapes += 1
            if parameters.random_kick_strength == 2:
                # Optional bounded kick is exactly one sampled two-bit move;
                # it is never a 3/4-bit or exhaustive neighborhood operation.
                kick = self._random_two_bit_move()
                self._apply_move(kick)
                self._set_current_from_correlation(self._correlation.score)
                self.state.accepted_escapes += 1
                self.state.escapes_since_restart += 1
                self.state.pending_escape_improvement = True
                self._record_best_if_improved()
        if self.state.escapes_since_restart >= parameters.max_escapes_per_restart:
            self._restart()

    def _sample_best_move(self, samples: int) -> Tuple[TwoBitMove, int]:
        move = self._random_two_bit_move()
        best_move, best_score = move, self.trial_two_bit_move(move)
        for _ in range(samples - 1):
            move = self._random_two_bit_move()
            score = self.trial_two_bit_move(move)
            if score < best_score:
                best_move, best_score = move, score
        return best_move, best_score

    def _random_two_bit_move(self) -> TwoBitMove:
        move = choose_weight_preserving_swap(self._correlation.a, self._correlation.b, self._rng)
        if move is None:
            raise ValueError("current pair has no weight-preserving two-bit move")
        return TwoBitMove("AA" if move.sequence == "a" else "BB", move.zero_position, move.one_position)

    def _apply_move(self, move: TwoBitMove) -> None:
        swap = self._as_swap(move)
        apply_weight_preserving_swap(self._correlation, swap)

    def _rollback_move(self, move: TwoBitMove) -> None:
        # In the applied state the zero/one orientation is reversed; applying
        # that reverse exchange restores the original bits and profile.
        apply_weight_preserving_swap(self._correlation, self._as_swap(move))

    def _as_swap(self, move: TwoBitMove) -> WeightPreservingSwap:
        values = self._correlation.a if move.kind == "AA" else self._correlation.b
        if values[move.p] == values[move.q]:
            raise ValueError("enhanced move must exchange one zero and one one")
        zero_position, one_position = (move.p, move.q) if values[move.p] == 0 else (move.q, move.p)
        return WeightPreservingSwap("a" if move.kind == "AA" else "b", zero_position, one_position)

    def _set_current_from_correlation(self, score: int) -> None:
        self.state.current_a = self._correlation.a
        self.state.current_b = self._correlation.b
        self.state.current_score = score

    def _record_best_if_improved(self) -> None:
        if self.state.current_score < self.state.best_score:
            self.state.best_a = self.state.current_a
            self.state.best_b = self.state.current_b
            self.state.best_score = self.state.current_score
            self.state.last_improvement_iteration = self.state.iteration
            if self.state.pending_escape_improvement:
                self.state.escapes_followed_by_global_best += 1
                self.state.pending_escape_improvement = False

    def _is_stagnant(self) -> bool:
        p = self.state.algorithm_parameters
        return (self.state.iteration - self.state.last_improvement_iteration >= p.stagnation_iterations
                and self.state.iteration - self.state.last_escape_iteration >= p.stagnation_iterations)

    def _within_escape_budget(self, samples: int) -> bool:
        p = self.state.algorithm_parameters
        if p.max_escape_fraction == 0:
            return False
        total_after = self.state.iteration + self.state.sampled_moves + samples
        return (self.state.sampled_moves + samples) / total_after <= p.max_escape_fraction

    def _restart(self) -> None:
        self.state.restart_index += 1
        restart_seed = self.state.seed + self.state.restart_index
        a, b = _weighted_fkm_initialization(
            self.state.L, self.state.seed, self.state.restart_index, self.state.algorithm_parameters
        )
        self._correlation = CorrelationState(a, b)
        self._rng = random.Random(restart_seed)
        self.state._rng_provider = self._rng
        self.state.current_a, self.state.current_b = a, b
        self.state.current_score = self._correlation.score
        self.state.temperature = self.state.algorithm_parameters.initial_temperature
        self.state.last_improvement_iteration = self.state.iteration
        self.state.last_escape_iteration = self.state.iteration
        self.state.escapes_since_restart = 0
        self.state.pending_escape_improvement = False
        if self.state.current_score < self.state.best_score:
            self._record_best_if_improved()


def _weighted_fkm_initialization(L: int, seed: int, restart_index: int, parameters: EnhancedParameters):
    """Use an optional safe ordered weight schedule only for FKM starts."""
    restart_seed = seed + restart_index
    if parameters.weight_pairs is None:
        return initialize_from_fkm(L, weight=parameters.weight, seed=restart_seed, pool_size=parameters.fkm_pool_size)
    weight_a, weight_b = parameters.weight_pairs[restart_index % len(parameters.weight_pairs)]
    return initialize_from_fkm_weights(L, weight_a, weight_b, seed=restart_seed, pool_size=parameters.fkm_pool_size)


def save_enhanced_checkpoint(path: Path, state: EnhancedState) -> Path:
    """Atomically save enhanced escape state using the existing JSON I/O primitive."""
    atomic_write_json(Path(path), {"schema_version": 1, "kind": "enhanced_search", "state": state.to_dict()})
    return Path(path)


def load_enhanced_checkpoint(path: Path) -> EnhancedState:
    """Safely restore enhanced JSON state without pickle or implicit code execution."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CheckpointError("cannot read enhanced checkpoint: {}".format(error)) from error
    if not isinstance(payload, dict) or payload.get("kind") != "enhanced_search" or payload.get("schema_version") != 1:
        raise CheckpointError("unsupported enhanced checkpoint")
    return EnhancedState.from_dict(payload.get("state"))


def _jsonify(value):
    if isinstance(value, tuple):
        return [_jsonify(item) for item in value]
    if isinstance(value, list):
        return [_jsonify(item) for item in value]
    return value


def _tupleify(value):
    return tuple(_tupleify(item) for item in value) if isinstance(value, list) else value
