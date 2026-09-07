"""Versioned, JSON-only checkpoint data and atomic persistence utilities."""

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, Optional, Tuple


SCHEMA_VERSION = 1


class CheckpointError(ValueError):
    """Raised when a checkpoint is missing, malformed, or incompatible."""


@dataclass(frozen=True)
class SearchParameters:
    """Serializable controls for the long-running fixed-weight swap SA move."""

    initial_temperature: float = 8.0
    cooling_rate: float = 0.999
    min_temperature: float = 0.05
    temperature_schedule: str = "geometric"
    metropolis_scale: float = 1.0
    fkm_pool_size: int = 128
    fkm_seed_policy: str = "legacy"
    fkm_candidate_count: int = 16
    fkm_elite_count: int = 4
    weight: Optional[int] = None
    weight_pairs: Optional[Tuple[Tuple[int, int], ...]] = None
    stagnation_iterations: Optional[int] = 100_000
    kick_stagnation_iterations: Optional[int] = None
    kick_swaps: int = 3
    max_iterations_per_restart: Optional[int] = None
    max_restarts: Optional[int] = None
    acceptance_mode: str = "objective"
    proposal_samples: int = 1
    objective_energy_weight: int = 2
    target_content_profiles: Optional[Tuple[Tuple[int, int, int, int, int, int], ...]] = None
    preserve_alternating_content: bool = False
    randomize_target_profiles: bool = False
    target_profile_offset: int = 0

    def __post_init__(self) -> None:
        """Validate controls without imposing any Project 2 weight assumption."""
        if self.initial_temperature <= 0 or self.min_temperature <= 0:
            raise ValueError("temperatures must be positive")
        if self.min_temperature > self.initial_temperature:
            raise ValueError("min_temperature must not exceed initial_temperature")
        if not 0 < self.cooling_rate <= 1:
            raise ValueError("cooling_rate must be in (0, 1]")
        if self.temperature_schedule not in ("geometric", "linear_restart"):
            raise ValueError("temperature_schedule must be geometric or linear_restart")
        if (
            not isinstance(self.metropolis_scale, (int, float))
            or isinstance(self.metropolis_scale, bool)
            or not self.metropolis_scale > 0
        ):
            raise ValueError("metropolis_scale must be positive")
        if self.acceptance_mode not in (
            "objective", "target_pair_squared", "objective_plus_target_pair",
            "fixed_target_full", "fixed_target_full_compressed_tiebreak",
            "fixed_target_full_e2", "fixed_target_multiscale",
        ):
            raise ValueError(
                "unsupported acceptance_mode"
            )
        if not isinstance(self.fkm_pool_size, int) or isinstance(self.fkm_pool_size, bool) or self.fkm_pool_size <= 0:
            raise ValueError("fkm_pool_size must be a positive integer")
        if self.fkm_seed_policy not in (
            "legacy", "phase_random", "phase_top_q", "compressed_top_q",
            "compressed_a_random_b_top_q",
        ):
            raise ValueError(
                "fkm_seed_policy must be legacy, phase_random, phase_top_q, "
                "compressed_top_q, or compressed_a_random_b_top_q"
            )
        for name, value in (
            ("fkm_candidate_count", self.fkm_candidate_count),
            ("fkm_elite_count", self.fkm_elite_count),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError("{} must be a positive integer".format(name))
        if (
            self.fkm_seed_policy in (
                "phase_top_q", "compressed_top_q",
                "compressed_a_random_b_top_q",
            )
            and self.fkm_elite_count > self.fkm_candidate_count
        ):
            raise ValueError("fkm_elite_count must not exceed fkm_candidate_count")
        if (
            not isinstance(self.proposal_samples, int)
            or isinstance(self.proposal_samples, bool)
            or self.proposal_samples <= 0
        ):
            raise ValueError("proposal_samples must be a positive integer")
        if (
            not isinstance(self.objective_energy_weight, int)
            or isinstance(self.objective_energy_weight, bool)
            or self.objective_energy_weight <= 0
        ):
            raise ValueError("objective_energy_weight must be a positive integer")
        if self.weight_pairs is not None:
            if not self.weight_pairs or any(
                not isinstance(pair, tuple) or len(pair) != 2 or
                any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in pair)
                for pair in self.weight_pairs
            ):
                raise ValueError("weight_pairs must be non-empty pairs of non-negative integers or None")
        if self.target_content_profiles is not None:
            if not self.target_content_profiles or any(
                not isinstance(profile, tuple) or len(profile) != 6 or
                any(not isinstance(value, int) or isinstance(value, bool) for value in profile)
                for profile in self.target_content_profiles
            ):
                raise ValueError("target_content_profiles must contain integer 6-tuples")
        if not isinstance(self.preserve_alternating_content, bool):
            raise ValueError("preserve_alternating_content must be boolean")
        if not isinstance(self.randomize_target_profiles, bool):
            raise ValueError("randomize_target_profiles must be boolean")
        if self.randomize_target_profiles and self.target_content_profiles is None:
            raise ValueError("randomized target profiles require target_content_profiles")
        if (
            not isinstance(self.target_profile_offset, int)
            or isinstance(self.target_profile_offset, bool)
            or self.target_profile_offset < 0
        ):
            raise ValueError("target_profile_offset must be a non-negative integer")
        if self.target_profile_offset and self.target_content_profiles is None:
            raise ValueError("target profile offset requires target_content_profiles")
        if self.preserve_alternating_content and self.target_content_profiles is None:
            raise ValueError("preserving alternating content requires target_content_profiles")
        for name, value in (
            ("stagnation_iterations", self.stagnation_iterations),
            ("kick_stagnation_iterations", self.kick_stagnation_iterations),
            ("max_iterations_per_restart", self.max_iterations_per_restart),
            ("max_restarts", self.max_restarts),
        ):
            if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value <= 0):
                raise ValueError("{} must be a positive integer or None".format(name))
        if (
            not isinstance(self.kick_swaps, int)
            or isinstance(self.kick_swaps, bool)
            or self.kick_swaps <= 0
        ):
            raise ValueError("kick_swaps must be a positive integer")
        if self.temperature_schedule == "linear_restart" and self.max_iterations_per_restart is None:
            raise ValueError("linear_restart requires max_iterations_per_restart")

    def to_dict(self) -> Dict[str, Any]:
        """Return JSON-compatible parameters."""
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "SearchParameters":
        """Restore validated parameters from a checkpoint object."""
        if not isinstance(value, dict):
            raise CheckpointError("algorithm_parameters must be an object")
        try:
            restored = dict(value)
            if restored.get("weight_pairs") is not None:
                restored["weight_pairs"] = tuple(tuple(pair) for pair in restored["weight_pairs"])
            if restored.get("target_content_profiles") is not None:
                restored["target_content_profiles"] = tuple(
                    tuple(profile) for profile in restored["target_content_profiles"]
                )
            return cls(**restored)
        except (TypeError, ValueError) as error:
            raise CheckpointError("invalid algorithm_parameters: {}".format(error)) from error


@dataclass
class SearchState:
    """All deterministic trajectory state needed for save-and-resume."""

    L: int
    current_a: Tuple[int, ...]
    current_b: Tuple[int, ...]
    current_score: int
    best_a: Tuple[int, ...]
    best_b: Tuple[int, ...]
    best_score: int
    iteration: int
    restart_index: int
    restart_start_iteration: int
    restart_start_score: int
    restart_local_best_score: int
    temperature: float
    rng_state: Tuple[Any, ...]
    seed: int
    elapsed_seconds: float
    last_improvement_iteration: int
    stagnation_count: int
    algorithm_parameters: SearchParameters
    restart_best_energy: Optional[int] = None
    restart_last_energy_improvement_iteration: Optional[int] = None
    finished: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to a strict JSON-safe representation."""
        rng_provider = getattr(self, "_rng_provider", None)
        if rng_provider is not None:
            # The live runner avoids copying Python's large MT state every
            # iteration.  Serialization is the exact synchronization point
            # required for deterministic checkpoint/resume.
            self.rng_state = rng_provider.getstate()
        return {
            "L": self.L,
            "current_a": list(self.current_a),
            "current_b": list(self.current_b),
            "current_score": self.current_score,
            "best_a": list(self.best_a),
            "best_b": list(self.best_b),
            "best_score": self.best_score,
            "iteration": self.iteration,
            "restart_index": self.restart_index,
            "restart_start_iteration": self.restart_start_iteration,
            "restart_start_score": self.restart_start_score,
            "restart_local_best_score": self.restart_local_best_score,
            "temperature": self.temperature,
            "rng_state": _jsonify_tuple(self.rng_state),
            "seed": self.seed,
            "elapsed_seconds": self.elapsed_seconds,
            "last_improvement_iteration": self.last_improvement_iteration,
            "stagnation_count": self.stagnation_count,
            "algorithm_parameters": self.algorithm_parameters.to_dict(),
            "restart_best_energy": self.restart_best_energy,
            "restart_last_energy_improvement_iteration": self.restart_last_energy_improvement_iteration,
            "finished": self.finished,
        }

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "SearchState":
        """Restore and validate required state fields from JSON data."""
        required = {
            "L", "current_a", "current_b", "current_score", "best_a", "best_b", "best_score",
            "iteration", "restart_index", "restart_start_iteration", "temperature", "rng_state",
            "seed", "elapsed_seconds", "last_improvement_iteration", "stagnation_count",
            "algorithm_parameters", "finished",
        }
        if not isinstance(value, dict) or not required <= set(value):
            raise CheckpointError("checkpoint state is missing required fields")
        try:
            state = cls(
                L=value["L"],
                current_a=_bits(value["current_a"]),
                current_b=_bits(value["current_b"]),
                current_score=_integer(value["current_score"], "current_score"),
                best_a=_bits(value["best_a"]),
                best_b=_bits(value["best_b"]),
                best_score=_integer(value["best_score"], "best_score"),
                iteration=_nonnegative(value["iteration"], "iteration"),
                restart_index=_nonnegative(value["restart_index"], "restart_index"),
                restart_start_iteration=_nonnegative(value["restart_start_iteration"], "restart_start_iteration"),
                # Added in checkpoint 8.5; old checkpoint files retain usable
                # deterministic move state and get conservative log defaults.
                restart_start_score=_integer(value.get("restart_start_score", value["current_score"]), "restart_start_score"),
                restart_local_best_score=_integer(value.get("restart_local_best_score", value["current_score"]), "restart_local_best_score"),
                temperature=float(value["temperature"]),
                rng_state=_tupleify(value["rng_state"]),
                seed=_integer(value["seed"], "seed"),
                elapsed_seconds=float(value["elapsed_seconds"]),
                last_improvement_iteration=_nonnegative(value["last_improvement_iteration"], "last_improvement_iteration"),
                stagnation_count=_nonnegative(value["stagnation_count"], "stagnation_count"),
                algorithm_parameters=SearchParameters.from_dict(value["algorithm_parameters"]),
                restart_best_energy=(
                    None if value.get("restart_best_energy") is None
                    else _integer(value["restart_best_energy"], "restart_best_energy")
                ),
                restart_last_energy_improvement_iteration=(
                    None if value.get("restart_last_energy_improvement_iteration") is None
                    else _nonnegative(
                        value["restart_last_energy_improvement_iteration"],
                        "restart_last_energy_improvement_iteration",
                    )
                ),
                finished=bool(value["finished"]),
            )
        except (TypeError, ValueError) as error:
            raise CheckpointError("invalid checkpoint state: {}".format(error)) from error
        if not isinstance(state.L, int) or isinstance(state.L, bool) or state.L <= 0:
            raise CheckpointError("L must be a positive integer")
        if len(state.current_a) != state.L or len(state.current_b) != state.L:
            raise CheckpointError("current sequences do not have length L")
        if len(state.best_a) != state.L or len(state.best_b) != state.L:
            raise CheckpointError("best sequences do not have length L")
        return state


def save_checkpoint(path: Path, state: SearchState) -> Path:
    """Atomically save a versioned search state without pickle."""
    payload = {"schema_version": SCHEMA_VERSION, "state": state.to_dict()}
    _atomic_write_json(Path(path), payload)
    return Path(path)


def load_checkpoint(path: Path) -> SearchState:
    """Load a versioned JSON checkpoint and reject corrupt/untrusted formats."""
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise CheckpointError("cannot read checkpoint {}: {}".format(path, error)) from error
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise CheckpointError("unsupported or missing checkpoint schema version")
    return SearchState.from_dict(payload.get("state"))


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Atomically write JSON by fsyncing a sibling temporary file then renaming."""
    _atomic_write_json(Path(path), payload)


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Implement atomic JSON replace in the target directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".{}.".format(path.name), suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def _bits(value: Any) -> Tuple[int, ...]:
    """Validate a JSON binary sequence."""
    if not isinstance(value, list) or not value or any(bit not in (0, 1) or isinstance(bit, bool) for bit in value):
        raise ValueError("expected a non-empty binary sequence")
    return tuple(value)


def _integer(value: Any, name: str) -> int:
    """Require a non-boolean integer JSON field."""
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError("{} must be an integer".format(name))
    return value


def _nonnegative(value: Any, name: str) -> int:
    """Require a nonnegative integer JSON field."""
    value = _integer(value, name)
    if value < 0:
        raise ValueError("{} must be nonnegative".format(name))
    return value


def _jsonify_tuple(value: Any) -> Any:
    """Recursively turn tuples into JSON arrays."""
    if isinstance(value, tuple):
        return [_jsonify_tuple(item) for item in value]
    if isinstance(value, list):
        return [_jsonify_tuple(item) for item in value]
    return value


def _tupleify(value: Any) -> Any:
    """Recursively restore random.Random's tuple-oriented state."""
    if isinstance(value, list):
        return tuple(_tupleify(item) for item in value)
    return value
