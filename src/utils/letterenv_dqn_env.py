"""Environment construction utilities for LetterEnv DQN experiments."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.utils.legacy_paths import code_root, ensure_legacy_python_path, legacy_python_root
from src.utils.monitor_state_encoding import (
    build_numerical_event_index,
    build_one_hot_event_index,
    encode_numerical_monitor_state,
    encode_one_hot_monitor_state,
    extract_events,
    extract_numerical_values,
    normalize_monitor_state,
    replace_numerical_parts,
    split_top_level_factors,
)
from src.utils.letterenv_thesis_protocol import encoding_entry_point, load_monitor_state_catalogue

ensure_legacy_python_path()

from gymnasium.envs.registration import register, registry  # type: ignore  # noqa: E402
from rml.rmlgym import RMLGym, RMLGym_One_Hot, RMLGym_Simple  # type: ignore  # noqa: E402
from utils.encoding_functions import (  # type: ignore  # noqa: E402
    create_encoding,
    create_encoding_one_hot,
    generate_events_and_index,
    generate_events_and_index_one_hot,
)
SUPPORTED_ENCODINGS = ("simple", "one_hot", "numerical")

RUNTIME_COMPATIBLE_INITIAL_SIGNATURE = (
    "(star(not_abcd:eps)*var(n,(a_match(var(n)):eps)*app(gen([n],star(not_abcd:eps)*((b_match:eps)"
    "*app(gen([n],star(not_abcd:eps)*((c_match:eps)*app(,[var(n)]))),[var(n)]))),[var(n)])),"
    "[=gen([n],guarded(var(n)>0,star(not_abcd:eps)*((d_match:eps)*app(,[var(n)-1])),1))])",
)
RUNTIME_COMPATIBLE_ONE_HOT_B_APP_SIGNATURE = (
    "(app(gen([n],star(not_abcd:eps)*((b_match:eps)*app(gen([n],star(not_abcd:eps)*((c_match:eps)"
    "*app(,[var(n)]))),[var(n)]))),[{num}]),[=gen([n],guarded(var(n)>0,star(not_abcd:eps)*"
    "((d_match:eps)*app(,[var(n)-1])),1))])",
)
RUNTIME_COMPATIBLE_ONE_HOT_B_STAR_SIGNATURE = (
    "(star(not_abcd:eps)*((b_match:eps)*app(gen([n],star(not_abcd:eps)*((c_match:eps)*app(,[var(n)]))),"
    "[{num}])),[=gen([n],guarded(var(n)>0,star(not_abcd:eps)*((d_match:eps)*app(,[var(n)-1])),1))])",
)
RUNTIME_COMPATIBLE_ONE_HOT_C_SIGNATURE = (
    "(star(not_abcd:eps)*((c_match:eps)*app(gen([n],),[{num}])),[=guarded(var(n)>0,star(not_abcd:eps)"
    "*((d_match:eps)*app(gen([n],),[var(n)-1])),1)])",
)
RUNTIME_COMPATIBLE_ONE_HOT_D_SIGNATURE = (
    "(star(not_abcd:eps)*((d_match:eps)*app(gen([n],),[{num}])),[=guarded(var(n)>0,star(not_abcd:eps)"
    "*((d_match:eps)*app(gen([n],),[var(n)-1])),1)])",
)


@dataclass(frozen=True)
class LetterEnvDQNEnvConfig:
    """Configuration for constructing a LetterEnv DQN environment."""

    encoding: str
    config_path: Path
    n_value: int
    add_state_discovery_bonus: bool = False
    state_discovery_bonus: float = 2.0
    monitor_progress_bonus: float = 0.0
    monitor_regression_penalty: float = 0.0
    neutralize_legacy_transition_bonus: bool = False
    legacy_transition_bonus: float = 10.0
    step_penalty: float = 0.0
    no_op_penalty: float = 0.0
    simple_monitor_limit: int = 256


@contextmanager
def legacy_working_directory():
    """Temporarily execute from the inherited Python project root."""
    current_directory = Path.cwd()
    try:
        os.chdir(legacy_python_root())
        yield
    finally:
        os.chdir(current_directory)


def _validate_encoding(encoding: str) -> None:
    if encoding not in SUPPORTED_ENCODINGS:
        raise ValueError(f"Unsupported DQN encoding: {encoding}")


def _register_letterenv_variant(encoding: str) -> None:
    env_id = "letter-env"
    if env_id in registry:
        del registry[env_id]

    register(
        id=env_id,
        entry_point=encoding_entry_point(encoding),
        max_episode_steps=200,
    )


def _build_catalogue_state_lookup(
    encoding: str,
) -> tuple[dict[str, int], dict[str, np.ndarray], np.ndarray, dict[tuple[str, ...], np.ndarray] | None]:
    states_for_encoding = load_monitor_state_catalogue()

    if encoding == "one_hot":
        _, event_index = generate_events_and_index_one_hot(states_for_encoding)
        state_lookup = {
            normalize_monitor_state(state): create_encoding_one_hot(state, event_index).astype(np.float32)
            for state in states_for_encoding.values()
        }
        signature_lookup: dict[tuple[str, ...], np.ndarray] | None = {}
        for state in states_for_encoding.values():
            signature = tuple(extract_events(state))
            vector = create_encoding_one_hot(state, event_index).astype(np.float32)
            if signature in signature_lookup:
                if not np.array_equal(signature_lookup[signature], vector):
                    raise RuntimeError(
                        "Inconsistent legacy one-hot vectors for thesis signature "
                        f"{signature!r}."
                    )
            else:
                signature_lookup[signature] = vector
        target_signatures = {
            "initial": tuple(extract_events(states_for_encoding[0])),
            "a_or_b": tuple(extract_events(states_for_encoding[1])),
            "c_pending": tuple(extract_events(states_for_encoding[11])),
            "d_pending": tuple(extract_events(states_for_encoding[15])),
        }
        runtime_aliases = {
            RUNTIME_COMPATIBLE_INITIAL_SIGNATURE: target_signatures["initial"],
            RUNTIME_COMPATIBLE_ONE_HOT_B_APP_SIGNATURE: target_signatures["a_or_b"],
            RUNTIME_COMPATIBLE_ONE_HOT_B_STAR_SIGNATURE: target_signatures["a_or_b"],
            RUNTIME_COMPATIBLE_ONE_HOT_C_SIGNATURE: target_signatures["c_pending"],
            RUNTIME_COMPATIBLE_ONE_HOT_D_SIGNATURE: target_signatures["d_pending"],
        }
        for runtime_signature, target_signature in runtime_aliases.items():
            signature_lookup[runtime_signature] = signature_lookup[target_signature].copy()
    elif encoding == "numerical":
        _, event_index = generate_events_and_index(states_for_encoding)
        state_lookup = {
            normalize_monitor_state(state): create_encoding(state, event_index).astype(np.float32)
            for state in states_for_encoding.values()
        }
        signature_lookup = None
    else:
        raise ValueError(f"Unsupported catalogue lookup encoding: {encoding}")

    initial_state = normalize_monitor_state(states_for_encoding[0])
    return event_index, state_lookup, state_lookup[initial_state].copy(), signature_lookup


def _build_monitor_resources(
    encoding: str,
) -> tuple[
    dict[str, int] | None,
    np.ndarray | None,
    dict[str, np.ndarray] | None,
    dict[tuple[str, ...], np.ndarray] | None,
]:
    if encoding == "simple":
        return None, None, None, None

    if encoding == "one_hot":
        event_index, state_lookup, initial_encoding, signature_lookup = _build_catalogue_state_lookup(encoding)
        return event_index, initial_encoding, state_lookup, signature_lookup

    event_index, state_lookup, initial_encoding, signature_lookup = _build_catalogue_state_lookup(encoding)
    return event_index, initial_encoding, state_lookup, signature_lookup


def build_runtime_compatible_monitor_resources(
    encoding: str,
) -> tuple[
    dict[str, int] | None,
    np.ndarray | None,
    dict[str, np.ndarray] | None,
    dict[tuple[str, ...], np.ndarray] | None,
]:
    """Expose runtime-compatible monitor resources for non-DQN experiments."""
    return _build_monitor_resources(encoding)


def _set_environment_n(env: Any, n_value: int) -> None:
    candidates = [
        getattr(env, "env", None),
        getattr(getattr(env, "env", None), "unwrapped", None),
        getattr(env, "unwrapped", None),
        env,
    ]
    for candidate in candidates:
        if candidate is not None and hasattr(candidate, "set_n"):
            candidate.set_n(n_value)
            return
    raise AttributeError("Unable to locate set_n on the constructed LetterEnv wrapper.")


def _stabilize_legacy_position_encoding(env: Any) -> None:
    legacy_grid_env = getattr(env, "unwrapped", None)
    if legacy_grid_env is None:
        raise AttributeError("Unable to locate the inherited GridEnv wrapper.")
    if not hasattr(legacy_grid_env, "propositions"):
        raise AttributeError("The inherited GridEnv wrapper does not expose propositions.")
    if not hasattr(legacy_grid_env, "one_hot_propositions"):
        raise AttributeError("The inherited GridEnv wrapper does not expose one-hot propositions.")
    if not hasattr(legacy_grid_env, "env"):
        raise AttributeError("The inherited GridEnv wrapper does not expose the base LetterEnv.")

    base_env = legacy_grid_env.env
    propositions = list(dict.fromkeys(prop for prop in list(base_env.propositions) if prop != "_"))
    base_env.propositions = propositions.copy()
    base_env.prop_idx = {prop: index for index, prop in enumerate(base_env.propositions)}
    legacy_grid_env.propositions = base_env.propositions.copy()
    legacy_grid_env.propositions.append("_")
    legacy_grid_env.one_hot_propositions = legacy_grid_env.generate_one_hot_propisition(
        legacy_grid_env.propositions
    )


def _as_position_array(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float32)


def _as_monitor_array(value: Any, encoding: str) -> np.ndarray:
    if encoding == "simple":
        return np.asarray([value], dtype=np.float32)
    return np.asarray(value, dtype=np.float32)


def _state_key(observation: dict[str, np.ndarray]) -> tuple[tuple[float, ...], tuple[float, ...]]:
    position = tuple(np.asarray(observation["position"], dtype=np.float32).round(6).tolist())
    monitor = tuple(np.asarray(observation["monitor"], dtype=np.float32).round(6).tolist())
    return position, monitor


def _monitor_progress_potential(raw_monitor_state: str | None) -> float:
    """Return a monotone-like progress score for LetterEnv monitor states.

    The score is based on coarse task phases, not arbitrary monitor-state changes:
    initial/counting < after B < after C < D countdown < accept.
    """
    if raw_monitor_state is None:
        return 0.0

    normalized_state = normalize_monitor_state(raw_monitor_state)
    if normalized_state == "false_verdict":
        return -1000.0
    if normalized_state == "1":
        return 1000.0

    factors = [replace_numerical_parts(factor) for factor in split_top_level_factors(normalized_state)]
    values: list[float] = []
    for factor in split_top_level_factors(normalized_state):
        factor_values = extract_numerical_values(factor)
        if factor_values:
            values.extend(factor_values)
    primary_value = values[0] if values else 0.0

    if any("star(not_abcd:eps)*((d_match:eps)*app(gen([n],),[{num}]))" in factor for factor in factors):
        return 400.0 - primary_value
    if any("(app(gen([n],),[{num}]),[=guarded(var(n)>0" in factor for factor in factors):
        return 350.0 - primary_value
    if any("star(not_abcd:eps)*((c_match:eps)*app(gen([n],),[{num}]))" in factor for factor in factors):
        return 250.0 + primary_value
    if any("app(gen([n],star(not_abcd:eps)*((c_match:eps)*app(,[var(n)]))),[{num}])" in factor for factor in factors):
        return 150.0 + primary_value
    if any("(star(not_abcd:eps)*app(,[{num}]))" in factor for factor in factors):
        return primary_value
    if any(
        "star(not_abcd:eps)*(app(gen([n],),[{num}])\\/app(gen([n],(b_match:eps)*app(gen([n],star(not_abcd:eps)*((c_match:eps)*app(,[var(n)]))),[var(n)])),[{num}]))"
        in factor
        for factor in factors
    ):
        return 50.0 + primary_value
    return 0.0


class LetterEnvObservationAdapter(gym.ObservationWrapper):
    """Normalize inherited LetterEnv observations for SB3 DQN."""

    def __init__(self, env: gym.Env, encoding: str, simple_monitor_limit: int) -> None:
        super().__init__(env)
        self.encoding = encoding
        self.simple_monitor_limit = simple_monitor_limit

        sample_observation, _ = self.env.reset()
        position = _as_position_array(sample_observation["position"])
        monitor = _as_monitor_array(sample_observation["monitor"], encoding)

        monitor_high = (
            np.full((1,), float(simple_monitor_limit - 1), dtype=np.float32)
            if encoding == "simple"
            else np.full(monitor.shape, np.inf, dtype=np.float32)
        )

        self.observation_space = spaces.Dict(
            {
                "position": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=position.shape,
                    dtype=np.float32,
                ),
                "monitor": spaces.Box(
                    low=np.zeros(monitor.shape, dtype=np.float32),
                    high=monitor_high,
                    shape=monitor.shape,
                    dtype=np.float32,
                ),
            }
        )

    def observation(self, observation: dict[str, Any]) -> dict[str, np.ndarray]:
        return {
            "position": _as_position_array(observation["position"]),
            "monitor": _as_monitor_array(observation["monitor"], self.encoding),
        }


class StateDiscoveryRewardWrapper(gym.Wrapper):
    """Reproduce the tabular first-visit reward bonus for DQN training."""

    def __init__(self, env: gym.Env, bonus: float) -> None:
        super().__init__(env)
        self.bonus = float(bonus)
        self._seen_state_keys: set[tuple[tuple[float, ...], tuple[float, ...]]] = set()

    def reset_state_tracking(self) -> None:
        self._seen_state_keys.clear()

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self._seen_state_keys.add(_state_key(observation))
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        key = _state_key(observation)
        if key not in self._seen_state_keys:
            reward += self.bonus
            self._seen_state_keys.add(key)
        return observation, reward, terminated, truncated, info


class MonitorProgressRewardWrapper(gym.Wrapper):
    """Reward only forward monitor progress instead of any monitor-state change."""

    def __init__(
        self,
        env: gym.Env,
        *,
        progress_bonus: float,
        regression_penalty: float,
        neutralize_legacy_transition_bonus: bool,
        legacy_transition_bonus: float,
    ) -> None:
        super().__init__(env)
        self.progress_bonus = float(progress_bonus)
        self.regression_penalty = float(regression_penalty)
        self.neutralize_legacy_transition_bonus = bool(neutralize_legacy_transition_bonus)
        self.legacy_transition_bonus = float(legacy_transition_bonus)
        self._previous_raw_monitor_state: str | None = None
        self._previous_progress_potential = 0.0

    def _current_raw_monitor_state(self) -> str | None:
        return getattr(self.env, "monitor_state_unencoded", None)

    @staticmethod
    def _terminal_progress_potential(terminated: bool, reward_before_wrapper: float) -> float | None:
        if not terminated:
            return None
        if reward_before_wrapper > 0.0:
            return 1000.0
        if reward_before_wrapper < 0.0:
            return -1000.0
        return None

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        # RMLGym leaves monitor_state_unencoded stale across reset(), so start
        # progress tracking from a neutral baseline for each episode.
        self._previous_raw_monitor_state = None
        self._previous_progress_potential = 0.0
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        current_raw_monitor_state = self._current_raw_monitor_state()
        reward_before_wrapper = float(reward)
        terminal_progress_potential = self._terminal_progress_potential(
            bool(terminated),
            reward_before_wrapper,
        )
        current_progress_potential = (
            terminal_progress_potential
            if terminal_progress_potential is not None
            else _monitor_progress_potential(current_raw_monitor_state)
        )

        removed_transition_bonus = 0.0
        if (
            self.neutralize_legacy_transition_bonus
            and self._previous_raw_monitor_state is not None
            and current_raw_monitor_state is not None
            and normalize_monitor_state(current_raw_monitor_state)
            != normalize_monitor_state(self._previous_raw_monitor_state)
        ):
            removed_transition_bonus = self.legacy_transition_bonus

        applied_bonus = 0.0
        applied_regression_penalty = 0.0
        if current_progress_potential > self._previous_progress_potential:
            applied_bonus = self.progress_bonus
        elif current_progress_potential < self._previous_progress_potential:
            applied_regression_penalty = self.regression_penalty

        shaped_reward = (
            reward_before_wrapper
            - removed_transition_bonus
            + applied_bonus
            + applied_regression_penalty
        )
        self._previous_raw_monitor_state = current_raw_monitor_state
        self._previous_progress_potential = current_progress_potential

        info = dict(info)
        info.setdefault("base_reward", reward_before_wrapper)
        info["neutralized_legacy_transition_bonus"] = -removed_transition_bonus
        info["monitor_progress_bonus"] = applied_bonus
        info["monitor_regression_penalty"] = applied_regression_penalty
        info["reward_before_monitor_progress_bonus"] = reward_before_wrapper
        info["shaped_reward"] = shaped_reward
        return observation, shaped_reward, terminated, truncated, info


class ActionPenaltyWrapper(gym.Wrapper):
    """Apply optional step and no-op penalties while preserving the base reward in ``info``."""

    def __init__(self, env: gym.Env, *, step_penalty: float, no_op_penalty: float) -> None:
        super().__init__(env)
        self.step_penalty = float(step_penalty)
        self.no_op_penalty = float(no_op_penalty)
        self._last_observation: dict[str, np.ndarray] | None = None

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self._last_observation = {
            "position": _as_position_array(observation["position"]),
            "monitor": _as_monitor_array(observation["monitor"], "simple"),
        }
        return observation, info

    def step(self, action):
        previous_observation = self._last_observation
        observation, reward, terminated, truncated, info = self.env.step(action)
        shaped_reward = float(reward) + self.step_penalty

        applied_no_op_penalty = 0.0
        if (
            previous_observation is not None
            and self.no_op_penalty != 0.0
            and self._same_position(previous_observation, observation)
        ):
            shaped_reward += self.no_op_penalty
            applied_no_op_penalty = self.no_op_penalty

        self._last_observation = {
            "position": _as_position_array(observation["position"]),
            "monitor": _as_monitor_array(observation["monitor"], "simple"),
        }

        info = dict(info)
        info["base_reward"] = float(reward)
        info["step_penalty"] = self.step_penalty
        info["no_op_penalty"] = applied_no_op_penalty
        info["shaped_reward"] = shaped_reward
        return observation, shaped_reward, terminated, truncated, info

    @staticmethod
    def _same_position(previous_observation: dict[str, np.ndarray], observation: dict[str, np.ndarray]) -> bool:
        previous_position = tuple(np.asarray(previous_observation["position"], dtype=np.float32)[:2].tolist())
        current_position = tuple(np.asarray(observation["position"], dtype=np.float32)[:2].tolist())
        return previous_position == current_position


class RuntimeCompatibleRMLGymOneHot(RMLGym_One_Hot):
    """One-hot RMLGym variant with runtime-compatible monitor-state parsing."""

    def __init__(
        self,
        event_index: dict[str, int],
        initial_monitor_state_encoding,
        config_path: str,
        state_encoding_lookup: dict[str, np.ndarray],
        signature_lookup: dict[tuple[str, ...], np.ndarray],
    ):
        super().__init__(event_index, initial_monitor_state_encoding, config_path)
        self.event_index = event_index
        self.state_encoding_lookup = state_encoding_lookup
        self.signature_lookup = {
            signature: vector.astype(np.float32).copy()
            for signature, vector in signature_lookup.items()
        }
        self.unknown_monitor_signatures: dict[str, int] = {}
        self.unknown_monitor_states: dict[str, int] = {}

    def transform_monitor_state(self, monitor_state: str):
        normalized = normalize_monitor_state(monitor_state)
        if normalized not in self.monitor_states.values():
            self.monitor_states[self.mon_number] = normalized
            self.mon_number += 1
        if normalized in self.state_encoding_lookup:
            return self.state_encoding_lookup[normalized].copy()
        signature = tuple(extract_events(normalized))
        vector = self.signature_lookup.get(signature)
        if vector is not None:
            return vector.copy()

        signature_key = json.dumps(signature)
        self.unknown_monitor_signatures[signature_key] = (
            self.unknown_monitor_signatures.get(signature_key, 0) + 1
        )
        self.unknown_monitor_states[normalized] = self.unknown_monitor_states.get(normalized, 0) + 1
        return encode_one_hot_monitor_state(normalized, self.event_index)

    def get_monitor_diagnostics(self) -> dict[str, Any]:
        return {
            "unknown_monitor_signature_count": int(sum(self.unknown_monitor_signatures.values())),
            "unknown_monitor_state_count": int(sum(self.unknown_monitor_states.values())),
            "unknown_monitor_signatures": self.unknown_monitor_signatures,
            "unknown_monitor_states": self.unknown_monitor_states,
        }


class RuntimeCompatibleRMLGymNumerical(RMLGym):
    """Numerical RMLGym variant with runtime-compatible monitor-state parsing."""

    def __init__(
        self,
        event_index: dict[str, int],
        initial_monitor_state_encoding,
        config_path: str,
        state_encoding_lookup: dict[str, np.ndarray],
    ):
        super().__init__(event_index, initial_monitor_state_encoding, config_path)
        self.event_index = event_index
        self.state_encoding_lookup = state_encoding_lookup

    def transform_monitor_state(self, monitor_state: str):
        normalized = normalize_monitor_state(monitor_state)
        if normalized not in self.monitor_states.values():
            self.monitor_states[self.mon_number] = normalized
            self.mon_number += 1
        if normalized in self.state_encoding_lookup:
            return self.state_encoding_lookup[normalized].copy()
        return encode_numerical_monitor_state(normalized, self.event_index)


def build_runtime_compatible_tabular_env(
    *,
    encoding: str,
    config_path: Path,
) -> Any:
    """Construct a runtime-compatible LetterEnv monitor wrapper for tabular experiments."""
    _validate_encoding(encoding)
    if encoding == "simple":
        env = RMLGym_Simple(str(config_path))
    else:
        event_index, initial_monitor_encoding, state_encoding_lookup, signature_lookup = (
            build_runtime_compatible_monitor_resources(
            encoding
            )
        )
        assert event_index is not None
        assert initial_monitor_encoding is not None
        assert state_encoding_lookup is not None
        if encoding == "one_hot":
            assert signature_lookup is not None
            env = RuntimeCompatibleRMLGymOneHot(
                event_index,
                initial_monitor_encoding,
                str(config_path),
                state_encoding_lookup,
                signature_lookup,
            )
        else:
            env = RuntimeCompatibleRMLGymNumerical(
                event_index,
                initial_monitor_encoding,
                str(config_path),
                state_encoding_lookup,
            )
    _stabilize_legacy_position_encoding(env)
    return env


def build_letterenv_dqn_env(
    config: LetterEnvDQNEnvConfig,
    *,
    evaluation: bool = False,
) -> gym.Env:
    """Construct a LetterEnv environment for a DQN experiment."""
    _validate_encoding(config.encoding)

    with legacy_working_directory():
        _register_letterenv_variant(config.encoding)
        env: gym.Env = build_runtime_compatible_tabular_env(
            encoding=config.encoding,
            config_path=config.config_path,
        )

    _set_environment_n(env, config.n_value)
    adapted_env: gym.Env = LetterEnvObservationAdapter(
        env,
        encoding=config.encoding,
        simple_monitor_limit=config.simple_monitor_limit,
    )

    if config.step_penalty != 0.0 or config.no_op_penalty != 0.0:
        adapted_env = ActionPenaltyWrapper(
            adapted_env,
            step_penalty=config.step_penalty,
            no_op_penalty=config.no_op_penalty,
        )

    if (
        config.monitor_progress_bonus != 0.0
        or config.monitor_regression_penalty != 0.0
        or config.neutralize_legacy_transition_bonus
    ):
        adapted_env = MonitorProgressRewardWrapper(
            adapted_env,
            progress_bonus=config.monitor_progress_bonus,
            regression_penalty=config.monitor_regression_penalty,
            neutralize_legacy_transition_bonus=config.neutralize_legacy_transition_bonus,
            legacy_transition_bonus=config.legacy_transition_bonus,
        )

    if config.add_state_discovery_bonus and not evaluation:
        adapted_env = StateDiscoveryRewardWrapper(adapted_env, config.state_discovery_bonus)

    return adapted_env


def default_dqn_results_root() -> Path:
    """Return the standard output root for LetterEnv DQN runs."""
    return code_root() / "src" / "results" / "dqn_letterenv"


def collect_monitor_diagnostics(env: Any) -> dict[str, Any] | None:
    """Return monitor diagnostics from a wrapped environment when available."""
    current = env
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if hasattr(current, "get_monitor_diagnostics"):
            return current.get_monitor_diagnostics()
        current = getattr(current, "env", None)
    return None
