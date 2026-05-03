"""Reproduce the inherited tabular LetterEnv experiment protocol."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np
import pandas as pd

from contextlib import contextmanager

from src.utils.legacy_paths import code_root, ensure_legacy_python_path, legacy_python_root
from src.utils.monitor_state_encoding import (
    extract_events,
    extract_numerical_values,
    normalize_monitor_state,
)
from src.utils.letterenv_dqn_env import build_runtime_compatible_tabular_env
from src.utils.letterenv_thesis_protocol import (
    PROTOCOL,
    encoding_entry_point,
    load_monitor_state_catalogue,
    protocol_as_dict,
)

ensure_legacy_python_path()

from envs.letterenv import Actions  # type: ignore  # noqa: E402
from gymnasium.envs.registration import register, registry  # type: ignore  # noqa: E402
from rml.rmlgym import RMLGym, RMLGym_One_Hot  # type: ignore  # noqa: E402
from utils.encoding_functions import (  # type: ignore  # noqa: E402
    create_encoding,
    create_encoding_one_hot,
    generate_events_and_index,
    generate_events_and_index_one_hot,
)
ENCODING_CHOICES = ("simple", "one_hot", "numerical", "all")


@dataclass(frozen=True)
class ExperimentRecord:
    """Single convergence measurement for one iteration and one ``n`` value."""

    encoding: str
    iteration: int
    n_value: int
    episodes: int
    steps: int
    seed: int | None
    converged: bool


@dataclass(frozen=True)
class EncodingProgress:
    """Progress snapshot for one encoding run."""

    encoding: str
    total_conditions: int
    completed_conditions: int
    current_iteration: int | None
    current_n_value: int | None
    started_at_utc: str
    updated_at_utc: str
    elapsed_seconds: float
    current_condition_episodes: int | None
    current_condition_steps: int | None


class ThesisFaithfulRMLGymOneHot(RMLGym_One_Hot):
    """Encode current runtime monitor strings onto the inherited thesis one-hot basis."""

    def __init__(
        self,
        event_index: dict[str, int],
        initial_monitor_state_encoding: np.ndarray,
        config_path: str,
        signature_to_vector: dict[tuple[str, ...], np.ndarray],
    ) -> None:
        super().__init__(event_index, initial_monitor_state_encoding, config_path)
        self.signature_to_vector = {
            signature: vector.astype(np.float32).copy()
            for signature, vector in signature_to_vector.items()
        }
        self.unknown_monitor_signatures: dict[str, int] = {}
        self.unknown_monitor_states: dict[str, int] = {}

    def transform_monitor_state(self, monitor_state):
        monitor_state = re.sub(r"_[0-9]+", "", monitor_state)
        if monitor_state not in self.monitor_states.values():
            self.monitor_states[self.mon_number] = monitor_state
            self.mon_number += 1

        signature = tuple(extract_events(monitor_state))
        vector = self.signature_to_vector.get(signature)
        if vector is not None:
            return vector.copy()

        signature_key = json.dumps(signature)
        self.unknown_monitor_signatures[signature_key] = (
            self.unknown_monitor_signatures.get(signature_key, 0) + 1
        )
        self.unknown_monitor_states[monitor_state] = self.unknown_monitor_states.get(monitor_state, 0) + 1
        return create_encoding_one_hot(monitor_state, self.event_index)

    def get_monitor_diagnostics(self) -> dict[str, Any]:
        return {
            "unknown_monitor_signature_count": int(sum(self.unknown_monitor_signatures.values())),
            "unknown_monitor_state_count": int(sum(self.unknown_monitor_states.values())),
            "unknown_monitor_signatures": self.unknown_monitor_signatures,
            "unknown_monitor_states": self.unknown_monitor_states,
        }


class ThesisFaithfulRMLGymNumerical(RMLGym):
    """Encode current runtime monitor strings onto the inherited thesis numerical basis."""

    def __init__(
        self,
        event_index: dict[str, int],
        initial_monitor_state_encoding: np.ndarray,
        config_path: str,
        state_lookup: dict[str, np.ndarray],
        runtime_signature_to_vector: dict[tuple[str, ...], np.ndarray],
        template_by_signature: dict[tuple[str, ...], tuple[str, str]],
    ) -> None:
        super().__init__(event_index, initial_monitor_state_encoding, config_path)
        self.state_lookup = {
            state: vector.astype(np.float32).copy()
            for state, vector in state_lookup.items()
        }
        self.runtime_signature_to_vector = {
            signature: vector.astype(np.float32).copy()
            for signature, vector in runtime_signature_to_vector.items()
        }
        self.template_by_signature = dict(template_by_signature)
        self.unknown_monitor_signatures: dict[str, int] = {}
        self.unknown_monitor_states: dict[str, int] = {}

    def transform_monitor_state(self, monitor_state):
        monitor_state = re.sub(r"_[0-9]+", "", monitor_state)
        if monitor_state not in self.monitor_states.values():
            self.monitor_states[self.mon_number] = monitor_state
            self.mon_number += 1

        if monitor_state in self.state_lookup:
            return self.state_lookup[monitor_state].copy()

        signature = tuple(extract_events(monitor_state))
        values = extract_numerical_values(monitor_state) or []
        template = self.template_by_signature.get(signature)
        if template is not None and values:
            template_state, placeholder = template
            template_state = template_state.replace(placeholder, f"[{values[0]}]")
            return create_encoding(template_state, self.event_index).astype(np.float32)

        vector = self.runtime_signature_to_vector.get(signature)
        if vector is not None:
            return vector.copy()

        signature_key = json.dumps(signature)
        self.unknown_monitor_signatures[signature_key] = (
            self.unknown_monitor_signatures.get(signature_key, 0) + 1
        )
        self.unknown_monitor_states[monitor_state] = self.unknown_monitor_states.get(monitor_state, 0) + 1
        return create_encoding(monitor_state, self.event_index)

    def get_monitor_diagnostics(self) -> dict[str, Any]:
        return {
            "unknown_monitor_signature_count": int(sum(self.unknown_monitor_signatures.values())),
            "unknown_monitor_state_count": int(sum(self.unknown_monitor_states.values())),
            "unknown_monitor_signatures": self.unknown_monitor_signatures,
            "unknown_monitor_states": self.unknown_monitor_states,
        }


RUNTIME_COMPATIBLE_INITIAL_SIGNATURE = (
    "(star(not_abcd:eps)*var(n,(a_match(var(n)):eps)*app(gen([n],star(not_abcd:eps)*((b_match:eps)"
    "*app(gen([n],star(not_abcd:eps)*((c_match:eps)*app(,[var(n)]))),[var(n)]))),[var(n)])),"
    "[=gen([n],guarded(var(n)>0,star(not_abcd:eps)*((d_match:eps)*app(,[var(n)-1])),1))])",
)
RUNTIME_COMPATIBLE_NUMERICAL_B_APP_SIGNATURE = (
    "(app(gen([n],star(not_abcd:eps)*((b_match:eps)*app(gen([n],star(not_abcd:eps)*((c_match:eps)"
    "*app(,[var(n)]))),[var(n)]))),[{num}]),[=gen([n],guarded(var(n)>0,star(not_abcd:eps)*"
    "((d_match:eps)*app(,[var(n)-1])),1))])",
)
RUNTIME_COMPATIBLE_NUMERICAL_B_STAR_SIGNATURE = (
    "(star(not_abcd:eps)*((b_match:eps)*app(gen([n],star(not_abcd:eps)*((c_match:eps)*app(,[var(n)]))),"
    "[{num}])),[=gen([n],guarded(var(n)>0,star(not_abcd:eps)*((d_match:eps)*app(,[var(n)-1])),1))])",
)
RUNTIME_COMPATIBLE_NUMERICAL_B_SIGNATURES = {
    RUNTIME_COMPATIBLE_NUMERICAL_B_APP_SIGNATURE,
    RUNTIME_COMPATIBLE_NUMERICAL_B_STAR_SIGNATURE,
}
RUNTIME_COMPATIBLE_NUMERICAL_C_SIGNATURE = (
    "(star(not_abcd:eps)*((c_match:eps)*app(gen([n],),[{num}])),[=guarded(var(n)>0,star(not_abcd:eps)"
    "*((d_match:eps)*app(gen([n],),[var(n)-1])),1)])",
)
RUNTIME_COMPATIBLE_NUMERICAL_D_SIGNATURE = (
    "(star(not_abcd:eps)*((d_match:eps)*app(gen([n],),[{num}])),[=guarded(var(n)>0,star(not_abcd:eps)"
    "*((d_match:eps)*app(gen([n],),[var(n)-1])),1)])",
)


def add_runtime_compatible_one_hot_aliases(
    signature_to_vector: dict[tuple[str, ...], np.ndarray],
    states_for_encoding: dict[int, str],
) -> None:
    """Map runtime-compatible monitor signatures onto the inherited thesis vectors."""
    target_signatures = {
        "initial": tuple(extract_events(states_for_encoding[0])),
        "a_or_b": tuple(extract_events(states_for_encoding[1])),
        "c_pending": tuple(extract_events(states_for_encoding[11])),
        "d_pending": tuple(extract_events(states_for_encoding[15])),
    }

    runtime_aliases = {
        RUNTIME_COMPATIBLE_INITIAL_SIGNATURE: target_signatures["initial"],
        RUNTIME_COMPATIBLE_NUMERICAL_B_APP_SIGNATURE: target_signatures["a_or_b"],
        RUNTIME_COMPATIBLE_NUMERICAL_B_STAR_SIGNATURE: target_signatures["a_or_b"],
        RUNTIME_COMPATIBLE_NUMERICAL_C_SIGNATURE: target_signatures["c_pending"],
        RUNTIME_COMPATIBLE_NUMERICAL_D_SIGNATURE: target_signatures["d_pending"],
    }

    for runtime_signature, target_signature in runtime_aliases.items():
        signature_to_vector[runtime_signature] = signature_to_vector[target_signature].copy()


def add_runtime_compatible_numerical_aliases(
    runtime_signature_to_vector: dict[tuple[str, ...], np.ndarray],
    states_for_encoding: dict[int, str],
    event_index: dict[str, int],
) -> None:
    """Map runtime-compatible monitor signatures onto the inherited thesis numerical vectors."""
    target_vectors = {
        "initial": create_encoding(states_for_encoding[0], event_index).astype(np.float32),
        "c_pending": create_encoding(states_for_encoding[11], event_index).astype(np.float32),
        "d_pending": create_encoding(states_for_encoding[15], event_index).astype(np.float32),
    }
    runtime_signature_to_vector[RUNTIME_COMPATIBLE_INITIAL_SIGNATURE] = target_vectors["initial"]
    runtime_signature_to_vector[RUNTIME_COMPATIBLE_NUMERICAL_C_SIGNATURE] = target_vectors["c_pending"]
    runtime_signature_to_vector[RUNTIME_COMPATIBLE_NUMERICAL_D_SIGNATURE] = target_vectors["d_pending"]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce the inherited tabular LetterEnv protocol that precedes "
            "the DQN extension study."
        )
    )
    parser.add_argument(
        "--encoding",
        choices=ENCODING_CHOICES,
        default="all",
        help="Monitor-state encoding to evaluate.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=code_root() / "legacy" / "rml_reward_machines" / PROTOCOL.config_relative_path,
        help="Path to the YAML environment configuration used by the legacy wrapper.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory. If omitted, a timestamped directory is created under src/results/tabular_letterenv/.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=PROTOCOL.iterations,
        help="Number of repeated measurements per n value.",
    )
    parser.add_argument(
        "--max-n",
        type=int,
        default=PROTOCOL.max_n,
        help="Largest task parameter n included in the sweep.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=PROTOCOL.epsilon,
        help="Initial exploration rate.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=PROTOCOL.alpha,
        help="Tabular Q-learning step size.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=PROTOCOL.gamma,
        help="Discount factor.",
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=None,
        help=(
            "Optional base seed. When provided, each (encoding, iteration, n) "
            "condition is derived deterministically from this base."
        ),
    )
    parser.add_argument(
        "--max-episodes-per-condition",
        type=int,
        default=None,
        help=(
            "Optional upper bound on training episodes per condition. This is "
            "intended for bounded verification runs and should be omitted for "
            "thesis-protocol reproduction."
        ),
    )
    parser.add_argument(
        "--heartbeat-episodes",
        type=int,
        default=25,
        help="Episode interval for in-condition progress updates.",
    )
    parser.add_argument(
        "--heartbeat-seconds",
        type=float,
        default=30.0,
        help="Maximum time between in-condition progress updates.",
    )
    parser.add_argument(
        "--calibration",
        action="store_true",
        help="Use a smaller exploratory run profile to estimate runtime before the full sweep.",
    )
    return parser.parse_args()


def utc_now_iso() -> str:
    """Return the current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


def register_letterenv_variant(encoding: str) -> None:
    """Register the legacy LetterEnv wrapper needed for a given encoding."""
    env_id = PROTOCOL.environment_id
    if env_id in registry:
        del registry[env_id]

    register(
        id=env_id,
        entry_point=encoding_entry_point(encoding),
        max_episode_steps=PROTOCOL.max_episode_steps,
    )


def resolve_output_directory(requested: Path | None, encoding: str) -> Path:
    """Resolve the directory used to store experiment artifacts."""
    if requested is not None:
        return requested if requested.is_absolute() else code_root() / requested

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return code_root() / "src" / "results" / "tabular_letterenv" / encoding / timestamp


def records_to_frame(records: list[ExperimentRecord]) -> pd.DataFrame:
    """Convert experiment records into a tabular DataFrame."""
    return pd.DataFrame(
        [
            {
                "encoding": record.encoding,
                "iteration": record.iteration,
                "n_value": record.n_value,
                "episodes": record.episodes,
                "steps": record.steps,
                "seed": record.seed,
                "converged": record.converged,
            }
            for record in records
        ]
    )


def configure_seed(seed: int | None) -> None:
    """Configure Python and NumPy RNGs for a single experiment condition."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)


@contextmanager
def legacy_working_directory():
    """Execute a block from the inherited Python project root."""
    current_directory = Path.cwd()
    os.chdir(legacy_python_root())
    try:
        yield
    finally:
        os.chdir(current_directory)


def derive_seed(seed_base: int | None, encoding_index: int, iteration: int, n_value: int) -> int | None:
    """Derive a deterministic condition-specific seed from a base seed."""
    if seed_base is None:
        return None
    return seed_base + encoding_index * 10000 + iteration * 100 + n_value


def get_actions() -> list[int]:
    """Return the action list used in the inherited LetterEnv experiments."""
    return [
        Actions.RIGHT.value,
        Actions.LEFT.value,
        Actions.UP.value,
        Actions.DOWN.value,
    ]


def make_environment_factory(
    encoding: str,
    config_path: Path,
) -> tuple[Callable[[], Any], dict[str, Any]]:
    """Create the environment constructor for a chosen encoding."""
    if encoding == "one_hot":
        states_for_encoding = load_monitor_state_catalogue()
        _, event_index = generate_events_and_index_one_hot(states_for_encoding)
        initial_encoding = create_encoding_one_hot(states_for_encoding[0], event_index)
        signature_to_vector: dict[tuple[str, ...], np.ndarray] = {}
        for state in states_for_encoding.values():
            signature = tuple(extract_events(state))
            vector = create_encoding_one_hot(state, event_index).astype(np.float32)
            if signature in signature_to_vector:
                if not np.array_equal(signature_to_vector[signature], vector):
                    raise RuntimeError(
                        "Inconsistent legacy one-hot vectors for thesis signature "
                        f"{signature!r}."
                    )
            else:
                signature_to_vector[signature] = vector
        add_runtime_compatible_one_hot_aliases(signature_to_vector, states_for_encoding)
        return (
            lambda: ThesisFaithfulRMLGymOneHot(
                event_index,
                initial_encoding,
                str(config_path),
                signature_to_vector,
            ),
            {
                "environment_class": "ThesisFaithfulRMLGymOneHot",
                "event_index_size": len(event_index),
                "compatibility_signature_count": len(signature_to_vector),
                "thesis_faithful_monitor_encoding": True,
            },
        )

    if encoding == "numerical":
        states_for_encoding = load_monitor_state_catalogue()
        _, event_index = generate_events_and_index(states_for_encoding)
        initial_encoding = create_encoding(states_for_encoding[0], event_index).astype(np.float32)
        state_lookup = {
            normalize_monitor_state(state): create_encoding(state, event_index).astype(np.float32)
            for state in states_for_encoding.values()
        }
        runtime_signature_to_vector: dict[tuple[str, ...], np.ndarray] = {}
        template_by_signature = {
            RUNTIME_COMPATIBLE_NUMERICAL_B_APP_SIGNATURE: (states_for_encoding[1], "[0+1]"),
            RUNTIME_COMPATIBLE_NUMERICAL_B_STAR_SIGNATURE: (states_for_encoding[1], "[0+1]"),
            tuple(extract_events(states_for_encoding[10])): (states_for_encoding[10], "[1]"),
            tuple(extract_events(states_for_encoding[18])): (states_for_encoding[18], "[1]"),
        }
        add_runtime_compatible_numerical_aliases(
            runtime_signature_to_vector,
            states_for_encoding,
            event_index,
        )
        return (
            lambda: ThesisFaithfulRMLGymNumerical(
                event_index,
                initial_encoding,
                str(config_path),
                state_lookup,
                runtime_signature_to_vector,
                template_by_signature,
            ),
            {
                "environment_class": "ThesisFaithfulRMLGymNumerical",
                "event_index_size": len(event_index),
                "exact_state_lookup_count": len(state_lookup),
                "runtime_signature_alias_count": (
                    len(runtime_signature_to_vector) + len(RUNTIME_COMPATIBLE_NUMERICAL_B_SIGNATURES)
                ),
                "thesis_faithful_monitor_encoding": True,
            },
        )

    environment_class = {
        "simple": "RMLGym_Simple",
    }.get(encoding)
    if environment_class is None:
        raise ValueError(f"Unsupported encoding: {encoding}")

    return (
        lambda: build_runtime_compatible_tabular_env(
            encoding=encoding,
            config_path=config_path,
        ),
        {
            "environment_class": environment_class,
            "runtime_compatible_monitor_encoding": encoding != "simple",
        },
    )


def learning_episode_letter_runtime_compatible(
    rewards: list[float],
    env: Any,
    q_table: dict[Any, dict[int, float]],
    actions: list[int],
    alpha: float,
    gamma: float,
    epsilon: float,
    total_steps: int,
    n_value: int,
    reward_if_correct: tuple[int, int] = PROTOCOL.terminal_reward_targets,
) -> tuple[list[float], bool, dict[Any, dict[int, float]], Any, float, int]:
    """Run one tabular LetterEnv episode while treating truncation as terminal."""
    successful_policy = False
    env.env.set_n(n_value)
    state, _ = env.reset()

    if isinstance(state["monitor"], int):
        state_tuple = (state["position"], (state["monitor"]))
    else:
        state_tuple = (state["position"], tuple(state["monitor"]))

    done = False
    truncated = False
    while not done and not truncated:
        if state_tuple not in q_table:
            q_table[state_tuple] = {action: 0 for action in actions}

        if random.random() < epsilon:
            action = random.choice(actions)
        else:
            max_value = max(q_table[state_tuple].values())
            best_actions = [action for action in actions if q_table[state_tuple][action] == max_value]
            action = random.choice(best_actions)

        next_state, reward, done, truncated, _info = env.step(action)
        if isinstance(next_state["monitor"], int):
            next_state_tuple = (next_state["position"], (next_state["monitor"]))
        else:
            next_state_tuple = (next_state["position"], tuple(next_state["monitor"]))

        if next_state_tuple not in q_table:
            q_table[next_state_tuple] = {action: 0 for action in actions}
            reward += 2

        old_value = q_table[state_tuple][action]
        next_max = max(q_table[next_state_tuple].values())
        q_table[state_tuple][action] = old_value + alpha * (reward + gamma * next_max - old_value)

        state_tuple = next_state_tuple
        total_steps += 1

    epsilon *= PROTOCOL.epsilon_decay
    rewards.append(reward)
    if len(rewards) >= PROTOCOL.success_window:
        average_reward = sum(rewards[-PROTOCOL.success_window :]) / PROTOCOL.success_window
        if average_reward in reward_if_correct:
            successful_policy = True

    return rewards, successful_policy, q_table, state, epsilon, total_steps


def run_single_condition(
    *,
    encoding: str,
    config_path: Path,
    iteration: int,
    n_value: int,
    epsilon: float,
    alpha: float,
    gamma: float,
    seed: int | None,
    max_episodes_per_condition: int | None,
    episode_callback: Callable[[int, int, float, bool], None] | None = None,
 ) -> tuple[ExperimentRecord, dict[str, Any] | None]:
    """Run one inherited convergence measurement for a chosen encoding and ``n``."""
    configure_seed(seed)
    with legacy_working_directory():
        register_letterenv_variant(encoding)
        actions = get_actions()
        environment_factory, _metadata = make_environment_factory(encoding, config_path)
        env = environment_factory()

        q_table: dict[Any, dict[int, float]] = {}
        rewards: list[float] = []
        successful_policy = False
        total_steps = 0
        num_episodes = 0
        current_epsilon = epsilon

        while not successful_policy:
            if max_episodes_per_condition is not None and num_episodes >= max_episodes_per_condition:
                break
            num_episodes += 1
            rewards, successful_policy, q_table, _state, current_epsilon, total_steps = learning_episode_letter_runtime_compatible(
                rewards,
                env,
                q_table,
                actions,
                alpha,
                gamma,
                current_epsilon,
                total_steps,
                n_value,
            )
            if episode_callback is not None:
                episode_callback(num_episodes, total_steps, current_epsilon, successful_policy)

        env.close()

    diagnostics = None
    if hasattr(env, "get_monitor_diagnostics"):
        diagnostics = env.get_monitor_diagnostics()

    return (
        ExperimentRecord(
            encoding=encoding,
            iteration=iteration,
            n_value=n_value,
            episodes=num_episodes,
            steps=total_steps,
            seed=seed,
            converged=successful_policy,
        ),
        diagnostics,
    )


def summarise_records(results_df: pd.DataFrame) -> dict[str, Any]:
    """Build a compact JSON summary from the experiment table."""
    if results_df.empty:
        return {
            "row_count": 0,
            "n_values": [],
            "overall_converged_fraction": None,
            "aggregate_by_n": {},
        }

    summary: dict[str, Any] = {
        "row_count": int(len(results_df)),
        "n_values": sorted(int(value) for value in results_df["n_value"].unique()),
        "overall_converged_fraction": float(results_df["converged"].mean()),
        "aggregate_by_n": {},
    }

    grouped = results_df.groupby("n_value", sort=True)
    for n_value, group in grouped:
        summary["aggregate_by_n"][str(int(n_value))] = {
            "converged_fraction": float(group["converged"].mean()),
            "mean_episodes": float(group["episodes"].mean()),
            "std_episodes": float(group["episodes"].std(ddof=0)),
            "mean_steps": float(group["steps"].mean()),
            "std_steps": float(group["steps"].std(ddof=0)),
        }

    return summary


def write_run_artifacts(
    *,
    output_dir: Path,
    encoding: str,
    config_path: Path,
    results_df: pd.DataFrame,
    args: argparse.Namespace,
    environment_metadata: dict[str, Any],
    monitor_diagnostics: dict[str, Any] | None,
) -> None:
    """Persist the run table and its metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)

    train_metrics_path = output_dir / "train_metrics.csv"
    config_path_out = output_dir / "config.json"
    monitor_diagnostics_path = output_dir / "monitor_diagnostics.json"
    summary_path = output_dir / "summary.json"

    results_df.to_csv(train_metrics_path, index=False)

    configuration = {
        "experiment": "letterenv_tabular_baseline_reproduction",
        "encoding": encoding,
        "config_path": str(config_path),
        "protocol": protocol_as_dict(),
        "run_parameters": {
            "iterations": args.iterations,
            "max_n": args.max_n,
            "epsilon": args.epsilon,
            "alpha": args.alpha,
            "gamma": args.gamma,
            "seed_base": args.seed_base,
            "max_episodes_per_condition": args.max_episodes_per_condition,
        },
        "environment": environment_metadata,
        "legacy_dependencies": {
            "python_root": "legacy/rml_reward_machines",
            "monitor_root": "legacy/RML",
            "monitor_launcher": PROTOCOL.monitor_launcher,
            "monitor_spec": PROTOCOL.numerical_monitor_spec,
        },
        "convergence_rule": {
            "type": "moving_average_terminal_reward",
            "window_size": PROTOCOL.success_window,
            "target_rewards": list(PROTOCOL.terminal_reward_targets),
        },
    }
    config_path_out.write_text(json.dumps(configuration, indent=2), encoding="utf-8")

    diagnostics_payload = monitor_diagnostics or {
        "unknown_monitor_signature_count": 0,
        "unknown_monitor_state_count": 0,
        "unknown_monitor_signatures": {},
        "unknown_monitor_states": {},
    }
    monitor_diagnostics_path.write_text(json.dumps(diagnostics_payload, indent=2), encoding="utf-8")

    summary_payload = summarise_records(results_df)
    summary_payload["encoding"] = encoding
    summary_payload["output_dir"] = str(output_dir)
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")


def write_progress_artifact(
    *,
    output_dir: Path,
    progress: EncodingProgress,
    last_record: ExperimentRecord | None,
) -> None:
    """Persist progress information for a long-running encoding sweep."""
    progress_path = output_dir / "progress.json"
    progress_payload: dict[str, Any] = {
        "encoding": progress.encoding,
        "total_conditions": progress.total_conditions,
        "completed_conditions": progress.completed_conditions,
        "remaining_conditions": progress.total_conditions - progress.completed_conditions,
        "current_iteration": progress.current_iteration,
        "current_n_value": progress.current_n_value,
        "started_at_utc": progress.started_at_utc,
        "updated_at_utc": progress.updated_at_utc,
        "elapsed_seconds": progress.elapsed_seconds,
        "current_condition_episodes": progress.current_condition_episodes,
        "current_condition_steps": progress.current_condition_steps,
    }
    if last_record is not None:
        progress_payload["last_completed_condition"] = {
            "iteration": last_record.iteration,
            "n_value": last_record.n_value,
            "episodes": last_record.episodes,
            "steps": last_record.steps,
            "seed": last_record.seed,
            "converged": last_record.converged,
        }
    progress_path.write_text(json.dumps(progress_payload, indent=2), encoding="utf-8")


def log_progress(message: str) -> None:
    """Print a timestamped progress line."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{timestamp}] {message}", flush=True)


def merge_monitor_diagnostics(
    aggregate: dict[str, Any],
    diagnostics: dict[str, Any] | None,
) -> None:
    """Merge per-condition monitor diagnostics into the run aggregate."""
    if diagnostics is None:
        return

    aggregate["unknown_monitor_signature_count"] += diagnostics.get("unknown_monitor_signature_count", 0)
    aggregate["unknown_monitor_state_count"] += diagnostics.get("unknown_monitor_state_count", 0)

    for signature, count in diagnostics.get("unknown_monitor_signatures", {}).items():
        aggregate["unknown_monitor_signatures"][signature] = (
            aggregate["unknown_monitor_signatures"].get(signature, 0) + count
        )

    for state, count in diagnostics.get("unknown_monitor_states", {}).items():
        aggregate["unknown_monitor_states"][state] = (
            aggregate["unknown_monitor_states"].get(state, 0) + count
        )


def apply_calibration_profile(args: argparse.Namespace) -> None:
    """Apply a smaller runtime-estimation profile when requested."""
    if not args.calibration:
        return
    if args.iterations == PROTOCOL.iterations:
        args.iterations = 2
    if args.max_n == PROTOCOL.max_n:
        args.max_n = 3
    if args.max_episodes_per_condition is None:
        args.max_episodes_per_condition = 200


def run_encoding(args: argparse.Namespace, encoding: str, encoding_index: int) -> Path:
    """Run the baseline reproduction for a single encoding."""
    output_dir = resolve_output_directory(args.output_dir, encoding)
    environment_factory, environment_metadata = make_environment_factory(encoding, args.config)
    del environment_factory

    output_dir.mkdir(parents=True, exist_ok=True)
    total_conditions = args.iterations * args.max_n
    started_at_utc = utc_now_iso()
    started_epoch = time.time()
    records: list[ExperimentRecord] = []
    monitor_diagnostics = {
        "unknown_monitor_signature_count": 0,
        "unknown_monitor_state_count": 0,
        "unknown_monitor_signatures": {},
        "unknown_monitor_states": {},
    }
    write_run_artifacts(
        output_dir=output_dir,
        encoding=encoding,
        config_path=args.config,
        results_df=records_to_frame(records),
        args=args,
        environment_metadata=environment_metadata,
        monitor_diagnostics=monitor_diagnostics,
    )
    write_progress_artifact(
        output_dir=output_dir,
        progress=EncodingProgress(
            encoding=encoding,
            total_conditions=total_conditions,
            completed_conditions=0,
            current_iteration=1 if total_conditions > 0 else None,
            current_n_value=1 if total_conditions > 0 else None,
            started_at_utc=started_at_utc,
            updated_at_utc=started_at_utc,
            elapsed_seconds=0.0,
            current_condition_episodes=0,
            current_condition_steps=0,
        ),
        last_record=None,
    )
    log_progress(
        f"Starting encoding '{encoding}' with {total_conditions} conditions. "
        f"Output directory: {output_dir}"
    )

    for iteration in range(1, args.iterations + 1):
        for n_value in range(1, args.max_n + 1):
            condition_index = len(records) + 1
            write_progress_artifact(
                output_dir=output_dir,
                progress=EncodingProgress(
                    encoding=encoding,
                    total_conditions=total_conditions,
                    completed_conditions=len(records),
                    current_iteration=iteration,
                    current_n_value=n_value,
                    started_at_utc=started_at_utc,
                    updated_at_utc=utc_now_iso(),
                    elapsed_seconds=time.time() - started_epoch,
                    current_condition_episodes=0,
                    current_condition_steps=0,
                ),
                last_record=records[-1] if records else None,
            )
            log_progress(
                f"Encoding '{encoding}': starting condition {condition_index}/{total_conditions} "
                f"(iteration={iteration}, n={n_value})."
            )
            seed = derive_seed(args.seed_base, encoding_index, iteration, n_value)
            last_heartbeat_timestamp = time.time()

            def episode_callback(
                num_episodes: int,
                total_steps: int,
                current_epsilon: float,
                successful_policy: bool,
            ) -> None:
                nonlocal last_heartbeat_timestamp
                now = time.time()
                should_emit = successful_policy
                if not should_emit and args.heartbeat_episodes > 0 and num_episodes % args.heartbeat_episodes == 0:
                    should_emit = True
                if not should_emit and args.heartbeat_seconds > 0 and now - last_heartbeat_timestamp >= args.heartbeat_seconds:
                    should_emit = True
                if not should_emit:
                    return

                last_heartbeat_timestamp = now
                write_progress_artifact(
                    output_dir=output_dir,
                    progress=EncodingProgress(
                        encoding=encoding,
                        total_conditions=total_conditions,
                        completed_conditions=len(records),
                        current_iteration=iteration,
                        current_n_value=n_value,
                        started_at_utc=started_at_utc,
                        updated_at_utc=utc_now_iso(),
                        elapsed_seconds=now - started_epoch,
                        current_condition_episodes=num_episodes,
                        current_condition_steps=total_steps,
                    ),
                    last_record=records[-1] if records else None,
                )
                log_progress(
                    f"Encoding '{encoding}': heartbeat for condition {condition_index}/{total_conditions} "
                    f"(iteration={iteration}, n={n_value}, episodes={num_episodes}, "
                    f"steps={total_steps}, epsilon={current_epsilon:.4f}, converged={successful_policy})."
                )

            record, condition_diagnostics = run_single_condition(
                encoding=encoding,
                config_path=args.config,
                iteration=iteration,
                n_value=n_value,
                epsilon=args.epsilon,
                alpha=args.alpha,
                gamma=args.gamma,
                seed=seed,
                max_episodes_per_condition=args.max_episodes_per_condition,
                episode_callback=episode_callback,
            )
            records.append(record)
            merge_monitor_diagnostics(monitor_diagnostics, condition_diagnostics)
            results_df = records_to_frame(records)
            elapsed_seconds = time.time() - started_epoch
            write_run_artifacts(
                output_dir=output_dir,
                encoding=encoding,
                config_path=args.config,
                results_df=results_df,
                args=args,
                environment_metadata=environment_metadata,
                monitor_diagnostics=monitor_diagnostics,
            )
            write_progress_artifact(
                output_dir=output_dir,
                progress=EncodingProgress(
                    encoding=encoding,
                    total_conditions=total_conditions,
                    completed_conditions=len(records),
                    current_iteration=iteration,
                    current_n_value=n_value,
                    started_at_utc=started_at_utc,
                    updated_at_utc=utc_now_iso(),
                    elapsed_seconds=elapsed_seconds,
                    current_condition_episodes=record.episodes,
                    current_condition_steps=record.steps,
                ),
                last_record=record,
            )
            log_progress(
                f"Encoding '{encoding}': completed condition {len(records)}/{total_conditions} "
                f"(iteration={iteration}, n={n_value}, episodes={record.episodes}, "
                f"steps={record.steps}, converged={record.converged})."
            )

    write_progress_artifact(
        output_dir=output_dir,
        progress=EncodingProgress(
            encoding=encoding,
            total_conditions=total_conditions,
            completed_conditions=len(records),
            current_iteration=None,
            current_n_value=None,
            started_at_utc=started_at_utc,
            updated_at_utc=utc_now_iso(),
            elapsed_seconds=time.time() - started_epoch,
            current_condition_episodes=None,
            current_condition_steps=None,
        ),
        last_record=records[-1] if records else None,
    )
    log_progress(f"Completed encoding '{encoding}'.")
    return output_dir


def main() -> None:
    """Entrypoint for the baseline reproduction runner."""
    args = parse_args()
    apply_calibration_profile(args)
    encodings = ["simple", "one_hot", "numerical"] if args.encoding == "all" else [args.encoding]

    output_directories: dict[str, str] = {}
    for encoding_index, encoding in enumerate(encodings):
        resolved_output_dir = args.output_dir
        if args.encoding == "all" and resolved_output_dir is not None:
            resolved_output_dir = resolved_output_dir / encoding

        run_args = argparse.Namespace(**vars(args))
        run_args.output_dir = resolved_output_dir
        output_directory = run_encoding(run_args, encoding, encoding_index)
        output_directories[encoding] = str(output_directory)

    print(json.dumps({"status": "completed", "outputs": output_directories}, indent=2))


if __name__ == "__main__":
    main()
