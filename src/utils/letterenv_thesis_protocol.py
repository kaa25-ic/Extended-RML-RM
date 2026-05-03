"""LetterEnv protocol definitions for the inherited tabular baseline."""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.utils.legacy_paths import legacy_python_root


@dataclass(frozen=True)
class LetterEnvTabularProtocol:
    """Formal specification of the inherited LetterEnv tabular study."""

    environment_id: str = "letter-env"
    config_relative_path: str = "examples/letter_env.yaml"
    numerical_monitor_spec: str = "letter_env_spec_numerical_runtime_compatible.pl"
    monitor_launcher: str = "online_monitor_edit.sh"
    max_episode_steps: int = 200
    iterations: int = 20
    max_n: int = 10
    epsilon: float = 0.4
    alpha: float = 0.5
    gamma: float = 0.9
    epsilon_decay: float = 0.99
    terminal_reward_targets: tuple[int, int] = (110, 112)
    success_window: int = 20
    transition_bonus: int = 10


PROTOCOL = LetterEnvTabularProtocol()


def protocol_as_dict() -> dict[str, Any]:
    """Return the tabular protocol in JSON-serializable form."""
    return asdict(PROTOCOL)


def encoding_entry_point(encoding: str) -> str:
    """Return the legacy Gym wrapper entry point for an encoding variant."""
    if encoding == "simple":
        return "envs.property_envs.letterenv_numerical_wrappers:RML_LetterEnv_numerical_4_Simple"
    if encoding in {"one_hot", "numerical"}:
        return "envs.property_envs.letterenv_numerical_wrappers:RML_LetterEnv_numerical_4"
    raise ValueError(f"Unsupported encoding: {encoding}")


def load_monitor_state_catalogue() -> dict[int, str]:
    """Load the inherited monitor-state catalogue without duplicating it in ``src``."""
    legacy_file = legacy_python_root() / "utils" / "new_vs_old_functions.py"
    source = legacy_file.read_text(encoding="utf-8")
    module = ast.parse(source, filename=str(legacy_file))

    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == "learning_episode_new_vector":
            for statement in node.body:
                if not isinstance(statement, ast.Assign):
                    continue
                if len(statement.targets) != 1:
                    continue
                target = statement.targets[0]
                if isinstance(target, ast.Name) and target.id == "states_for_encoding":
                    return ast.literal_eval(statement.value)

    raise RuntimeError(
        "Unable to locate the inherited LetterEnv monitor-state catalogue in "
        f"{legacy_file}."
    )
