"""Training utilities for LetterEnv DQN experiments."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

from src.agents.dqn.callbacks import PeriodicEvaluationCallback
from src.agents.dqn.policies import LetterEnvDQNPolicyConfig, build_policy_kwargs
from src.utils.letterenv_dqn_env import (
    LetterEnvDQNEnvConfig,
    build_letterenv_dqn_env,
    collect_monitor_diagnostics,
)
from src.utils.letterenv_thesis_protocol import PROTOCOL, protocol_as_dict


@dataclass(frozen=True)
class LetterEnvDQNTrainingConfig:
    """Training configuration for one LetterEnv DQN condition."""

    encoding: str = "one_hot"
    n_value: int = 1
    total_timesteps: int = 50_000
    seed: int | None = None
    learning_rate: float = 1e-3
    buffer_size: int = 50_000
    learning_starts: int = 1_000
    batch_size: int = 64
    gamma: float = 0.9
    tau: float = 1.0
    train_freq: int = 4
    gradient_steps: int = 1
    target_update_interval: int = 1_000
    exploration_fraction: float = 0.3
    exploration_initial_eps: float = 1.0
    exploration_final_eps: float = 0.05
    eval_freq: int = 2_500
    n_eval_episodes: int = 10
    add_state_discovery_bonus: bool = True
    state_discovery_bonus: float = 2.0
    step_penalty: float = 0.0
    no_op_penalty: float = 0.0
    simple_monitor_limit: int = 256
    success_reward_threshold: float = 110.0
    output_dir: Path = field(default_factory=Path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_train_monitor(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["r", "l", "t"])
    return pd.read_csv(path, skiprows=1)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _default_monitor_diagnostics() -> dict[str, Any]:
    return {
        "unknown_monitor_signature_count": 0,
        "unknown_monitor_state_count": 0,
        "unknown_monitor_signatures": {},
        "unknown_monitor_states": {},
    }


def train_letterenv_dqn(
    *,
    config: LetterEnvDQNTrainingConfig,
    config_path: Path,
    policy_config: LetterEnvDQNPolicyConfig | None = None,
) -> dict[str, Any]:
    """Train a fresh DQN agent for one LetterEnv encoding and n value."""
    policy_config = policy_config or LetterEnvDQNPolicyConfig()
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    train_env = Monitor(
        build_letterenv_dqn_env(
            LetterEnvDQNEnvConfig(
                encoding=config.encoding,
                config_path=config_path,
                n_value=config.n_value,
                add_state_discovery_bonus=config.add_state_discovery_bonus,
                state_discovery_bonus=config.state_discovery_bonus,
                step_penalty=config.step_penalty,
                no_op_penalty=config.no_op_penalty,
                simple_monitor_limit=config.simple_monitor_limit,
            ),
            evaluation=False,
        ),
        filename=str(output_dir / "train_monitor.csv"),
    )
    eval_env = Monitor(
        build_letterenv_dqn_env(
            LetterEnvDQNEnvConfig(
                encoding=config.encoding,
                config_path=config_path,
                n_value=config.n_value,
                add_state_discovery_bonus=False,
                state_discovery_bonus=0.0,
                step_penalty=config.step_penalty,
                no_op_penalty=config.no_op_penalty,
                simple_monitor_limit=config.simple_monitor_limit,
            ),
            evaluation=True,
        ),
        filename=str(output_dir / "eval_monitor.csv"),
    )

    if config.seed is not None:
        train_env.reset(seed=config.seed)
        eval_env.reset(seed=config.seed + 10_000)

    training_payload = {
        "experiment": "letterenv_dqn",
        "started_at_utc": _utc_now(),
        "config_path": str(config_path),
        "protocol": protocol_as_dict(),
        "training_config": {
            **asdict(config),
            "output_dir": str(config.output_dir),
        },
        "policy_config": asdict(policy_config),
        "legacy_dependencies": {
            "python_root": "legacy/rml_reward_machines",
            "monitor_root": "legacy/RML",
            "monitor_spec": PROTOCOL.numerical_monitor_spec,
            "monitor_launcher": PROTOCOL.monitor_launcher,
        },
    }
    _write_json(output_dir / "config.json", training_payload)

    callback = PeriodicEvaluationCallback(
        evaluation_env=eval_env,
        output_dir=output_dir,
        eval_freq=config.eval_freq,
        n_eval_episodes=config.n_eval_episodes,
        success_reward_threshold=config.success_reward_threshold,
    )

    model = DQN(
        policy="MultiInputPolicy",
        env=train_env,
        learning_rate=config.learning_rate,
        buffer_size=config.buffer_size,
        learning_starts=config.learning_starts,
        batch_size=config.batch_size,
        gamma=config.gamma,
        tau=config.tau,
        train_freq=(config.train_freq, "step"),
        gradient_steps=config.gradient_steps,
        target_update_interval=config.target_update_interval,
        exploration_fraction=config.exploration_fraction,
        exploration_initial_eps=config.exploration_initial_eps,
        exploration_final_eps=config.exploration_final_eps,
        policy_kwargs=build_policy_kwargs(config.encoding, policy_config),
        seed=config.seed,
        tensorboard_log=None,
        verbose=1,
    )
    model.learn(total_timesteps=config.total_timesteps, callback=callback, progress_bar=False)
    model.save(str(output_dir / "model_final"))

    train_monitor = _read_train_monitor(output_dir / "train_monitor.csv")
    eval_records = [asdict(record) for record in callback.records]
    best_eval = max(eval_records, key=lambda record: (record["success_rate"], record["mean_return"]), default=None)
    train_monitor_diagnostics = collect_monitor_diagnostics(train_env) or _default_monitor_diagnostics()
    eval_monitor_diagnostics = collect_monitor_diagnostics(eval_env) or _default_monitor_diagnostics()
    diagnostics_payload = {
        "train_env": train_monitor_diagnostics,
        "eval_env": eval_monitor_diagnostics,
    }
    _write_json(output_dir / "monitor_diagnostics.json", diagnostics_payload)

    summary = {
        "completed_at_utc": _utc_now(),
        "encoding": config.encoding,
        "n_value": config.n_value,
        "total_timesteps": config.total_timesteps,
        "seed": config.seed,
        "train_episodes_completed": int(len(train_monitor)),
        "train_mean_return": float(train_monitor["r"].mean()) if not train_monitor.empty else None,
        "train_mean_length": float(train_monitor["l"].mean()) if not train_monitor.empty else None,
        "evaluation_count": len(eval_records),
        "best_evaluation": best_eval,
        "artifacts": {
            "final_model": str(output_dir / "model_final.zip"),
            "best_model": str(output_dir / "best_model.zip"),
            "train_monitor": str(output_dir / "train_monitor.csv"),
            "eval_monitor": str(output_dir / "eval_monitor.csv"),
            "eval_metrics": str(output_dir / "eval_metrics.csv"),
            "monitor_diagnostics": str(output_dir / "monitor_diagnostics.json"),
        },
    }
    _write_json(output_dir / "summary.json", summary)

    train_env.close()
    eval_env.close()

    return summary
