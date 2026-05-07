"""Training utilities for LetterEnv DQN experiments."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import os
from pathlib import Path
import random
import shutil
import statistics
import subprocess
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

from src.agents.dqn.callbacks import CandidateCheckpoint, PeriodicEvaluationCallback
from src.agents.dqn.policies import LetterEnvDQNPolicyConfig, build_policy_kwargs
from src.utils.legacy_paths import code_root
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
    eval_seed_base: int = 0
    enable_fresh_verification: bool = True
    add_state_discovery_bonus: bool = True
    state_discovery_bonus: float = 2.0
    monitor_progress_bonus: float = 0.0
    monitor_regression_penalty: float = 0.0
    neutralize_legacy_transition_bonus: bool = False
    legacy_transition_bonus: float = 10.0
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


def _write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def _default_monitor_diagnostics() -> dict[str, Any]:
    return {
        "unknown_monitor_signature_count": 0,
        "unknown_monitor_state_count": 0,
        "unknown_monitor_signatures": {},
        "unknown_monitor_states": {},
    }


def _configure_global_seeds(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)


def _first_record_matching(
    records: list[dict[str, Any]],
    predicate,
) -> dict[str, Any] | None:
    for record in records:
        if predicate(record):
            return record
    return None


def _resolve_callback_best_source_path(
    candidate_checkpoints: list[CandidateCheckpoint],
    callback_best_eval: dict[str, Any] | None,
) -> Path | None:
    if callback_best_eval is None:
        return None

    best_timesteps = int(callback_best_eval["timesteps"])
    for candidate in candidate_checkpoints:
        if candidate.timesteps == best_timesteps:
            return candidate.model_path
    return None


def _verify_candidate_checkpoints(
    *,
    output_dir: Path,
    candidate_checkpoints: list[CandidateCheckpoint],
    verification_episodes: int,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    if not candidate_checkpoints:
        return None, []

    eval_script = code_root() / "src" / "experiments" / "run_letterenv_dqn_eval.sh"
    verification_root = output_dir / "fresh_eval_candidates"
    verification_root.mkdir(parents=True, exist_ok=True)

    verification_results: list[dict[str, Any]] = []
    best_result: dict[str, Any] | None = None

    for index, candidate in enumerate(candidate_checkpoints):
        candidate_output_dir = verification_root / f"checkpoint_{candidate.timesteps}"
        env = os.environ.copy()
        env["PYTHON_BIN"] = sys.executable
        env["MONITOR_PORT"] = str(18100 + index)
        command = [
            "bash",
            str(eval_script),
            "--run-dir",
            str(output_dir),
            "--model-kind",
            "best",
            "--model-path",
            str(candidate.model_path),
            "--episodes",
            str(verification_episodes),
            "--max-steps",
            str(PROTOCOL.max_episode_steps),
            "--output-dir",
            str(candidate_output_dir),
        ]
        completed = subprocess.run(
            command,
            cwd=code_root(),
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        summary_path = candidate_output_dir / "summary.json"
        traces_path = candidate_output_dir / "episode_traces.json"
        verification_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        traces = json.loads(traces_path.read_text(encoding="utf-8"))
        total_rewards = [float(trace["total_reward"]) for trace in traces]

        normalized_record = {
            "timesteps": candidate.timesteps,
            "episode_count": int(verification_summary["episodes"]),
            "mean_return": float(verification_summary["mean_total_reward"]),
            "std_return": float(statistics.pstdev(total_rewards)) if len(total_rewards) > 1 else 0.0,
            "mean_length": float(verification_summary["mean_step_count"]),
            "success_rate": float(verification_summary["success_rate"]),
            "mean_final_reward": float(verification_summary["mean_final_reward"]),
            "mean_final_base_reward": float(verification_summary["mean_final_base_reward"]),
        }
        result_payload = {
            "timesteps": candidate.timesteps,
            "model_path": str(candidate.model_path),
            "output_dir": str(candidate_output_dir),
            "callback_record": asdict(candidate.record),
            "verified_record": normalized_record,
            "verification_summary": verification_summary,
            "verification_stdout": completed.stdout,
        }
        verification_results.append(result_payload)

        if best_result is None or (
            normalized_record["success_rate"],
            normalized_record["mean_return"],
        ) > (
            best_result["verified_record"]["success_rate"],
            best_result["verified_record"]["mean_return"],
        ):
            best_result = result_payload

    return best_result, verification_results


def train_letterenv_dqn(
    *,
    config: LetterEnvDQNTrainingConfig,
    train_config_path: Path,
    eval_config_path: Path | None = None,
    policy_config: LetterEnvDQNPolicyConfig | None = None,
    invocation_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train a fresh DQN agent for one LetterEnv encoding and n value."""
    policy_config = policy_config or LetterEnvDQNPolicyConfig()
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    started_monotonic = time.monotonic()
    resolved_eval_config_path = eval_config_path or train_config_path
    _configure_global_seeds(config.seed)
    train_monitor_config_copy = output_dir / "monitor_train_config.yaml"
    eval_monitor_config_copy = output_dir / "monitor_eval_config.yaml"
    train_monitor_config_copy.write_text(train_config_path.read_text(encoding="utf-8"), encoding="utf-8")
    eval_monitor_config_copy.write_text(
        resolved_eval_config_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    train_env = Monitor(
        build_letterenv_dqn_env(
            LetterEnvDQNEnvConfig(
                encoding=config.encoding,
                config_path=train_config_path,
                n_value=config.n_value,
                add_state_discovery_bonus=config.add_state_discovery_bonus,
                state_discovery_bonus=config.state_discovery_bonus,
                monitor_progress_bonus=config.monitor_progress_bonus,
                monitor_regression_penalty=config.monitor_regression_penalty,
                neutralize_legacy_transition_bonus=config.neutralize_legacy_transition_bonus,
                legacy_transition_bonus=config.legacy_transition_bonus,
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
                config_path=resolved_eval_config_path,
                n_value=config.n_value,
                add_state_discovery_bonus=False,
                state_discovery_bonus=0.0,
                monitor_progress_bonus=config.monitor_progress_bonus,
                monitor_regression_penalty=config.monitor_regression_penalty,
                neutralize_legacy_transition_bonus=config.neutralize_legacy_transition_bonus,
                legacy_transition_bonus=config.legacy_transition_bonus,
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
        "train_config_path": str(train_monitor_config_copy),
        "eval_config_path": str(eval_monitor_config_copy),
        "original_train_config_path": str(train_config_path),
        "original_eval_config_path": str(resolved_eval_config_path),
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
    if invocation_metadata is not None:
        training_payload["invocation"] = invocation_metadata
    _write_json(output_dir / "config.json", training_payload)
    saved_command = None if invocation_metadata is None else invocation_metadata.get("preferred_command")
    if isinstance(saved_command, str) and saved_command:
        _write_text(output_dir / "command.txt", saved_command + "\n")

    callback = PeriodicEvaluationCallback(
        evaluation_env=eval_env,
        output_dir=output_dir,
        eval_freq=config.eval_freq,
        n_eval_episodes=config.n_eval_episodes,
        success_reward_threshold=config.success_reward_threshold,
        eval_seed_base=config.eval_seed_base,
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
    runtime_seconds = time.monotonic() - started_monotonic

    train_monitor = _read_train_monitor(output_dir / "train_monitor.csv")
    eval_records = [asdict(record) for record in callback.records]
    callback_best_eval = max(
        eval_records,
        key=lambda record: (record["success_rate"], record["mean_return"]),
        default=None,
    )
    final_eval = eval_records[-1] if eval_records else None
    first_success_eval = _first_record_matching(eval_records, lambda record: record["success_rate"] > 0.0)
    first_full_success_eval = _first_record_matching(eval_records, lambda record: record["success_rate"] >= 1.0)
    verified_best_eval = None
    verification_results: list[dict[str, Any]] = []
    if config.enable_fresh_verification:
        verified_best_eval, verification_results = _verify_candidate_checkpoints(
            output_dir=output_dir,
            candidate_checkpoints=callback.candidate_checkpoints,
            verification_episodes=config.n_eval_episodes,
        )
    best_eval = verified_best_eval["verified_record"] if verified_best_eval is not None else callback_best_eval
    best_model_source_path = (
        Path(verified_best_eval["model_path"])
        if verified_best_eval is not None
        else _resolve_callback_best_source_path(callback.candidate_checkpoints, callback_best_eval)
    )
    best_model_artifact_path = output_dir / "best_model.zip"
    if best_model_source_path is not None and best_model_source_path.exists():
        shutil.copy2(best_model_source_path, best_model_artifact_path)
    elif (output_dir / "model_final.zip").exists():
        shutil.copy2(output_dir / "model_final.zip", best_model_artifact_path)
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
        "runtime_seconds": runtime_seconds,
        "timesteps_per_second": (float(config.total_timesteps) / runtime_seconds) if runtime_seconds > 0 else None,
        "evaluation_count": len(eval_records),
        "first_success_evaluation": first_success_eval,
        "first_full_success_evaluation": first_full_success_eval,
        "callback_best_evaluation": callback_best_eval,
        "best_evaluation": best_eval,
        "final_evaluation": final_eval,
        "final_exploration_rate": float(model.exploration_rate) if hasattr(model, "exploration_rate") else None,
        "fresh_verification_enabled": bool(config.enable_fresh_verification),
        "fresh_verification_best_evaluation": verified_best_eval["verified_record"] if verified_best_eval else None,
        "fresh_verification_best_model_path": str(best_model_source_path) if verified_best_eval and best_model_source_path else None,
        "fresh_verification_results": verification_results,
        "monitor_diagnostics": {
            "train_env": {
                "unknown_monitor_signature_count": int(
                    train_monitor_diagnostics.get("unknown_monitor_signature_count", 0)
                ),
                "unknown_monitor_state_count": int(
                    train_monitor_diagnostics.get("unknown_monitor_state_count", 0)
                ),
            },
            "eval_env": {
                "unknown_monitor_signature_count": int(
                    eval_monitor_diagnostics.get("unknown_monitor_signature_count", 0)
                ),
                "unknown_monitor_state_count": int(
                    eval_monitor_diagnostics.get("unknown_monitor_state_count", 0)
                ),
            },
        },
        "artifacts": {
            "final_model": str(output_dir / "model_final.zip"),
            "best_model": str(best_model_artifact_path),
            "command": str(output_dir / "command.txt"),
            "train_monitor": str(output_dir / "train_monitor.csv"),
            "eval_monitor": str(output_dir / "eval_monitor.csv"),
            "eval_metrics": str(output_dir / "eval_metrics.csv"),
            "monitor_diagnostics": str(output_dir / "monitor_diagnostics.json"),
            "train_monitor_config": str(train_monitor_config_copy),
            "eval_monitor_config": str(eval_monitor_config_copy),
        },
    }
    _write_json(output_dir / "summary.json", summary)

    train_env.close()
    eval_env.close()

    return summary
