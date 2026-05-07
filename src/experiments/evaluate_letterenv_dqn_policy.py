"""Evaluate a trained LetterEnv DQN policy and record episode traces."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import DQN

from src.utils.letterenv_dqn_env import LetterEnvDQNEnvConfig, build_letterenv_dqn_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained LetterEnv DQN model and save step-by-step traces."
    )
    parser.add_argument("--run-dir", type=Path, required=True, help="Completed DQN run directory.")
    parser.add_argument("--config", type=Path, required=True, help="Temporary YAML config path for the active monitor.")
    parser.add_argument("--model-kind", choices=("best", "final"), default="best")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Optional explicit model artifact path. Overrides --model-kind resolution.",
    )
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument(
        "--reseed-each-episode",
        action="store_true",
        help="Reset every episode with a fresh explicit seed instead of using callback-style resets.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def _to_python(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [_to_python(item) for item in value]
    if isinstance(value, list):
        return [_to_python(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_python(item) for key, item in value.items()}
    return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_run_config(run_dir: Path) -> dict[str, Any]:
    config_path = run_dir / "config.json"
    return json.loads(config_path.read_text(encoding="utf-8"))


def _resolve_model_path(run_dir: Path, model_kind: str, explicit_model_path: Path | None) -> Path:
    if explicit_model_path is not None:
        path = explicit_model_path if explicit_model_path.is_absolute() else run_dir / explicit_model_path
        if not path.exists():
            raise FileNotFoundError(f"Unable to locate explicit model artifact: {path}")
        return path
    filename = "best_model.zip" if model_kind == "best" else "model_final.zip"
    path = run_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Unable to locate model artifact: {path}")
    return path


def _resolve_output_dir(run_dir: Path, requested: Path | None, model_kind: str) -> Path:
    if requested is not None:
        return requested if requested.is_absolute() else run_dir / requested
    return run_dir / "eval_traces" / model_kind


def _extract_training_config(run_config: dict[str, Any], config_path: Path) -> LetterEnvDQNEnvConfig:
    training = run_config["training_config"]
    return LetterEnvDQNEnvConfig(
        encoding=training["encoding"],
        config_path=config_path,
        n_value=int(training["n_value"]),
        add_state_discovery_bonus=False,
        state_discovery_bonus=0.0,
        step_penalty=float(training.get("step_penalty", 0.0)),
        no_op_penalty=float(training.get("no_op_penalty", 0.0)),
        simple_monitor_limit=int(training["simple_monitor_limit"]),
    )


def evaluate_policy(
    *,
    run_dir: Path,
    config_path: Path,
    model_kind: str,
    model_path: Path | None,
    episodes: int,
    max_steps: int,
    seed_base: int,
    reseed_each_episode: bool,
    output_dir: Path,
) -> dict[str, Any]:
    run_config = _load_run_config(run_dir)
    resolved_model_path = _resolve_model_path(run_dir, model_kind, model_path)
    env_config = _extract_training_config(run_config, config_path)
    env = build_letterenv_dqn_env(env_config, evaluation=True)
    model = DQN.load(str(resolved_model_path))

    output_dir.mkdir(parents=True, exist_ok=True)
    traces: list[dict[str, Any]] = []

    for episode_index in range(episodes):
        if episode_index == 0 or reseed_each_episode:
            observation, _info = env.reset(seed=seed_base + episode_index)
        else:
            observation, _info = env.reset()
        initial_observation = _to_python(observation)
        steps: list[dict[str, Any]] = []
        terminated = False
        truncated = False
        total_reward = 0.0
        final_reward = 0.0
        final_base_reward = 0.0
        raw_monitor_state = getattr(env, "monitor_state_unencoded", None)

        for step_index in range(max_steps):
            action, _state = model.predict(observation, deterministic=True)
            next_observation, reward, terminated, truncated, info = env.step(action)
            final_reward = float(reward)
            final_base_reward = float(info.get("base_reward", reward))
            total_reward += final_reward
            raw_monitor_state = getattr(env, "monitor_state_unencoded", raw_monitor_state)

            steps.append(
                {
                    "step_index": step_index + 1,
                    "action": int(action),
                    "reward": final_reward,
                    "base_reward": final_base_reward,
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "observation": _to_python(next_observation),
                    "raw_monitor_state": raw_monitor_state,
                    "info": _to_python(info),
                }
            )
            observation = next_observation
            if terminated or truncated:
                break

        traces.append(
            {
                "episode_index": episode_index + 1,
                "seed": seed_base + episode_index,
                "initial_observation": initial_observation,
                "step_count": len(steps),
                "total_reward": total_reward,
                "final_reward": final_reward,
                "final_base_reward": final_base_reward,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "success": bool(final_base_reward >= run_config["training_config"]["success_reward_threshold"]),
                "steps": steps,
            }
        )

    summary = {
        "evaluated_at_utc": _utc_now(),
        "run_dir": str(run_dir),
        "model_kind": model_kind,
        "model_path": str(resolved_model_path),
        "episodes": episodes,
        "reseed_each_episode": reseed_each_episode,
        "success_count": sum(1 for trace in traces if trace["success"]),
        "success_rate": sum(1 for trace in traces if trace["success"]) / max(episodes, 1),
        "mean_total_reward": float(np.mean([trace["total_reward"] for trace in traces])) if traces else None,
        "mean_final_reward": float(np.mean([trace["final_reward"] for trace in traces])) if traces else None,
        "mean_final_base_reward": float(np.mean([trace["final_base_reward"] for trace in traces])) if traces else None,
        "mean_step_count": float(np.mean([trace["step_count"] for trace in traces])) if traces else None,
    }

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "episode_traces.json").write_text(json.dumps(traces, indent=2), encoding="utf-8")
    env.close()
    return summary


def main() -> None:
    args = parse_args()
    output_dir = _resolve_output_dir(args.run_dir, args.output_dir, args.model_kind)
    summary = evaluate_policy(
        run_dir=args.run_dir,
        config_path=args.config,
        model_kind=args.model_kind,
        model_path=args.model_path,
        episodes=args.episodes,
        max_steps=args.max_steps,
        seed_base=args.seed_base,
        reseed_each_episode=args.reseed_each_episode,
        output_dir=output_dir,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
