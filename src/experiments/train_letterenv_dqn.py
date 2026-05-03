"""Train a fresh DQN baseline for LetterEnv."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from src.agents.dqn.policies import LetterEnvDQNPolicyConfig
from src.agents.dqn.trainer import LetterEnvDQNTrainingConfig, train_letterenv_dqn
from src.utils.legacy_paths import code_root
from src.utils.letterenv_dqn_env import SUPPORTED_ENCODINGS, default_dqn_results_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a DQN agent on LetterEnv using the current project implementation."
    )
    parser.add_argument("--encoding", choices=SUPPORTED_ENCODINGS, default="one_hot")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--n-value", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--total-timesteps", type=int, default=50_000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--buffer-size", type=int, default=50_000)
    parser.add_argument("--learning-starts", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--train-freq", type=int, default=4)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--target-update-interval", type=int, default=1_000)
    parser.add_argument("--exploration-fraction", type=float, default=0.3)
    parser.add_argument("--exploration-initial-eps", type=float, default=1.0)
    parser.add_argument("--exploration-final-eps", type=float, default=0.05)
    parser.add_argument("--eval-freq", type=int, default=2_500)
    parser.add_argument("--n-eval-episodes", type=int, default=10)
    parser.add_argument("--disable-state-discovery-bonus", action="store_true")
    parser.add_argument("--state-discovery-bonus", type=float, default=2.0)
    parser.add_argument("--step-penalty", type=float, default=0.0)
    parser.add_argument("--no-op-penalty", type=float, default=0.0)
    parser.add_argument("--simple-monitor-limit", type=int, default=256)
    parser.add_argument("--success-reward-threshold", type=float, default=110.0)
    parser.add_argument("--features-dim", type=int, default=128)
    parser.add_argument("--position-hidden-dim", type=int, default=64)
    parser.add_argument("--monitor-hidden-dim", type=int, default=64)
    parser.add_argument("--monitor-embedding-dim", type=int, default=16)
    parser.add_argument("--network-architecture", type=int, nargs="+", default=[128, 128])
    return parser.parse_args()


def resolve_output_directory(requested: Path | None, encoding: str, n_value: int) -> Path:
    if requested is not None:
        return requested if requested.is_absolute() else code_root() / requested

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return default_dqn_results_root() / encoding / f"n_{n_value}" / timestamp


def main() -> None:
    args = parse_args()

    training_config = LetterEnvDQNTrainingConfig(
        encoding=args.encoding,
        n_value=args.n_value,
        total_timesteps=args.total_timesteps,
        seed=args.seed,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        gamma=args.gamma,
        tau=args.tau,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        target_update_interval=args.target_update_interval,
        exploration_fraction=args.exploration_fraction,
        exploration_initial_eps=args.exploration_initial_eps,
        exploration_final_eps=args.exploration_final_eps,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        add_state_discovery_bonus=not args.disable_state_discovery_bonus,
        state_discovery_bonus=args.state_discovery_bonus,
        step_penalty=args.step_penalty,
        no_op_penalty=args.no_op_penalty,
        simple_monitor_limit=args.simple_monitor_limit,
        success_reward_threshold=args.success_reward_threshold,
        output_dir=resolve_output_directory(args.output_dir, args.encoding, args.n_value),
    )
    policy_config = LetterEnvDQNPolicyConfig(
        features_dim=args.features_dim,
        position_hidden_dim=args.position_hidden_dim,
        monitor_hidden_dim=args.monitor_hidden_dim,
        monitor_embedding_dim=args.monitor_embedding_dim,
        max_monitor_states=args.simple_monitor_limit,
        network_architecture=tuple(args.network_architecture),
    )

    summary = train_letterenv_dqn(
        config=training_config,
        config_path=args.config,
        policy_config=policy_config,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
