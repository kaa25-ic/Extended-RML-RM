"""Callbacks for LetterEnv DQN experiments."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


@dataclass(frozen=True)
class EvaluationRecord:
    """Aggregate metrics for one DQN evaluation checkpoint."""

    timesteps: int
    episode_count: int
    mean_return: float
    std_return: float
    mean_length: float
    success_rate: float
    mean_final_reward: float
    mean_final_base_reward: float


class PeriodicEvaluationCallback(BaseCallback):
    """Evaluate a DQN policy at fixed timestep intervals."""

    def __init__(
        self,
        *,
        evaluation_env: gym.Env,
        output_dir: Path,
        eval_freq: int,
        n_eval_episodes: int,
        success_reward_threshold: float,
    ) -> None:
        super().__init__(verbose=0)
        self.evaluation_env = evaluation_env
        self.output_dir = output_dir
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.success_reward_threshold = success_reward_threshold
        self.records: list[EvaluationRecord] = []
        self.best_success_rate = float("-inf")
        self.best_mean_return = float("-inf")
        self.metrics_path = self.output_dir / "eval_metrics.csv"
        self.best_model_path = self.output_dir / "best_model"

    def _on_training_start(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with self.metrics_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(asdict(self._empty_record()).keys()))
            writer.writeheader()

    def _on_step(self) -> bool:
        if self.eval_freq <= 0:
            return True
        if self.num_timesteps % self.eval_freq != 0:
            return True

        record = self._evaluate_current_policy()
        self.records.append(record)
        with self.metrics_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(asdict(record).keys()))
            writer.writerow(asdict(record))

        if (
            record.success_rate > self.best_success_rate
            or (
                np.isclose(record.success_rate, self.best_success_rate)
                and record.mean_return > self.best_mean_return
            )
        ):
            self.best_success_rate = record.success_rate
            self.best_mean_return = record.mean_return
            self.model.save(str(self.best_model_path))

        return True

    def _on_training_end(self) -> None:
        self.evaluation_env.close()

    def _evaluate_current_policy(self) -> EvaluationRecord:
        returns: list[float] = []
        lengths: list[int] = []
        final_rewards: list[float] = []
        final_base_rewards: list[float] = []
        successes: list[float] = []

        for _ in range(self.n_eval_episodes):
            observation, _ = self.evaluation_env.reset()
            terminated = False
            truncated = False
            episode_return = 0.0
            episode_length = 0
            final_reward = 0.0
            final_base_reward = 0.0

            while not terminated and not truncated:
                action, _ = self.model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, info = self.evaluation_env.step(action)
                episode_return += float(reward)
                episode_length += 1
                final_reward = float(reward)
                final_base_reward = float(info.get("base_reward", reward))

            returns.append(episode_return)
            lengths.append(episode_length)
            final_rewards.append(final_reward)
            final_base_rewards.append(final_base_reward)
            successes.append(1.0 if final_base_reward >= self.success_reward_threshold else 0.0)

        return EvaluationRecord(
            timesteps=int(self.num_timesteps),
            episode_count=len(self.records) + 1,
            mean_return=float(np.mean(returns)),
            std_return=float(np.std(returns)),
            mean_length=float(np.mean(lengths)),
            success_rate=float(np.mean(successes)),
            mean_final_reward=float(np.mean(final_rewards)),
            mean_final_base_reward=float(np.mean(final_base_rewards)),
        )

    @staticmethod
    def _empty_record() -> EvaluationRecord:
        return EvaluationRecord(
            timesteps=0,
            episode_count=0,
            mean_return=0.0,
            std_return=0.0,
            mean_length=0.0,
            success_rate=0.0,
            mean_final_reward=0.0,
            mean_final_base_reward=0.0,
        )
