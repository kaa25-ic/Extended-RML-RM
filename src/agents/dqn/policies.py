"""Policy configuration utilities for LetterEnv DQN experiments."""

from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn

from src.agents.dqn.features_extractors import (
    LetterEnvSimpleEmbeddingExtractor,
    LetterEnvVectorExtractor,
)


@dataclass(frozen=True)
class LetterEnvDQNPolicyConfig:
    """Policy architecture parameters for LetterEnv DQN."""

    features_dim: int = 128
    position_hidden_dim: int = 64
    monitor_hidden_dim: int = 64
    monitor_embedding_dim: int = 16
    max_monitor_states: int = 256
    network_architecture: tuple[int, ...] = (128, 128)


def build_policy_kwargs(encoding: str, config: LetterEnvDQNPolicyConfig) -> dict:
    """Build SB3 DQN policy kwargs for a selected monitor encoding."""
    if encoding == "simple":
        return {
            "features_extractor_class": LetterEnvSimpleEmbeddingExtractor,
            "features_extractor_kwargs": {
                "features_dim": config.features_dim,
                "position_hidden_dim": config.position_hidden_dim,
                "monitor_embedding_dim": config.monitor_embedding_dim,
                "max_monitor_states": config.max_monitor_states,
            },
            "net_arch": list(config.network_architecture),
            "activation_fn": nn.ReLU,
        }

    return {
        "features_extractor_class": LetterEnvVectorExtractor,
        "features_extractor_kwargs": {
            "features_dim": config.features_dim,
            "position_hidden_dim": config.position_hidden_dim,
            "monitor_hidden_dim": config.monitor_hidden_dim,
        },
        "net_arch": list(config.network_architecture),
        "activation_fn": nn.ReLU,
    }
