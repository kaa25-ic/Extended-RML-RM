# DQN Components

This directory is reserved for DQN-based agents and supporting components used
in the current project.

## Intended Contents

- `features_extractors.py`
  Custom feature extractors for monitor-state and environment observations.
- `policies.py`
  DQN policy configuration for the current project implementation.
- `trainer.py`
  Training utilities for the current project implementation.
- `callbacks.py`
  Logging, checkpointing, and periodic evaluation callbacks.

## Initial Experimental Scope

The first implementation target is a DQN replacement for the tabular
Q-learning baseline used in the inherited `LetterEnv` experiments, evaluated
under the three monitor-state encoding variants used in the thesis:

1. simple encoding
2. one-hot encoding
3. numerical encoding

The current implementation is independent of the inherited DQN files in
`legacy/` and only reuses the inherited environment and monitor interfaces.
