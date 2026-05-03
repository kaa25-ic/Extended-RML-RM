# Source Code

This directory contains the implementation developed for the current project.
It is intentionally separated from the inherited baseline under `legacy/` so
that new contributions can be identified unambiguously.

## Directory Structure

```text
src/
  agents/
    dqn/
  configs/
  experiments/
  results/
  utils/
```

## Scope

The initial objective for this directory is the implementation of DQN-based
experiments for `LetterEnv` using the same task and reward structure as the
inherited thesis baseline, while evaluating the three monitor-state encoding
variants discussed in that work.

## Development Principle

New implementation should be added to this directory wherever possible. Direct
modification of code under `legacy/` should be limited to cases where
integration requires it and should be documented clearly.
