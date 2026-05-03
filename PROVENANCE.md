# Provenance

This repository is intentionally split into inherited baseline code and new
project work.

## Inherited Baseline

The following folders contain code imported from the previous project and are
kept as baseline/reference material:

- `legacy/RML/`
- `legacy/rml_reward_machines/`
- `legacy/SWI-Prolog.app/` (local-only convenience bundle, not tracked in git)
- `legacy/smoke_test_letterenv.sh`

These files should be treated as inherited work unless explicitly modified and
documented.

One compatibility addition is intentionally included alongside the inherited
monitor files:

- `legacy/RML/letter_env_spec_numerical_runtime_compatible.pl`

This specification preserves the inherited task structure while matching the
current `deep_subdict/2` argument order used by the local monitor runtime.

## New Work

The following folder is reserved for original implementation in this project:

- `src/`

New DQN agents, experiment runners, logging utilities, and evaluation code
should be created here wherever possible, even when they reuse functionality
from `legacy/`.

## Working Rule

Prefer:

- wrapping or importing legacy code from `src/`
- documenting any required changes to legacy files in commit messages and notes

Avoid:

- mixing new code directly into `legacy/` unless there is a clear integration
  reason
