# Configuration Files

This directory stores configuration files for experiments implemented in
`src/`.

## Recommended Convention

Separate configuration files should be maintained for each encoding and
experimental condition. For example:

- `letterenv_dqn_simple.yaml`
- `letterenv_dqn_one_hot.yaml`
- `letterenv_dqn_numerical.yaml`

Each configuration file should distinguish clearly between:

- task and environment settings inherited from the baseline setup
- DQN hyperparameters introduced in the current project
- logging, evaluation, and output settings
