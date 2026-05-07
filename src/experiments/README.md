# Experiments

This directory stores experiment runners implemented for the current project.

## Baseline Reproduction

The first implemented study should reproduce the inherited tabular
`LetterEnv` protocol before introducing DQN. This establishes a controlled
comparison point for later deep RL experiments.

The baseline runner is:

- `python -m src.experiments.reproduce_letterenv_tabular_baseline`

The companion shell launcher:

- `src/experiments/run_letterenv_tabular_baseline.sh`

starts the inherited numerical monitor automatically and forwards arguments to
the Python runner.

By default, the launcher uses the runtime-compatible numerical specification
`legacy/RML/letter_env_spec_numerical_runtime_compatible.pl`. The inherited
`legacy/RML/letter_env_spec_numerical.pl` is retained unchanged for reference,
but it does not match the current `deep_subdict/2` argument order.

For `numerical`, the baseline reproduction now uses a thesis-faithful
compatibility wrapper over the inherited numerical encoder. It keeps the
original 20-feature basis while mapping current runtime monitor strings back
onto the legacy thesis state space, which avoids inherited `UNKNOWN State`
failures without changing the encoding target.

For `one_hot`, the baseline reproduction likewise uses a thesis-faithful
compatibility wrapper over the inherited `RMLGym_One_Hot` encoder so the
tabular sweep stays on the original one-hot basis.

For bounded verification runs, `--max-episodes-per-condition` can be supplied
to the Python runner or the shell launcher. Omitting that flag preserves the
full convergence-based thesis protocol.

For runtime calibration, `--calibration` reduces the sweep to a smaller profile
and sets a finite episode cap unless you override those values explicitly.

During long runs, the runner writes:

- `train_metrics.csv` after each completed condition
- `summary.json` after each completed condition
- `progress.json` with the current encoding, completed-condition count, and the
  current in-condition episode and step counters

The shell launcher also supports monitor selection through environment
variables:

- `MONITOR_SCRIPT_NAME=online_monitor_edit.sh` for the default thesis-faithful monitor
- `MONITOR_SCRIPT_NAME=online_monitor_edit_fast.sh` for the faster variant
- `MONITOR_SPEC_NAME=./letter_env_spec_numerical.pl` to force the original inherited numerical specification

The plotting utility for thesis-style comparison figures is:

- `python -m src.experiments.plot_letterenv_tabular_results`

It reads completed runs from `src/results/tabular_letterenv/`, aggregates the
results by encoding and `n`, and saves PNG, PDF, CSV, and JSON artifacts for
the resulting figure.

## DQN Extension

The initial experiment set should evaluate DQN on `LetterEnv` using the same
task definition and reward structure as the inherited thesis baseline while
varying the monitor-state representation:

- simple integer encoding
- one-hot encoding
- numerical encoding

The current training entry points are:

- `python -m src.experiments.train_letterenv_dqn`
- `src/experiments/run_letterenv_dqn.sh`
- `src/experiments/run_letterenv_dqn_one_hot_sweep_option_b.sh`
- `python -m src.experiments.evaluate_letterenv_dqn_policy`
- `src/experiments/run_letterenv_dqn_eval.sh`

The shell launcher starts the inherited monitor automatically and uses the
runtime-compatible numerical specification by default.
For training runs, it now starts isolated monitor servers for training and
evaluation on separate ports by default (`18081` and `18082`) to avoid
cross-contaminating callback evaluation with the live training monitor state.

The evaluation-trace runner records deterministic policy rollouts from a saved
`best_model.zip` or `model_final.zip`, including step-by-step observations,
actions, rewards, and raw monitor states. By default it now mirrors the
callback reset style: the first episode is seeded, then subsequent episodes use
plain resets unless `--reseed-each-episode` is supplied.

For a focused one-hot DQN hyperparameter sweep on `LetterEnv`, the repository
also includes:

- `src/experiments/run_letterenv_dqn_one_hot_sweep_option_b.sh`

This launcher runs the reduced "option B" sweep for `n=1`, `seed=0`, and
`20,000` timesteps while varying:

- `exploration_fraction` in `{0.3, 0.5}`
- `learning_starts` in `{500, 1000}`
- `learning_rate` in `{0.0005, 0.001, 0.002}`
- `batch_size` in `{32, 64}`

The sweep uses `5,000`-step evaluation intervals, `5` evaluation episodes, the
fast monitor launcher by default, and writes a `sweep_manifest.csv` alongside the
per-run output directories.

The first DQN implementation path is designed to:

- use Stable-Baselines3 DQN with a fresh `src` implementation
- reuse only the inherited environment and monitor interfaces
- keep the tabular `+2` first-visit state bonus as an explicit optional
  training wrapper

Optional DQN ablations may also add:

- a per-step penalty
- a no-op penalty when the agent remains in the same grid cell

These should be reported separately from the thesis-faithful baseline.

## Comparison Standard

New DQN runs should be compared against:

- the results reported in the inherited thesis
- and, where practical, rerun legacy baselines obtained from the imported code

## Output Convention

Each experiment run should produce a self-contained output directory under
`src/results/` containing:

- `config.json`
- `command.txt`
- `train_metrics.csv`
- `summary.json`
- `eval_metrics.csv` where applicable
- optional model checkpoints
