# Project Code Base

This folder is a cleaned standalone starting point for extending the previous
RML Reward Machines work toward deep reinforcement learning and
counterfactual-based methods.

## Layout

- `RML/`: local Prolog monitor runtime and task specifications.
- `SWI-Prolog.app/`: optional local-only SWI-Prolog app bundle for macOS convenience. Keep this out of git.
- `rml_reward_machines/`: Python RL code, environments, agents, wrappers, and experiments.

## First-Time Setup

1. Create a Python virtual environment:

```bash
cd rml_reward_machines
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Install SWI-Prolog in one of these ways:

- Recommended for GitHub: install SWI-Prolog on your machine so `swipl` is available on `PATH`.
- Local macOS convenience: keep a local `SWI-Prolog.app` inside this `code/` folder, but do not commit it.

3. Run the standalone smoke test from the `code/` folder:

```bash
cd ..
./smoke_test_letterenv.sh
```

The smoke test starts the copied RML monitor, runs a short
`RMLGym_Simple`-based episode on the numerical letter environment, and then
shuts the monitor down.

## Notes

- The monitor launch scripts in `RML/` first try `swipl` from your system `PATH`.
- If `swipl` is not installed globally, the scripts will fall back to a sibling
  `SWI-Prolog.app` bundle if you keep one locally in this folder.
- `SWI-Prolog.app` is intentionally gitignored so the repository stays portable.
- The `results/` folder under `rml_reward_machines/` keeps plotting scripts but
  excludes copied experiment output files.
- This codebase still contains older Gym/Gymnasium compatibility quirks, so
  some warnings during resets/steps are expected until the env wrappers are
  modernized.
