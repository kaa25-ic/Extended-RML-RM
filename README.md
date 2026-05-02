# Project Code Base

This repository is now split into two clearly separated areas:

- `legacy/`: inherited baseline code from the previous project, kept for reuse,
  comparison, and reference.
- `src/`: your own implementation area for new work such as DQN, deep RL
  monitor embeddings, and counterfactual extensions.

The goal of this structure is to make provenance explicit: baseline code stays
isolated, while your original work is developed separately.

## Layout

- `legacy/RML/`: inherited Prolog monitor runtime and task specifications.
- `legacy/SWI-Prolog.app/`: optional local-only SWI-Prolog app bundle for
  macOS convenience. Keep this out of git.
- `legacy/rml_reward_machines/`: inherited Python RL code, environments,
  wrappers, agents, and experiments.
- `legacy/smoke_test_letterenv.sh`: legacy smoke test for the imported code.
- `src/`: clean area for your new implementation.

## Recommended Workflow

1. Treat `legacy/` as the baseline.
2. Build new code in `src/`.
3. Reuse legacy modules intentionally, rather than mixing new files into the
   inherited folders unless necessary.
4. Compare new results against the legacy baseline using shared evaluation
   setups.

## Legacy Setup

If you need to run the inherited baseline:

1. Create a Python virtual environment:

```bash
cd legacy/rml_reward_machines
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Install SWI-Prolog in one of these ways:

- Recommended for GitHub: install SWI-Prolog on your machine so `swipl` is available on `PATH`.
- Local macOS convenience: keep a local `legacy/SWI-Prolog.app`, but do not commit it.

3. Run the inherited smoke test from `legacy/`:

```bash
cd legacy
./smoke_test_letterenv.sh
```

## Notes

- The monitor launch scripts in `legacy/RML/` first try `swipl` from your
  system `PATH`.
- If `swipl` is not installed globally, the scripts fall back to a sibling
  `SWI-Prolog.app` bundle in `legacy/`.
- `SWI-Prolog.app` is intentionally gitignored so the repository stays
  portable.
- The inherited code still contains older Gym/Gymnasium compatibility quirks,
  so some warnings during resets and steps are expected until the wrappers are
  modernized.
