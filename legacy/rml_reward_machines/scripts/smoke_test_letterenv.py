import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gymnasium.envs.registration import register, registry

from envs.property_envs.letterenv_numerical import Actions
from rml.rmlgym import RMLGym_Simple


def ensure_env_registered() -> None:
    env_id = "letter-env"
    entry_point = "envs.property_envs.letterenv_numerical_wrappers:RML_LetterEnv_numerical_4_Simple"
    if env_id not in registry:
        register(id=env_id, entry_point=entry_point, max_episode_steps=200)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a short RML letter environment smoke test.")
    parser.add_argument("--config", required=True, help="Path to the YAML config file to use.")
    parser.add_argument("--steps", type=int, default=5, help="Number of environment steps to execute.")
    args = parser.parse_args()

    ensure_env_registered()

    env = RMLGym_Simple(args.config)
    obs, info = env.reset()

    if not isinstance(obs, dict):
        raise RuntimeError(f"Expected dict observation on reset, got {type(obs)!r}")
    required_keys = {"position", "monitor"}
    missing = required_keys.difference(obs.keys())
    if missing:
        raise RuntimeError(f"Reset observation is missing required keys: {sorted(missing)}")

    actions = [Actions.RIGHT.value, Actions.UP.value, Actions.RIGHT.value, Actions.UP.value]

    print("reset_ok", sorted(obs.keys()))
    for idx in range(args.steps):
        action = actions[idx % len(actions)]
        obs, reward, done, truncated, info = env.step(action)
        print(
            "step",
            idx,
            "reward",
            reward,
            "done",
            done,
            "truncated",
            truncated,
            "monitor_type",
            type(obs.get("monitor")).__name__,
        )

    env.close()
    print("smoke_ok")


if __name__ == "__main__":
    main()
