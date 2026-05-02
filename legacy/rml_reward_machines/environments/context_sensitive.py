from gymnasium import make, register
from gymnasium.envs.registration import registry

from environments.wrappers import (
    IdentityWrapper,
    LabellingFunctionWrapper,
    MarkovWrapper,
)


def _create_context_sensitive_env(N=1):
    env_id = f"LetterEnv-Context-Sensitive-N-{N}"

    if env_id not in registry:
        register(
            id=env_id,
            entry_point="letterenv:LetterEnv",
            max_episode_steps=100,
            kwargs={
                "n_rows": 6,
                "n_cols": 6,
                "locations": {
                    "A": (1, 1),
                    "B": (1, 4),
                    "C": (4, 1),
                },
                "max_observation_counts": {
                    "A": N,
                    "B": None,
                    "E": None,
                    "C": None,
                },
                "replacement_mapping": {"A": "E"},
                "task_string": "A" * N + "E" + "B" * N + "C" * N,
            },
        )
    return make(env_id)


def create_context_sensitive_env_mdp(N=1):
    return MarkovWrapper(_create_context_sensitive_env(N))


def create_context_sensitive_env_pomdp(N=1):
    return IdentityWrapper(_create_context_sensitive_env(N))


def create_context_sensitive_env_labelled(N=1):
    return LabellingFunctionWrapper(_create_context_sensitive_env(N))


mdp_action_space = create_context_sensitive_env_mdp().action_space
pomdp_action_space = create_context_sensitive_env_pomdp().action_space
labelled_action_space = create_context_sensitive_env_labelled().action_space
