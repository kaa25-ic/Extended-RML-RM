from gymnasium import make, register
from gymnasium.envs.registration import registry

from environments.wrappers import (
    IdentityWrapper,
    LabellingFunctionWrapper,
    MarkovWrapper,
)
import random

def _create_context_free_env(N=1):
    env_id = f"LetterEnv-Context-Free-Numerical-N-{N}"
    if env_id not in registry:
        register(
            id=env_id,
            entry_point="envs.property_envs.letterenv_numerical:LetterEnv",
            max_episode_steps=200,
            kwargs={
                "n": N,
                "n_rows": 6,
                "n_cols": 6,
                "locations": {
                    "A": (1, 1),
                    "C": (1, 4),
                    "D": (4, 1),
                },
                "max_observation_counts": {
                    "A": 1,
                    "B": None,
                    "C": None,
                    "D": None,
                },
                "replacement_mapping": {"A": "B"},
            },
        )
    return make(f"LetterEnv-Context-Free-Numerical-N-{N}")


def create_context_free_env_mdp(N=1):
    return MarkovWrapper(_create_context_free_env(N))


def create_context_free_env_pomdp(N=1):
    return IdentityWrapper(_create_context_free_env(N))


def create_context_free_env_labelled(N=1):
    return LabellingFunctionWrapper(_create_context_free_env(N))


mdp_action_space = create_context_free_env_mdp().action_space
pomdp_action_space = create_context_free_env_pomdp().action_space
labelled_action_space = create_context_free_env_labelled().action_space
