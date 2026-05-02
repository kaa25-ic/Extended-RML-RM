from gymnasium import make, register
from gymnasium.envs.registration import registry

from environments.wrappers import (
    IdentityWrapper,
    LabellingFunctionWrapper,
    MarkovWrapper,
)
import random

def _create_random_objects_env(N=1):
    env_id = f"RandomObjectsEnv-N-{N}"
    if env_id not in registry:
        register(
            id=env_id,
            entry_point="envs.property_envs.random_objects_env:RandomObjectsEnv",
            max_episode_steps=100,
            kwargs={
                "n_rows": 6,
                "n_cols": 6,
                "n": N,
            },
        )
    return make(f"RandomObjectsEnv-N-{N}")


def create_random_objects_env_mdp(N=1):
    return MarkovWrapper(_create_random_objects_env(N))


def create_random_objects_env_pomdp(N=1):
    return IdentityWrapper(_create_random_objects_env(N))


def create_random_objects_env_labelled(N=1):
    return LabellingFunctionWrapper(_create_random_objects_env(N))


mdp_action_space = create_random_objects_env_mdp().action_space
pomdp_action_space = create_random_objects_env_pomdp().action_space
labelled_action_space = create_random_objects_env_labelled().action_space
