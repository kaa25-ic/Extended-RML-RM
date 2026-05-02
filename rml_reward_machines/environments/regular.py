from gymnasium import make, register
from gymnasium.envs.registration import registry

from environments.wrappers import (
    IdentityWrapper,
    LabellingFunctionWrapper,
    MarkovWrapper,
)


def _create_regular_env():
    env_id = "LetterEnv-Regular"

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
                    "A": None,
                    "B": None,
                    "C": None,
                },
                "replacement_mapping": None,
                "task_string": "ABC",
            },
        )
    return make(env_id)


def create_regular_env_mdp():
    return MarkovWrapper(_create_regular_env())


def create_regular_env_pomdp():
    return IdentityWrapper(_create_regular_env())


def create_regular_env_labelled():
    return LabellingFunctionWrapper(_create_regular_env())


mdp_action_space = create_regular_env_mdp().action_space
pomdp_action_space = create_regular_env_pomdp().action_space
labelled_action_space = create_regular_env_labelled().action_space
