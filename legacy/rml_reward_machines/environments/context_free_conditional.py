from gymnasium import make, register
from gymnasium.envs.registration import registry
import random

from environments.wrappers import (
    IdentityWrapper,
    LabellingFunctionWrapper,
    MarkovWrapper,
)


def _create_context_free_env_conditional(N=1):
    n = random.randint(1,N)
    env_id = f"LetterEnv-Context-Free-Conditional-N-{n}"

    if n > N/2:
        string = "A"*n + "B" + "D"
    else:
        string = "A"*n + "B" + "C"
    if env_id not in registry:
        register(
            id=env_id,
            entry_point="envs.letterenv:LetterEnv",
            max_episode_steps=100,
            kwargs={
                "n_rows": 6,
                "n_cols": 6,
                "propositions":["A","B","C","D"],
                "locations": {
                    "A": (1, 1),
                    "C": (1, 4),
                    "D": (4, 1),
                },
                "max_observation_counts": {
                    "A": n,
                    "B": None,
                    "C": None,
                    "D": None,
                },
                "replacement_mapping": {"A": "B"},
                "task_string": string,
            },
        )
    return make(f"LetterEnv-Context-Free-Conditional-N-{n}")


def create_context_free_env_conditional_mdp(N=1):
    return MarkovWrapper(_create_context_free_env_conditional(N))


def create_context_free_env_conditional_pomdp(N=1):
    return IdentityWrapper(_create_context_free_env_conditional(N))


def create_context_free_env_conditional_labelled(N=1):
    return LabellingFunctionWrapper(_create_context_free_env_conditional(N))


mdp_action_space = create_context_free_env_conditional_mdp().action_space
pomdp_action_space = create_context_free_env_conditional_pomdp().action_space
labelled_action_space = create_context_free_env_conditional_labelled().action_space
