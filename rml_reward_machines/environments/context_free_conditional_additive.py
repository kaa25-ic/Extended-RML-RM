from gymnasium import make, register
from gymnasium.envs.registration import registry
import random

from environments.wrappers import (
    IdentityWrapper,
    LabellingFunctionWrapper,
    MarkovWrapper,
)


def _create_context_free_env_conditional_additive(N=1):
    n = random.randint(1,N)
    m =random.randint(1,N)
    env_id = f"LetterEnv-Context-Free-Conditional-Additive-N-{n}"

    if n+m > 5:    # Should only work with 3
        string = "A" *n + "B" + "C"*m + "D" + "E"
    else:
        string = "A" *n + "B" + "C"*m + "D" + "B"
    if env_id not in registry:
        register(
            id=env_id,
            entry_point="envs.letterenv:LetterEnv",
            max_episode_steps=100,
            kwargs={
                "n_rows": 6,
                "n_cols": 6,
                "propositions":["A","B","C","D","E"],
                "locations": {
                    "A": (1, 1),
                    "C": (1, 4),
                    "E": (4, 1),
                },
                "max_observation_counts": {
                    "A": n,
                    "B": None,
                    "C": m,
                    "D": None,
                    "E": None
                },
                "replacement_mapping": {"A": "B", "C": "D"},
                "task_string": string,
            },
        )
    return make(f"LetterEnv-Context-Free-Conditional-Additive-N-{n}")


def create_context_free_env_conditional_additive_mdp(N=1):
    return MarkovWrapper(_create_context_free_env_conditional_additive(N))


def create_context_free_env_conditional_additive_pomdp(N=1):
    return IdentityWrapper(_create_context_free_env_conditional_additive(N))


def create_context_free_env_conditional_additive_labelled(N=1):
    return LabellingFunctionWrapper(_create_context_free_env_conditional_additive(N))


mdp_action_space = create_context_free_env_conditional_additive_mdp().action_space
pomdp_action_space = create_context_free_env_conditional_additive_pomdp().action_space
labelled_action_space = create_context_free_env_conditional_additive_labelled().action_space
