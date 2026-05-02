from gymnasium import make, register
from gymnasium.envs.registration import registry
from environments.wrappers import (
    IdentityWrapper,
    LabellingFunctionWrapper,
    MarkovWrapper,
)

def _create_office_world_env(N=1):
    env_id = f"office-world-N-{N}"

    if env_id not in registry:
        register(
            id=env_id,
            entry_point="envs.office_world:OfficeWorld",
            max_episode_steps=100,
            kwargs={
                "task_string": "E" * N + "O" + "F" * N + "G" * N,
                "max_observation_counts": {
                    "E": N,
                    "A": None,
                    "B": None,
                    "C": None,
                    "D": None,
                    "F": None,
                    "G": None,
                    "N": None,
                    "O": None
                },
                "replacement_mapping": {"E": "O"},
            },
        )
    return make(f"office-world-N-{N}")

def create_office_world_labelled(N=1):
    return LabellingFunctionWrapper(_create_office_world_env(N))



labelled_action_space = create_office_world_labelled().action_space