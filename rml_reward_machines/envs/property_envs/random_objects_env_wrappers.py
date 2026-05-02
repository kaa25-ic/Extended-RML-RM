
from envs.grid_environment import GridEnv_objects
from envs. property_envs.random_objects_env import RandomObjectsEnv

class RML_RandomObjectCollectionEnv(GridEnv_objects):
    """
    Gor the "Varying Number of Objects" experiment. 
    """

    def __init__(
        self,
        max_objects: int = 3,
        n_rows: int = 4,
        n_cols: int = 4,
        agent_start_location: tuple[int,int] = (0,0),
        max_episode_steps: int = 200,
        number_of_monitor_states: int = 1,
        render_mode = None
    ):
        base_env = RandomObjectsEnv(
            max_objects=max_objects,
            n_rows=n_rows,
            n_cols=n_cols,
            agent_start_location=agent_start_location,
            max_episode_steps=max_episode_steps,
            render_mode=render_mode  # or 'ansi' if desired
        )
        super().__init__(
            env=base_env,
            monitor_states=number_of_monitor_states
        )

    def set_max_objects(self, max_objects):
        """
        Reconfigure the number of objects if desired.
        """
        self.env.max_objects = max_objects


