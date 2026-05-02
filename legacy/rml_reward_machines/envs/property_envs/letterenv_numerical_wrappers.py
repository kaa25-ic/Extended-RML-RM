from envs.grid_environment import GridEnv_numerical, GridEnv_numerical_Original
from envs.grid_environment_old import GridEnv_Old
from envs.property_envs.letterenv_numerical import LetterEnv

class RML_LetterEnv_numerical_3(GridEnv_numerical):
    metadata = {'render_modes': [22]}
    def __init__(self, number_of_monitor_states = 20, render_mode = None):
        self.N = 3
        self._initialize_env()
        config_path = './examples/letter_env.yaml'
        super().__init__(self.env, config_path, monitor_states=number_of_monitor_states, render_mode = render_mode)
    
    def _initialize_env(self):
        self.env = LetterEnv(
            n=self.N,
            n_rows=6,
            n_cols=6,
            locations={
                "A": (1, 1),
                "C": (1, 4),
                "D": (4, 1),
            },
            agent_start_location = (4, 4),
            max_observation_counts={
                "A": 1,
                "B": None,
                "C": None,
                "D": None,
            },
            replacement_mapping={"A": "B"}
        )

    def evaluation_start(self):
        self.env.evaluation_n = True
    
    def evaluation_end(self):
        self.env.evaluation_n = False
    
    def set_n(self, n):
        self.N = n
        self._initialize_env()

class RML_LetterEnv_numerical_3_Simple(RML_LetterEnv_numerical_3):
    metadata = {'render_modes': [22]}
    def __init__(self, render_mode = None):  # Only change from normal is monitor_states = 1 instead of 20
        super().__init__(number_of_monitor_states=1, render_mode = render_mode)


class RML_LetterEnv_numerical_4(GridEnv_numerical):
    metadata = {'render_modes': [22]}
    def __init__(self, number_of_monitor_states = 20, render_mode = None):
        self.N = 4
        self._initialize_env()
        config_path = './examples/letter_env.yaml'
        super().__init__(self.env, config_path, monitor_states=number_of_monitor_states, render_mode = render_mode)
    
    def _initialize_env(self):
        self.env = LetterEnv(
            n=self.N,
            n_rows=6,
            n_cols=6,
            locations={
                "A": (1, 1),
                "C": (1, 4),
                "D": (4, 1),
            },
            agent_start_location = (4, 4),
            max_observation_counts={
                "A": 1,
                "B": None,
                "C": None,
                "D": None,
            },
            replacement_mapping={"A": "B"}
        )

    def evaluation_start(self):
        self.env.evaluation_n = True
    
    def evaluation_end(self):
        self.env.evaluation_n = False
    
    def set_n(self, n):
        self.N = n
        self._initialize_env()

class RML_LetterEnv_numerical_4_Simple(RML_LetterEnv_numerical_4):
    metadata = {'render_modes': [22]}
    def __init__(self, render_mode = None):  # Only change from normal is monitor_states = 1 instead of 20
        super().__init__(number_of_monitor_states=1, render_mode = render_mode)

class RML_LetterEnv_numerical_Original(GridEnv_numerical_Original):
    metadata = {'render_modes': [22]}
    def __init__(self, render_mode = None):
        self.N = 3
        self._initialize_env()
        config_path = './examples/letter_env.yaml'
        super().__init__(self.env, config_path, render_mode = render_mode)

    def _initialize_env(self):
        self.env = LetterEnv(
            n=self.N,
            n_rows=6,
            n_cols=6,
            locations={
                "A": (1, 1),
                "C": (1, 4),
                "D": (4, 1),
            },
            agent_start_location = (4, 4),
            max_observation_counts={
                "A": 1,
                "B": None,
                "C": None,
                "D": None,
            },
            replacement_mapping={"A": "B"}
        )

    def evaluation_start(self):
        self.env.evaluation_n = True
    
    def evaluation_end(self):
        self.env.evaluation_n = False
    
    def set_n(self, n):
        self.N = n
        self._initialize_env()