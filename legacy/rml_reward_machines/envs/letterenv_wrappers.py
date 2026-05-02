from envs.grid_environment import GridEnv, GridEnv_RNN, GridEnv_Original
from envs.grid_environment_old import GridEnv_Old
from envs.letterenv import LetterEnv
import random

class RML_LetterEnv_5(GridEnv):
    metadata = {'render_modes': [22]}
    def __init__(self, number_of_monitor_states = 20, render_mode = None):
        self.N = 5
        self._initialize_env()
        config_path = './examples/letter_env.yaml'
        super().__init__(self.env, config_path, monitor_states=number_of_monitor_states, render_mode = render_mode)
    
    def _initialize_env(self):
        self.env = LetterEnv(
            n_rows=6,
            n_cols=6,
            locations={
                "A": (1, 1),
                "B": (1, 4),
                "C": (4, 1),
            },
            max_observation_counts={
                "A": self.N,
                "B": None,
                "E": None,
                "C": None,
            },
            replacement_mapping={"A": "E"},
            task_string="A" * self.N + "E" + "B" + "C" * self.N
        )

    def evaluation_start(self):
        self.env.evaluation_n = True
    
    def evaluation_end(self):
        self.env.evaluation_n = False
    
    def set_n(self, n):
        self.N = n
        self._initialize_env()

class RML_LetterEnv_5_Simple(RML_LetterEnv_5):
    metadata = {'render_modes': [22]}
    def __init__(self, render_mode = None):  # Only change from normal is monitor_states = 1 instead of 20
        super().__init__(number_of_monitor_states=1, render_mode = render_mode)


class Letter_RMLEnv_Old(GridEnv_Old):
    metadata = {'render_modes': [22]}
    def __init__(self, render_mode = None):
        self.N = 5
        self._initialize_env()
        config_path = './examples/letter_env.yaml'
        super().__init__(self.env, config_path, render_mode)

    def _initialize_env(self):
        self.env = LetterEnv(
            n_rows=6,
            n_cols=6,
            locations={
                "A": (1, 1),
                "B": (1, 4),
                "C": (4, 1),
            },
            max_observation_counts={
                "A": self.N,
                "B": None,
                "E": None,
                "C": None,
            },
            replacement_mapping={"A": "E"},
            task_string="A" * self.N + "E" + "B" + "C" * self.N
        )

    def set_n(self, n):
        self.N = n
        self._initialize_env()


class RML_LetterEnv_Conditional_Simple(GridEnv):
    metadata = {'render_modes': [22]}
    def __init__(self, number_of_monitor_states = 1, N = 5, render_mode = None):
        self.N = N
        self._initialize_env()
        config_path = './examples/letter_env.yaml'
        super().__init__(self.env, config_path, monitor_states=number_of_monitor_states, render_mode = render_mode)
    
    def _initialize_env(self):
        random_n = random.randint(1,self.N)
        if random_n > self.N/2:
            string = "A" * random_n + "B" + "D"
        else:
            string = "A" * random_n + "B" + "C"
        self.env = LetterEnv(
            n_rows=6,
            n_cols=6,
            propositions=["A","B","C","D"],
            locations={
                "A": (1, 1),
                "C": (1, 4),
                "D": (4, 1),
            },
            max_observation_counts={
                "A": random_n,
                "B": None,
                "C": None,
                "D": None,
            },
            replacement_mapping={"A": "B"},
            task_string=string
        )

    def evaluation_start(self):
        self.env.evaluation_n = True
    
    def evaluation_end(self):
        self.env.evaluation_n = False
    
    def set_n(self, n):
        self.N = n
        self._initialize_env()

class RML_LetterEnv_Conditional_Multiplicative(GridEnv):
    metadata = {'render_modes': [22]}
    def __init__(self, number_of_monitor_states = 1, N = 5, render_mode = None):
        self.N = N
        self._initialize_env()
        config_path = './examples/letter_env.yaml'
        super().__init__(self.env, config_path, monitor_states=number_of_monitor_states, render_mode = render_mode)
    
    def _initialize_env(self):
        random_n = random.randint(1,self.N)
        random_m = random.randint(1,self.N)
        if random_n*random_m > 6:
            string = "A" * random_n + "B" + "C"*random_m + "D" + "E"
        else:
            string = "A" * random_n + "B" + "C"*random_m + "D" + "B"
        self.env = LetterEnv(
            n_rows=6,
            n_cols=6,
            propositions=["A","B","C","D","E"],
            locations={
                "A": (1, 1),
                "C": (1, 4),
                "E": (4, 1),
            },
            max_observation_counts={
                "A": random_n,
                "B": None,
                "C": random_m,
                "D": None,
                "E": None
            },
            replacement_mapping={"A": "B", "C": "D"},
            task_string=string
        )

    def evaluation_start(self):
        self.env.evaluation_n = True
    
    def evaluation_end(self):
        self.env.evaluation_n = False
    
    def set_n(self, n):
        self.N = n
        self._initialize_env()


class RML_LetterEnv_Conditional_Additive(GridEnv):
    metadata = {'render_modes': [22]}
    def __init__(self, number_of_monitor_states = 1, N = 3, render_mode = None):
        self.N = N
        self._initialize_env()
        config_path = './examples/letter_env.yaml'
        super().__init__(self.env, config_path, monitor_states=number_of_monitor_states, render_mode = render_mode)
    
    def _initialize_env(self):
        random_n = random.randint(1,self.N)
        random_m = random.randint(1,self.N)
        if random_n + random_m > 3:
            string = "A" * random_n + "B" + "C"*random_m + "D" + "E"
        else:
            string = "A" * random_n + "B" + "C"*random_m + "D" + "B"
        self.env = LetterEnv(
            n_rows=6,
            n_cols=6,
            propositions=["A","B","C","D","E"],
            locations={
                "A": (1, 1),
                "C": (1, 4),
                "E": (4, 1),
            },
            max_observation_counts={
                "A": random_n,
                "B": None,
                "C": random_m,
                "D": None,
                "E": None
            },
            replacement_mapping={"A": "B", "C": "D"},
            task_string=string
        )

    def evaluation_start(self):
        self.env.evaluation_n = True
    
    def evaluation_end(self):
        self.env.evaluation_n = False
    
    def set_n(self, n):
        self.N = n
        self._initialize_env()

class RML_LetterEnv_Conditional_Simple_Original(GridEnv_Original):
    metadata = {'render_modes': [22]}
    def __init__(self, N = 5, render_mode = None):
        self.N = N
        self._initialize_env()
        config_path = './examples/letter_env.yaml'
        super().__init__(self.env, config_path, render_mode = render_mode)
    
    def _initialize_env(self):
        random_n = random.randint(1,self.N)
        if random_n > self.N/2:
            string = "A" * random_n + "B" + "D"
        else:
            string = "A" * random_n + "B" + "C"
        self.env = LetterEnv(
            n_rows=6,
            n_cols=6,
            propositions=["A","B","C","D"],
            locations={
                "A": (1, 1),
                "C": (1, 4),
                "D": (4, 1),
            },
            max_observation_counts={
                "A": random_n,
                "B": None,
                "C": None,
                "D": None,
            },
            replacement_mapping={"A": "B"},
            task_string=string
        )

    def evaluation_start(self):
        self.env.evaluation_n = True
    
    def evaluation_end(self):
        self.env.evaluation_n = False
    
    def set_n(self, n):
        self.N = n
        self._initialize_env()