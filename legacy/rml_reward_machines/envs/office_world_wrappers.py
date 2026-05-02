from envs.grid_environment import GridEnv, GridEnv_RNN
from envs.office_world import OfficeWorld, OfficeWorld_Delivery


class OfficeRMLEnv(GridEnv):
    metadata = {'render_modes': [22]}
    def __init__(self, render_mode = None):
        self.N = 5
        self._initialize_env()
        config_path = './examples/office.yaml'
        super().__init__(self.env, config_path, monitor_states=20, render_mode=render_mode)

    def _initialize_env(self):
        self.env = OfficeWorld(
            task_string="E" * self.N + "O" + "F" * self.N + "G" * self.N,
            max_observation_counts={
            "E": self.N,
            "A": None,
            "B": None,
            "C": None,
            "D": None,
            "F": None,
            "G": None,
            "O": None,
            "N": None
        }
        )

    def set_n(self, n):
        self.N = n
        self._initialize_env()


class OfficeRMLEnv_Delivery(GridEnv):
    metadata = {'render_modes': [22]}
    def __init__(self, render_mode = None):
        self.N = 5
        self._initialize_env()
        config_path = './examples/office.yaml'
        super().__init__(self.env, config_path, monitor_states=20, render_mode=render_mode)

    def _initialize_env(self):
        self.env = OfficeWorld_Delivery(
            task_string="E" + "F" * self.N + "G" * self.N
        )

    def set_n(self, n):
        self.N = n
        self._initialize_env()

