import numpy as np
from gymnasium import Env, spaces
from enum import Enum
import random

class Actions(Enum):
    RIGHT = 0
    LEFT = 1
    UP = 2
    DOWN = 3


class LetterEnv(Env):
    """Letter environment."""

    RIGHT = 0
    LEFT = 1
    UP = 2
    DOWN = 3

    metadata = {"render_modes": ["ansi"], "render_fps": 1}

    def __init__(
        self,
        n: int = 3,
        n_rows: int = 5,
        n_cols: int = 5,
        propositions: list[str] = ["A", "B", "C", "D"],
        locations: dict[str, tuple[int, int]] = {"A": (1, 1), "C": (0, 4), "D": (4, 0)},
        agent_start_location: tuple[int, int] = (4, 4),
        max_observation_counts: dict[str, int | None] = {
            "A": 1,
            "B": None,
            "C": None,
            "D": None,
        },
        replacement_mapping: dict[str, str] = {"A": "B"},
    ) -> None:
        super().__init__()

        # Setup environment configuation
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.propositions = propositions
        self.locations = locations
        self.agent_start_location = agent_start_location
        self.max_observation_counts = max_observation_counts
        self.replacement_mapping = replacement_mapping
        self.prop_idx = {p: i for i, p in enumerate(self.propositions)}
        self.max_n = n
        self.n = random.randint(1,self.max_n)
        self.task_string = "ABC" + "D"*self.n


        # Define environment spaces
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Tuple(
            spaces=(
                spaces.Discrete(self.n_rows),
                spaces.Discrete(self.n_cols),
                spaces.Discrete(len(self.propositions) + self.max_n)
            ),
        )
        self.reward_range = (0, 1)
        self.create_forbidden_transitions()

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ):
        super().reset(seed=seed)
        # Reset number of steps taken in environment
        self.n_steps = 0
        self.n = random.randint(1,self.max_n)
        self.task_string = "ABC" + "D"*self.n

        self.task_string_idx = 0
        self.task_failed = False

        # Set number of times each proposition has been observed to 0
        self.prop_obs_counts = np.zeros((len(self.propositions),))

        # Define active propositions for each environment location
        self.active_propositions = {pos: p for p, pos in self.locations.items()}

        # Set agent initial position
        self.agent_position = self.agent_start_location
        if self.agent_position in self.active_propositions:
            obs_prop = self.active_propositions[self.agent_position]
            if obs_prop == 'A':
                obs_prop = self.n
        else:
            obs_prop = "_"
        obs = (
            self.agent_position[0],
            self.agent_position[1],
            obs_prop,
        )
        return obs, {}

    def step(self, action: int):
        self.n_steps += 1
        # Move agent in environment
        self._update_agent_position(action)
        # Calculate which propositions are true in the environment
        if self.agent_position in self.active_propositions:
            obs_prop = self.active_propositions[self.agent_position]

            # Update number of times proposition has been observed
            prop_idx = self.prop_idx[obs_prop]
            self.prop_obs_counts[prop_idx] += 1

            if self.prop_obs_counts[prop_idx] == self.max_observation_counts[obs_prop]:
                # Replace proposition with next proposition in replacement mapping
                self.active_propositions[
                    self.agent_position
                ] = self.replacement_mapping[obs_prop]

            try:
                if obs_prop == self.task_string[self.task_string_idx]:
                    self.task_string_idx += 1
                else:
                    self.task_failed = True
            except IndexError:
                pass

            if obs_prop == 'A':
                obs_prop = self.n
        else:
            obs_prop = "_"

        obs = (
            self.agent_position[0],
            self.agent_position[1],
            obs_prop,
        )
        # Determine if episode is terminated due to max number of steps
        if self.spec is not None and self.spec.max_episode_steps == self.n_steps:
            terminated = True
            # Episode ended based on condition outside MDP (interface for more details)
            truncated = True
            reward = 0
        else:
            if self.task_failed:
                terminated = True
                truncated = False
                reward = 0
            elif self.task_string_idx == len(self.task_string):
                terminated = True
                truncated = False
                reward = 1
            else:
                terminated = False
                truncated = False
                reward = 0

        return (
            obs,
            reward,
            terminated,
            truncated,
            {},
        )

    def render(self) -> str:
        """Render the environment as a string."""
        str_repr = ""

        for r in range(self.n_rows):
            for c in range(self.n_cols):
                if (r, c) == self.agent_position:
                    str_repr += "\x1b[1;37;42m" + "x" + "\x1b[0m" + " "
                elif (r, c) in self.active_propositions:
                    str_repr += self.active_propositions[(r, c)] + " "
                else:
                    str_repr += "." + " "
            str_repr += "\n"
        return str_repr

    def _update_agent_position(self, action: int) -> None:
        """Moves that take agent out of the grid leave it in the same position."""
        row, col = self.agent_position

        if action == self.RIGHT:
            n_row = row
            n_col = col + 1 if col < self.n_cols - 1 else col
        elif action == self.LEFT:
            n_row = row
            n_col = col - 1 if col > 0 else col
        elif action == self.UP:
            n_col = col
            n_row = row - 1 if row > 0 else row
        elif action == self.DOWN:
            n_col = col
            n_row = row + 1 if row < self.n_rows - 1 else row
        else:
            raise ValueError(f"Invalid action {action}.")
        self.agent_position = (n_row, n_col)

    def _construct_observation(self, obs_prop):
        if self.agent_position in self.active_propositions:
            obs_props = self.active_propositions[self.agent_position]
            if obs_props == 'A':
                obs_props = self.n
        else:
            obs_props = "_"

        return (
            self.agent_position[0],
            self.agent_position[1],
            obs_props,
        )
    
    def create_forbidden_transitions(self):
        self.forbidden_transitions = set()
        # Add forbidden transitions for the bottom and top walls
        for x in range(self.n_cols):
            self.forbidden_transitions.add((0, x, Actions.UP))  # Bottom wall
            self.forbidden_transitions.add((self.n_rows-1, x, Actions.DOWN))  # Top wall

        # Add forbidden transitions for the left and right walls
        for y in range(self.n_rows):
            self.forbidden_transitions.add((y, 0, Actions.LEFT))  # Left wall
            self.forbidden_transitions.add((y, self.n_cols-1, Actions.RIGHT))  # Right wall
