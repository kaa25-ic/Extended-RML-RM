import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from enum import Enum


class Actions(Enum):
    RIGHT = 0
    LEFT = 1
    UP = 2
    DOWN = 3

class RandomObjectsEnv(gym.Env):
    """
    An environment that places a random number of objects on the board.
    Each object has a unique integer value from 1 up to the number of objects.
    The agent must:
      1. Collect all objects (stepping on them).
      2. Revisit the location of the object with the highest integer value.
    """

    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(
        self,
        max_objects: int = 6,
        n_rows: int = 5,
        n_cols: int = 5,
        agent_start_location: tuple[int, int] = (0, 0),
        max_episode_steps: int = 50,
        render_mode: str | None = None,
    ):
        """
        Args:
            max_objects: Maximum number of objects to spawn.
            n_rows: Number of rows in the grid.
            n_cols: Number of columns in the grid.
            agent_start_location: Starting (row, col) of the agent.
            max_episode_steps: Force episode to end (truncated) after these many steps if not done.
            render_mode: If "ansi", prints the grid to stdout on `render()`.
        """
        super().__init__()

        self.max_objects = max_objects
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.agent_start_location = agent_start_location
        self._max_episode_steps = max_episode_steps
        self.render_mode = render_mode

        # 4 actions: UP, RIGHT, DOWN, LEFT
        self.action_space = spaces.Discrete(4)

        # Observation: (row, col, current_cell_value)
        # current_cell_value = 0 if empty or collected, else the int value of the object.
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0]),  # Minimum values for [row, col, cell_value, all_collected]
            high=np.array([self.n_rows - 1, self.n_cols - 1, self.max_objects, 1]),  # Max values
            dtype=np.float32
        )


        # Internal tracking
        # (r, c) -> int_value
        self.objects_data = {}
        self.collected_objects = set()
        self.highest_value_location = None
        self.highest_value = None

        self.agent_position = None
        self.n_steps = 0
        self.episode_over = False
        self.all_collected = 0

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        self.n_steps = 0
        self.episode_over = False

        # Random number of objects (1..max_objects)
        num_objects = random.randint(1, self.max_objects)

        # Shuffle possible positions on the grid, excluding the agent's start
        all_positions = [
            (r, c) 
            for r in range(self.n_rows) 
            for c in range(self.n_cols)
            if (r, c) != self.agent_start_location
        ]
        random.shuffle(all_positions)
        chosen_positions = all_positions[:num_objects]

        # Unique integer values from 1..num_objects
        object_values = list(range(1, num_objects + 1))
        random.shuffle(object_values)

        # Assign each chosen position a unique integer
        self.objects_data.clear()
        for pos, val in zip(chosen_positions, object_values):
            self.objects_data[pos] = val

        # Identify the object with the highest integer
        self.highest_value_location = None
        self.highest_value = -1
        for pos, val in self.objects_data.items():
            if val > self.highest_value:
                self.highest_value = val
                self.highest_value_location = pos

        self.collected_objects.clear()

        # Place agent at start
        self.agent_position = self.agent_start_location
        self.all_collected = 0

        return self._get_observation(), {}

    def step(self, action):
        """
        Moves the agent, applies step penalty (-1),
        checks whether the agent collected an object,
        and checks whether the task is completed.
        """
        if self.episode_over:
            # If already ended, just return same observation and no reward.
            return self._get_observation(), 0, True, True, {}

        self.n_steps += 1

        # Default reward = -1 each step (step penalty)
        reward = -1
        done = False
        truncated = False

        # Move agent and get obs
        self._move_agent(action)
        obs = self._get_observation()

        # Check if agent is on an object location
        if self.agent_position in self.objects_data:
            # If not collected yet, collect it
            if self.agent_position not in self.collected_objects:
                self.collected_objects.add(self.agent_position)

        # Check if all objects are collected
        self.all_collected = int(len(self.collected_objects) == len(self.objects_data))

        # If all collected, next check: did they revisit the highest-value object location?
        if self.all_collected:
            if self.agent_position == self.highest_value_location:
                # Success, give reward=1 (relative to the step penalty).
                reward = 1
                done = True

        # Check for time-based truncation
        if self.n_steps >= self._max_episode_steps:
            truncated = True
            done = True

        self.episode_over = done or truncated
        return obs , reward, done, truncated, {}

    def render(self):
        """
        Renders the environment in ASCII if render_mode='ansi'.
        Displays:
          - 'A' for the agent
          - each object's integer value if not collected
          - '_' if collected
          - '.' if empty
        """
        if self.render_mode == "ansi":
            grid_str = ""
            for r in range(self.n_rows):
                row_str = ""
                for c in range(self.n_cols):
                    if (r, c) == self.agent_position:
                        row_str += "A "
                    elif (r, c) in self.objects_data:
                        val = self.objects_data[(r, c)]
                        if (r, c) in self.collected_objects:
                            row_str += "_ "
                        else:
                            row_str += f"{val} "
                    else:
                        row_str += ". "
                grid_str += row_str + "\n"
            print(grid_str)

    def _move_agent(self, action):
        r, c = self.agent_position

        if action == 0:  # UP
            r -= 1
        elif action == 1:  # RIGHT
            c += 1
        elif action == 2:  # DOWN
            r += 1
        elif action == 3:  # LEFT
            c -= 1

        # Clamp the agent's position within grid bounds
        self.agent_position = (
            max(0, min(r, self.n_rows - 1)),
            max(0, min(c, self.n_cols - 1)),
        )


    def _get_observation(self):
        """
        Returns a numpy array [row, col, cell_value].
        """
        r, c = self.agent_position
        if (r, c) in self.objects_data:
            val = self.objects_data[(r, c)]
            if (r, c) in self.collected_objects:
                cell_value = 0
            else:
                cell_value = val
        else:
            cell_value = 0

        # Return as a numpy array
        return np.array([r, c, cell_value,self.all_collected], dtype=np.float32)

