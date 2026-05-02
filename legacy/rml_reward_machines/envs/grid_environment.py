import gymnasium as gym
from gymnasium import spaces
import numpy as np
from envs.office_world import OfficeWorld, OfficeWorld_Delivery, Actions
from envs.property_envs.random_objects_env import RandomObjectsEnv
from envs.letterenv import LetterEnv
import yaml
import copy

class GridEnv(gym.Env):
    def __init__(self, env, config_path, monitor_states = 20, render_mode = None):   #  observation_space_siz Default is 2 as it's (x,y). Else it adds the number of monitor states as well
        self.env = env
        self.propositions = self.env.propositions
        self.propositions.append("_")
        self.one_hot_propositions = self.generate_one_hot_propisition(self.propositions)

        N,M      = self.env.n_rows, self.env.n_cols
        L = len(self.propositions)
        self.action_space = spaces.Discrete(4) # up, right, down, left
        self.position_space = spaces.Box(low=0, high=max([N, M]), shape=(L,), dtype=np.uint8)  # (x, y) + objects
        self.monitor_space = spaces.Box(low=-np.inf, high=np.inf, shape=(monitor_states,), dtype=np.float32)  # Monitor state of size 20

        self.observation_space = spaces.Dict({
            'position': self.position_space,
            'monitor': self.monitor_space
        })        

        with open(config_path, "r") as stream:
            try:    
                config_dict = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        self.max_steps = config_dict.get('max_episode_steps', 200)  # Default to 200 if not specified
        self.step_num = 0
        self.locations = copy.deepcopy(self.env.locations)

    def generate_one_hot_propisition(self,propositions):
        proposition_dict = {}
        i = 0
        for prop in propositions:
            proposition_dict[prop] = i
            i += 1
        return proposition_dict


    def get_events(self):
        return self.env.get_true_propositions()

    def step(self, action):
        self.step_num += 1
        o = self.env.step(action)
        obs_tuple = o[0]
        obs_position = self.obs_from_tuple(obs_tuple)
        obs = {
            'position': obs_position,
            'monitor': self.monitor_state
        }
        done = o[2]
        truncated = o[3]

        reward = 0  # reward done at monitor_reward level
        #reward = super().monitor_reward(done, truncated) # all the reward comes from the RM
        info = {}
        return obs, reward, done, truncated, info

    def reset(self):
        self.step_num = 0
        o, _ = self.env.reset()
        obs_position = self.obs_from_tuple(o)
        obs = {
            'position': obs_position,
            'monitor': self.monitor_state
        }
        return obs

    def show(self):
        self.env.show()

    def get_model(self):
        return self.env.get_model()
    
    def render(self, mode='agent'):
        if mode == 'human':
            # commands
            str_to_action = {"w":0,"d":1,"s":2,"a":3}

            # play the game!
            done = True
            while True:
                if done:
                    print("New episode --------------------------------")
                    obs = self.reset()
                    self.env.show()
                    print("Features:", obs)
                    print("Events:", self.get_events())

                print("\nAction? (WASD keys or q to quite) ", end="")
                a = input()
                print()
                if a == 'q':
                    break
                # Executing action
                if a in str_to_action:
                    obs, rew, done, _, __ = self.step(str_to_action[a])
                    self.env.show()
                    print("Features:", obs)
                    print("Reward:", rew)
                    print("Events:", self.get_events())
                else:
                    print("Forbidden action")
        elif mode == 'agent':
            return self._render_agent()
        else:
            raise NotImplementedError
        
    def _render_agent(self):
        self.env.show()
    
    def obs_from_tuple(self,observation_tuple):
        obs = list(observation_tuple[:2])
        proposition_one_hot = [0]*len(self.propositions)
        index_of_proposition = self.one_hot_propositions[observation_tuple[2]]
        proposition_one_hot[index_of_proposition] = 1
        obs = obs + proposition_one_hot
        return tuple(obs)
    
    def get_monitor_state(self, state):
        self.monitor_state = state

class GridEnv_RNN(GridEnv):
    def __init__(self, env, config_path, observation_space_size = 2, render_mode = None, max_monitor_length=12):   #  observation_space_siz Default is 2 as it's (x,y). Else it adds the number of monitor states as well
        #super().__init__(config_path = './examples/office.yaml')
        self.env = env
        N,M, L      = self.env.map_height, self.env.map_width, len(self.env.object_list) + 2  # + 2 for x,y axis
        self.action_space = spaces.Discrete(4) # up, right, down, left

        # Observation spaces
        self.position_space = spaces.Box(low=0, high=max([N, M]), shape=(L,), dtype=np.uint8)  # (x, y) + objects
        self.monitor_space = spaces.Box(low=-np.inf, high=np.inf, shape=(max_monitor_length,20), dtype=np.float32)  # Monitor state of size 20

        self.observation_space = spaces.Dict({
            'position': self.position_space,
            'monitor': self.monitor_space
        })        
        
        #self.observation_dict  = spaces.Dict({'features': self.observation_space})
        
        with open(config_path, "r") as stream:
            try:    
                config_dict = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        self.max_steps = config_dict.get('max_episode_steps', 200)  # Default to 200 if not specified
        self.step_num = 0  

    def reset(self):
        self.env.reset()
        self.step_num = 0
        self.monitor_state = self.env.monitor_state
        return {
            'position': self.env.get_features(),
            'monitor': self.monitor_state
        }    

    def step(self, action):
        self.env.execute_action(action)
        obs = self._get_observation()
        self.step_num += 1
        if self.step_num < self.max_steps:
            truncated = False
        else:
            truncated = True
        done = truncated

        reward = 0  # reward done at monitor_reward level
        info = {}
        return obs, reward, done, truncated, info

    def _get_observation(self):
        position = self.env.get_features()
        monitor = self.env.monitor_state
        return {
            'position': position,
            'monitor': monitor
        }
    
class GridEnv_numerical(GridEnv):
    def obs_from_tuple(self,observation_tuple):
        obs = list(observation_tuple[:2])
        proposition_one_hot = [0]*len(self.propositions)
        prop = observation_tuple[2]
        prop_val = 1     # What the value gets set to at the index
        changed = False
        if isinstance(prop, int):
            prop_val = copy.deepcopy(prop)
            for key, value in self.locations.items():
                if value == self.env.agent_position:
                    prop = key
                    changed = True
            if changed == False:
                for key, value in self.env.locations.items():
                    if value == self.env.agent_position:
                        prop = key
                        changed = True
            if changed == False:
                print('change errror!!!!!!!!!!!!!!!!!!!!!!!!!!!')

        index_of_proposition = self.one_hot_propositions[prop]
        proposition_one_hot[index_of_proposition] = prop_val
        obs = obs + proposition_one_hot

        return tuple(obs)
    
class GridEnv_random(GridEnv):
    def __init__(self, env, config_path, monitor_states = 20, render_mode = None):   #  observation_space_siz Default is 2 as it's (x,y). Else it adds the number of monitor states as well
        self.env = env
        self.propositions = self.env.propositions
        self.propositions.append("_")
        self.one_hot_propositions = self.generate_one_hot_propisition(self.propositions)

        N,M      = self.env.n_rows, self.env.n_cols
        L = N * M + 2  # n_rows * n_cols + 2 (one for y and one for prop)
        self.action_space = spaces.Discrete(4) # up, right, down, left
        self.position_space = spaces.Box(low=-1, high=self.env.N, shape=(L,), dtype=np.int8)
        self.monitor_space = spaces.Box(low=-np.inf, high=np.inf, shape=(monitor_states,), dtype=np.int8)  # Monitor state of size 20

        self.observation_space = spaces.Dict({
            'position': self.position_space,
            'monitor': self.monitor_space
        })        

        with open(config_path, "r") as stream:
            try:    
                config_dict = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        self.max_steps = config_dict.get('max_episode_steps', 100)  # Default to 200 if not specified
        self.step_num = 0
        self.locations = copy.deepcopy(self.env.locations)

    def obs_from_tuple(self,observation_tuple):
        obs = list(observation_tuple[0])   # This is the grid
        proposition = [observation_tuple[2]]
        obs = obs + [0] + proposition
        return obs
    
class GridEnv_objects(gym.Env):
    """
    A simple wrapper that uses the underlying RandomObjectsEnv but creates
    a Dict observation space with 'position' and 'monitor' (like your previous wrappers).
    This is purely illustrative – adapt as you wish!
    """

    def __init__(
        self,
        env: RandomObjectsEnv,
        monitor_states: int = 1
    ):
        super().__init__()
        self.env = env
        self.action_space = self.env.action_space

        # We'll define the 'position' space and a 'monitor' space:
        # You can adapt these to your own logic, e.g. one-hot, etc.
        self.position_space = spaces.Box(
            low=np.array([0, 0, 0, 0]),  # [min_row, min_col, min_value, all_collected]
            high=np.array([
                self.env.n_rows - 1,  # max_row
                self.env.n_cols - 1,  # max_col
                self.env.max_objects,  # max_value (from RandomObjectsEnv)
                1                      # all_collected is boolean
            ]),
            dtype=np.float32
        )

        self.monitor_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(monitor_states,),
            dtype=np.float32
        )

        self.observation_space = spaces.Dict({
            'position': self.position_space,
            'monitor': self.monitor_space
        })


    def reset(self, seed: int | None = None, options: dict | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        return self._convert_obs(obs)

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        return self._convert_obs(obs), reward, done, truncated, info

    def render(self):
        return self.env.render()

    def _convert_obs(self, obs):
        """
        obs is (row, col, last_prop, all_collected).
        We'll just feed that into 'position', and supply a 'monitor' as well.
        """
        row, col, prop_val, all_collected = obs

        pos_array = np.array([row, col, prop_val, all_collected], dtype=np.float32)
        return {
            'position': pos_array,
            'monitor': self.monitor_state
        }

    def get_monitor_state(self, state):
        self.monitor_state = state

class GridEnv_Original(GridEnv):
    def __init__(self, env, config_path, render_mode = None):   #  observation_space_siz Default is 2 as it's (x,y). Else it adds the number of monitor states as well
        self.env = env
        self.propositions = self.env.propositions
        self.propositions.append("_")
        self.one_hot_propositions = self.generate_one_hot_propisition(self.propositions)

        N,M      = self.env.n_rows, self.env.n_cols
        L = len(self.propositions)
        self.action_space = spaces.Discrete(4) # up, right, down, left
        self.position_space = spaces.Box(low=0, high=max([N, M]), shape=(L,), dtype=np.uint8)  # (x, y) + objects
        self.observation_space = self.position_space


        with open(config_path, "r") as stream:
            try:    
                config_dict = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        self.max_steps = config_dict.get('max_episode_steps', 200)  # Default to 200 if not specified
        self.step_num = 0
        self.locations = copy.deepcopy(self.env.locations)

    def step(self, action):
        self.step_num += 1
        o = self.env.step(action)
        obs_tuple = o[0]
        obs_position = self.obs_from_tuple(obs_tuple)
        obs = obs_position
        done = o[2]
        truncated = o[3]

        reward = 0  # reward done at monitor_reward level
        #reward = super().monitor_reward(done, truncated) # all the reward comes from the RM
        info = {}
        return obs, reward, done, truncated, info

    def reset(self):
        self.step_num = 0
        o, _ = self.env.reset()
        obs_position = self.obs_from_tuple(o)
        obs = obs_position
        return obs

class GridEnv_numerical_Original(GridEnv_Original):
    def obs_from_tuple(self,observation_tuple):
        obs = list(observation_tuple[:2])
        proposition_one_hot = [0]*len(self.propositions)
        prop = observation_tuple[2]
        prop_val = 1     # What the value gets set to at the index
        changed = False
        if isinstance(prop, int):
            prop_val = copy.deepcopy(prop)
            for key, value in self.locations.items():
                if value == self.env.agent_position:
                    prop = key
                    changed = True
            if changed == False:
                for key, value in self.env.locations.items():
                    if value == self.env.agent_position:
                        prop = key
                        changed = True
            if changed == False:
                print('change errror!!!!!!!!!!!!!!!!!!!!!!!!!!!')

        index_of_proposition = self.one_hot_propositions[prop]
        proposition_one_hot[index_of_proposition] = prop_val
        obs = obs + proposition_one_hot

        return tuple(obs)