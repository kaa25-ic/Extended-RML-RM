import gymnasium as gym
from gymnasium import spaces
import numpy as np
from envs.office_world import OfficeWorld
import yaml

class GridEnv_Old(gym.Env):
    def __init__(self, env, config_path, render_mode = None):
        #super().__init__(config_path = './examples/office.yaml')
        self.env = env
        N,M      = self.env.n_rows, self.env.n_cols
        self.propositions = self.env.propositions
        self.propositions.append("_")
        self.one_hot_propositions = self.generate_one_hot_propisition(self.propositions)

        self.action_space = spaces.Discrete(4) # up, right, down, left
        self.observation_space = spaces.Box(low=0, high=max([N,M]), shape=(2,), dtype=np.uint8)
        self.observation_dict  = spaces.Dict({'features': self.observation_space})
        with open(config_path, "r") as stream:
            try:    
                config_dict = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        self.max_steps = config_dict.get('max_episode_steps', 200)  # Default to 200 if not specified
        self.step_num = 0

    def generate_one_hot_propisition(self,propositions):
        proposition_dict = {}
        i = 0
        for prop in propositions:
            proposition_dict[prop] = i
            i += 1
        return proposition_dict

    def obs_from_tuple(self,observation_tuple):
        obs = list(observation_tuple[:2])
        proposition_one_hot = [0]*len(self.propositions)
        index_of_proposition = self.one_hot_propositions[observation_tuple[2]]
        proposition_one_hot[index_of_proposition] = 1
        obs = obs + proposition_one_hot
        return tuple(obs)


    def get_events(self):
        return self.env.get_true_propositions()

    def step(self, action):
        self.step_num += 1
        o = self.env.step(action)
        obs_tuple = o[0]

        obs = self.obs_from_tuple(obs_tuple)

        if self.step_num < self.max_steps:
            truncated = False
        else:
            truncated = True
        done = truncated

        reward = 0  # reward done at monitor_reward level
        #reward = super().monitor_reward(done, truncated) # all the reward comes from the RM
        if obs_tuple[2] in ['A','B','C','D','E']:
            info = obs_tuple[2]
        else:
            info = {}
        return obs, reward, done, truncated, info

    def reset(self):
        o, _ = self.env.reset()
        obs = self.obs_from_tuple(o)
        self.step_num = 0
        return obs

    def show(self):
        self.env.show()

    def get_model(self):
        return self.env.get_model()
    
    def render(self, mode='human'):
        if mode == 'human':
            # commands
            str_to_action = {"w":0,"d":1,"s":2,"a":3}

            # play the game!
            done = True
            while True:
                if done:
                    print("New episode --------------------------------")
                    obs = self.reset()
                    print("Current task:", self.rm_files[self.current_rm_id])
                    self.env.show()
                    print("Features:", obs)
                    print("RM state:", self.current_u_id)
                    print("Events:", self.env.get_events())

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
                    print("RM state:", self.current_u_id)
                    print("Events:", self.env.get_events())
                else:
                    print("Forbidden action")
        else:
            raise NotImplementedError


class Letter_RMLEnv_Old(GridEnv_Old):
    metadata = {'render_modes': [22]}
    def __init__(self, render_mode = None):
        env = OfficeWorld()
        config_path = './examples/office.yaml'
        super().__init__(env, config_path, render_mode)

