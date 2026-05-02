from envs.letterenv import LetterEnv, Actions
import gymnasium as gym
import numpy as np
import pandas as pd
import random
from enum import Enum
from gymnasium.envs.registration import registry
from gymnasium.envs.registration import load_env_creator
from gymnasium.envs.registration import register
from rml.rmlgym import RMLGym_Simple, RMLGym_Simple_No_Intermediate_Reward
from tqdm import tqdm
from utils.learning_functions import learning_episode_letter, learning_episode_letter_no_exploration
from utils.train_rml import rml_training_episode_rewards
import matplotlib.pyplot as plt
import pickle

import matplotlib.pyplot as plt
from envs.property_envs.letterenv_numerical import Actions

SEED = 0

N_EPISODES = 1000

agent_kwargs = {
    "initial_epsilon": 0.4,
    "final_epsilon": 0.01,
    "epsilon_decay": 0.99,
    "learning_rate": 0.5,
    "discount_factor": 0.9,
}
train_conv_kwargs = {
    "n_repeats": 1,
}


if __name__ == "__main__":    
    
    N = 1

    print(' ')
    print('Training started for simple encoding')
    print(' ')
    config_path = './examples/letter_env.yaml'

    register(
        id='letter-env',
        entry_point='envs.property_envs.letterenv_numerical_wrappers:RML_LetterEnv_numerical_4_Simple',
        max_episode_steps=200
    )

    """env = RMLGym_Simple(config_path)
    actions = [Actions.RIGHT.value, Actions.LEFT.value, Actions.UP.value, Actions.DOWN.value]
    training_class = rml_training_episode_rewards(learning_episode_letter, RMLGym_Simple, actions, config_path, n=N, epsilon=0.4)
    results, num_successes = training_class.get_test_statistics(N_EPISODES)
    print(num_successes)"""

    env = RMLGym_Simple_No_Intermediate_Reward(config_path)
    actions = [Actions.RIGHT.value, Actions.LEFT.value, Actions.UP.value, Actions.DOWN.value]
    training_class = rml_training_episode_rewards(learning_episode_letter_no_exploration, RMLGym_Simple_No_Intermediate_Reward, 
                                                  actions, config_path, n=N, epsilon=0.4)
    training_class.correct_reward = 100
    results_2, num_successes_2 = training_class.get_test_statistics(N_EPISODES)
    print(num_successes_2)

    #with open('results/letterenv_numerical_rml_rm_rewards.pkl', 'wb') as f:
       # pickle.dump(results, f)

    with open('results/letterenv_numerical_rml_rm_rewards_no_intermediate.pkl', 'wb') as f:
       pickle.dump(results_2, f)
