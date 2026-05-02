from envs.letterenv import LetterEnv, Actions
import gymnasium as gym
import numpy as np
import pandas as pd
import random
from enum import Enum
from gymnasium.envs.registration import registry
from gymnasium.envs.registration import load_env_creator
from gymnasium.envs.registration import register
from rml.rmlgym import RMLGym_Simple
from tqdm import tqdm
from utils.learning_functions import learning_episode_letter, evaluation_episode_encoding
from utils.train_rml import rml_training_simple
import matplotlib.pyplot as plt
import pickle


config_path = './examples/letter_env.yaml'

register(
    id='letter-env',
    entry_point='envs.letterenv_wrappers:RML_LetterEnv_Conditional_Simple',
    max_episode_steps=200
)

env = RMLGym_Simple(config_path)

actions = [Actions.RIGHT.value, Actions.LEFT.value, Actions.UP.value, Actions.DOWN.value]


training_class = rml_training_simple(learning_episode_letter, RMLGym_Simple, actions, config_path, n=5)


results = training_class.get_test_statistics()

print(results)


"""
with open('results/results_LetterEnv_RML_Conditional_simple_encoding.pkl', 'wb') as f:
    pickle.dump(results, f)
"""
