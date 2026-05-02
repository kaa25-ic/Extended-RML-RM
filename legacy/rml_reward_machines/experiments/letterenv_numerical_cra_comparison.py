"""

Code for the Numerical comparison with CRA. Needs to be run with letter_env_spec_numerical.pl monitor.

"""


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
from utils.train_rml import rml_training_numerical_demo
import matplotlib.pyplot as plt
import pickle

import os
import sys
from collections import defaultdict
import matplotlib.pyplot as plt
from envs.property_envs.letterenv_numerical import Actions

from agents.counter_machine.agent import (
    CounterMachineCRMAgent_NewAction_ExtraReward,
    CounterMachineAgent
)

from agents.counter_machine.context_free.config_extra_reward_numerical import (
    ContextFreeCounterMachine_Rewards
)

from environments.context_free_numerical import (
    create_context_free_env_labelled,
    labelled_action_space,
    mdp_action_space,
)
from utils.train import train_till_conv_repeat_letter, train_till_conv_repeat_letter_no_counterfactual

SEED = 0

N_EPISODES = [
    25000,
    50000,
    100000,
    1000000,
    1000000,
]


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



def create_counter_crm_agent():
    return CounterMachineCRMAgent_NewAction_ExtraReward(
        machine=ContextFreeCounterMachine_Rewards(),
        action_space=labelled_action_space,
        **agent_kwargs,
    )

def create_counter_agent():
    return CounterMachineAgent(
        machine=ContextFreeCounterMachine_Rewards(),
        action_space=labelled_action_space,
        **agent_kwargs,
    )


def get_agent_env_pairs(n):
    return (
        (
            "CQL",
            create_counter_crm_agent(),
            create_context_free_env_labelled(n),
        )
    )

def get_agent_env_pairs_no_counterfactual(n):
    return (
        (
            "CRA",
            create_counter_agent(),
            create_context_free_env_labelled(n),
        )
    )
if __name__ == "__main__":
    # Counting reward automata below for n up to 3
    
    convergence_results = {}
    
    for N in [1,2,3]:
        results_list = []
        for i in range(20):
            print(' ')
            print(f"Training started for Counting Reward Automata for N = {N}")
            print(' ')
            pairs = get_agent_env_pairs(N)
            name, agent, env = pairs[0], pairs[1], pairs[2]
            agent.epsilon_decay = agent_kwargs["epsilon_decay"]
            train_conv_kwargs["max_samples"] = 100 * N_EPISODES[-1]
            conv_results = train_till_conv_repeat_letter(Actions, agent, env, **train_conv_kwargs)
            results_list.append(conv_results)
        convergence_results[N] = results_list
    print(convergence_results)
    
    # Counting Reward Automata without counterfactual learning below
    convergence_results_no_counterfactual = {}
    
    for N in [1,2,3]:
        results_list = []
        for i in range(20):
            print(' ')
            print(f"Training started for Counting Reward Automata for N = {N}")
            print(' ')
            pairs = get_agent_env_pairs_no_counterfactual(N)
            name, agent, env = pairs[0], pairs[1], pairs[2]
            agent.epsilon_decay = agent_kwargs["epsilon_decay"]
            train_conv_kwargs["max_samples"] = 100 * N_EPISODES[-1]
            conv_results = train_till_conv_repeat_letter_no_counterfactual(Actions, agent, env, **train_conv_kwargs)
            results_list.append(conv_results)
        convergence_results_no_counterfactual[N] = results_list
    print(convergence_results_no_counterfactual)
    
    
    # Simple encoding below for n = [1,2,3,4,5]
    print(' ')
    print('Training started for simple encoding')
    print(' ')
    config_path = './examples/letter_env.yaml'

    register(
        id='letter-env',
        entry_point='envs.property_envs.letterenv_numerical_wrappers:RML_LetterEnv_numerical_4_Simple',
        max_episode_steps=200
    )

    env = RMLGym_Simple(config_path)
    actions = [Actions.RIGHT.value, Actions.LEFT.value, Actions.UP.value, Actions.DOWN.value]
    training_class = rml_training_numerical_demo(learning_episode_letter, RMLGym_Simple, actions, config_path, n=10, epsilon=0.4)
    results = training_class.get_test_statistics()
    print(results)
    
    """
    with open('results/results_LetterEnv_numerical_RML_simple_encoding_up_to_10.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    with open('results/results_LetterEnv_numerical_CRA_200_run.pkl', 'wb') as f:
        pickle.dump(convergence_results, f)
    
    with open('results/results_LetterEnv_numerical_CRA_no_counterfactual.pkl', 'wb') as f:
        pickle.dump(convergence_results_no_counterfactual, f)"""