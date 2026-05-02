import gymnasium as gym
import numpy as np
import pandas as pd
import random
from enum import Enum
from gymnasium.envs.registration import registry
from gymnasium.envs.registration import load_env_creator
from gymnasium.envs.registration import register
from rml.rmlgym import RMLGym
from tqdm import tqdm
from utils.learning_functions import learning_episode_office, learning_episode_letter, evaluation_episode_encoding
import matplotlib.pyplot as plt
import pickle
from utils.encoding_functions import generate_events_and_index, create_encoding
import copy
import itertools



class rml_training(): 
    def __init__(self, learn_func, rmlgym, states_for_encoding, actions, config_path, n, epsilon=0.35, alpha=0.5, gamma=0.9, correct_reward=110):
        
        self.unique_events, self.event_index = generate_events_and_index(states_for_encoding)
        self.initial_encoding = create_encoding(states_for_encoding[0],self.event_index)

        self.results_df = pd.DataFrame(columns=['n value', 'episodes', 'steps', 'iteration'])
        self.config_path = config_path
        self.n = n
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.correct_reward = correct_reward
        self.rmlgym = rmlgym
        self.actions = actions
        self.learning_function = learn_func

    def initialise_parameters(self):
        epsilon = copy.deepcopy(self.epsilon)
        alpha = copy.deepcopy(self.alpha)
        gamma = copy.deepcopy(self.gamma)
        correct_reward = copy.deepcopy(self.correct_reward)
        return epsilon, alpha, gamma, correct_reward

    def test_loop(self,n):
        epsilon, alpha, gamma, correct_reward = self.initialise_parameters()
        env = self.rmlgym(self.event_index, self.initial_encoding, self.config_path)
        q_table = {}
        succesful_policy = False
        num_episodes = 0
        total_steps = 0
        results_df = pd.DataFrame(columns=['n value', 'episodes', 'steps'])
        rewards = []

        while succesful_policy == False:
            num_episodes += 1
            rewards, succesful_policy, q_table, state, epsilon, total_steps = self.learning_function(rewards, env, q_table, self.actions, alpha, gamma, epsilon, total_steps,n)

            if succesful_policy:
                new_row = pd.DataFrame([{'n value': n, 'episodes': num_episodes, 'steps': total_steps}])
                results_df = pd.concat([results_df, new_row])
                print('(n val, steps) - ', (n, total_steps))
        return results_df

    def get_test_statistics(self):
        if self.results_df.empty:
            iteration = 0
        else:
            iteration = self.results_df['iteration'].iloc[-1]
        for i in tqdm(range(20)):
            j = 0     
            iteration += 1
            while j < self.n:
                j += 1
                iteration_results_df = self.test_loop(j)
                iteration_results_df['iteration'] = iteration
                self.results_df = pd.concat([self.results_df, iteration_results_df])

        return self.results_df
    
class rml_training_simple(rml_training):
    def __init__(self, learn_func, rmlgym, actions, config_path, n, epsilon=0.35, alpha=0.5, gamma=0.9, correct_reward=110):

        self.results_df = pd.DataFrame(columns=['n value', 'episodes', 'steps', 'iteration'])
        self.config_path = config_path
        self.n = n
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.correct_reward = correct_reward
        self.rmlgym = rmlgym
        self.actions = actions
        self.learning_function = learn_func

    def test_loop(self,n):
        epsilon, alpha, gamma, correct_reward = self.initialise_parameters()
        env = self.rmlgym(self.config_path)
        q_table = {}
        succesful_policy = False
        num_episodes = 0
        total_steps = 0
        results_df = pd.DataFrame(columns=['n value', 'episodes', 'steps'])
        rewards = []

        while succesful_policy == False:
            num_episodes += 1
            rewards, succesful_policy, q_table, state, epsilon, total_steps = self.learning_function(rewards, env, q_table, self.actions, alpha, gamma, epsilon, total_steps,n)

            if succesful_policy:
                new_row = pd.DataFrame([{'n value': n, 'episodes': num_episodes, 'steps': total_steps}])
                results_df = pd.concat([results_df, new_row])
                print('(n val, steps) - ', (n, total_steps))
        return results_df


class rml_training_numerical_demo(rml_training):
    def __init__(self, learn_func, rmlgym, actions, config_path, n, epsilon=0.35, alpha=0.5, gamma=0.9, correct_reward=110):

        self.results_df = pd.DataFrame(columns=['n value', 'episodes', 'steps', 'iteration'])
        self.config_path = config_path
        self.n = n
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.correct_reward = correct_reward
        self.rmlgym = rmlgym
        self.actions = actions
        self.learning_function = learn_func

    def test_loop(self,n):
        epsilon, alpha, gamma, correct_reward = self.initialise_parameters()
        env = self.rmlgym(self.config_path)
        q_table = {}
        succesful_policy = False
        num_episodes = 0
        total_steps = 0
        results_df = pd.DataFrame(columns=['n value', 'episodes', 'steps'])
        rewards = []

        while succesful_policy == False:
            num_episodes += 1
            rewards, succesful_policy, q_table, state, epsilon, total_steps = self.learning_function(rewards, env, q_table, self.actions, alpha, gamma, epsilon, total_steps,n)

            if succesful_policy:
                new_row = pd.DataFrame([{'n value': n, 'episodes': num_episodes, 'steps': total_steps}])
                results_df = pd.concat([results_df, new_row])
                print('(n val, steps) - ', (n, total_steps))
        return results_df

    def get_test_statistics(self):
        if self.results_df.empty:
            iteration = 0
        else:
            iteration = self.results_df['iteration'].iloc[-1]
        
        while iteration < 20:
            j = 0
            iteration += 1
            while j < self.n:
                j += 1
                iteration_results_df = self.test_loop(j)
                iteration_results_df['iteration'] = iteration
                self.results_df = pd.concat([self.results_df, iteration_results_df])

        return self.results_df
    

class rml_training_simple_no_loop(rml_training_simple):
    def get_test_statistics(self):
        iteration = 1
        for i in tqdm(range(20)):
            iteration_results_df = self.test_loop(self.n)
            iteration_results_df['iteration'] = iteration
            self.results_df = pd.concat([self.results_df, iteration_results_df])

        return self.results_df
    
class rml_training_original_hyper(rml_training): 
    def __init__(self, learn_func, rmlgym, actions, config_path, n=1, episodes=1000,
                 epsilon=[0.3,0.5,0.75,0.9], 
                 alpha=[0.01,0.1,0.3] ,
                 gamma=[0.9,0.95,0.99],
                 decay = [0.99,0.995,0.999], 
                 correct_reward=100):

        self.results_df = pd.DataFrame(columns=['epsilon', 'alpha', 'gamma', 'decay', 'total successes'])
        self.config_path = config_path
        self.n = n
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.correct_reward = correct_reward
        self.decay = decay
        self.rmlgym = rmlgym
        self.actions = actions
        self.learning_function = learn_func
        self.episodes = episodes
    
    def test_loop(self,epsilon, alpha, gamma, decay):
        env = self.rmlgym(self.config_path)
        q_table = {}
        total_steps = 0
        results_df = pd.DataFrame(columns=['epsilon', 'alpha', 'gamma', 'decay', 'total successes'])
        rewards = []
        epsilon_start = copy.deepcopy(epsilon)

        for num_episodes in tqdm(range(self.episodes), desc="Episodes", leave=False):
            rewards, q_table, epsilon = self.learning_function(rewards, env, q_table, self.actions, alpha, gamma, epsilon,self.n,decay)
        
        num_success = sum(1 for r in rewards if r == self.correct_reward)
        new_row = pd.DataFrame([{'epsilon' : epsilon_start, 'alpha' : alpha, 'gamma' : gamma, 
                                    'decay' : decay, 'total successes' : num_success}])
        results_df = pd.concat([results_df, new_row])
        return results_df
    
    def get_test_statistics(self):
        total_combos = len(self.epsilon) * len(self.alpha) * len(self.gamma) * len(self.decay)

        for eps, a, g, d in tqdm(itertools.product(self.epsilon, self.alpha, self.gamma, self.decay),
                         total=total_combos, desc="Hyperparameter combos"):
            iteration_results_df = self.test_loop(eps, a, g, d)
            self.results_df = pd.concat([self.results_df, iteration_results_df])
            print(iteration_results_df)

        return self.results_df
    
class rml_training_original_test(rml_training_original_hyper):
    def test_loop(self, epsilon, alpha, gamma, decay):
        env = self.rmlgym(self.config_path)
        q_table = {}
        rewards = []

        for num_episodes in tqdm(range(self.episodes), desc="Episodes", leave=False):
            rewards, q_table, epsilon = self.learning_function(rewards, env, q_table, self.actions, alpha, gamma, epsilon,self.n,decay)

        return rewards
    
    def get_test_statistics(self):
        results = self.test_loop(self.epsilon, self.alpha, self.gamma, self.decay)
        num_successess = results.count(self.correct_reward)

        return results, num_successess

    

class rml_training_episode_rewards(rml_training_simple):
    def test_loop(self,episodes):
        env = self.rmlgym(self.config_path)
        epsilon, alpha, gamma, correct_reward = self.initialise_parameters()
        q_table = {}
        rewards = []
        epsilon_start = copy.deepcopy(epsilon)
        total_steps = 0

        for num_episodes in tqdm(range(episodes), desc="Episodes", leave=False):
            rewards, succesful_policy, q_table, state, epsilon, total_steps = self.learning_function(rewards, env, q_table, self.actions, alpha, gamma, epsilon, total_steps,self.n)
        

        return rewards
    
    def get_test_statistics(self,episodes):

        results = self.test_loop(episodes)

        num_successess = results.count(self.correct_reward)

        return results, num_successess


    