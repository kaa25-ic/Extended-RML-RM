from collections import defaultdict

import numpy as np


class QLearningAgent:
    def __init__(
        self,
        learning_rate,
        initial_epsilon,
        epsilon_decay,
        final_epsilon,
        discount_factor,
        action_space,
    ):
        self.learning_rate = learning_rate
        self.initial_epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.discount_factor = discount_factor
        self.action_space = action_space
        self.reset_training()

    def reset_training(self):
        self.epsilon = self.initial_epsilon
        self.Q = defaultdict(lambda: np.zeros(self.action_space.n))
        self.epsilon = self.initial_epsilon

    def get_action(self, obs):
        if np.random.random() < self.epsilon or np.all(self.Q[obs] == 0):
            return self.action_space.sample()
        else:
            return np.argmax(self.Q[obs])

    def update(self, obs, action, next_obs, reward, terminated):
        if terminated:
            self.Q[obs][action] += self.learning_rate * (reward - self.Q[obs][action])
        else:
            self.Q[obs][action] += self.learning_rate * (
                reward
                + self.discount_factor * np.max(self.Q[next_obs])
                - self.Q[obs][action]
            )

    def get_greedy_action(self, obs):
        return np.argmax(self.Q[obs])

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    def reset(self):
        return

    def step(self, next_obs):
        return

    def terminated(self):
        return False
