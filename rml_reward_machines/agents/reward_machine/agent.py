from collections import defaultdict

import numpy as np


class RewardMachineAgent:
    def __init__(
        self,
        machine,
        learning_rate,
        initial_epsilon,
        epsilon_decay,
        final_epsilon,
        discount_factor,
        action_space,
    ):
        self.machine = machine
        self.learning_rate = learning_rate
        self.initial_epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.discount_factor = discount_factor
        self.action_space = action_space
        self.reset_training()

    def reset_training(self):
        self.Q = [
            defaultdict(lambda: np.zeros(self.action_space.n)) for _ in self.machine.U
        ]
        self.epsilon = self.initial_epsilon

    def reset(self):
        self.u = self.machine.u_0

    def get_action(self, obs):
        # Remove propositions
        o = obs[:2]

        if np.random.random() < self.epsilon or np.all(self.Q[self.u][o] == 0):
            return self.action_space.sample()
        else:
            return np.argmax(self.Q[self.u][o])

    def update(self, obs, action, next_obs, reward, terminated):
        props = next_obs[3]
        # Remove propositions
        o = obs[:2]
        next_o = next_obs[:2]

        next_u, reward = self.machine.transition(props, self.u)

        if next_u in self.machine.F or terminated:
            self.Q[self.u][o][action] += self.learning_rate * (
                reward - self.Q[self.u][o][action]
            )
        else:
            self.Q[self.u][o][action] += self.learning_rate * (
                reward
                + self.discount_factor * np.max(self.Q[next_u][next_o])
                - self.Q[self.u][o][action]
            )
        self.u = next_u

    def get_greedy_action(self, obs):
        # Remove propositions
        o = obs[:2]
        return np.argmax(self.Q[self.u][o])

    def step(self, next_obs):
        props = next_obs[3]
        next_u, _ = self.machine.transition(props, self.u)
        self.u = next_u

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    def terminated(self):
        return self.u in self.machine.F


class RewardMachineCRMAgent(RewardMachineAgent):
    def update(self, obs, action, next_obs, reward, terminated):
        props = next_obs[3]
        o = obs[:2]
        next_o = next_obs[:2]

        for u_i in self.machine.U:
            if (props, u_i) not in self.machine.delta_u:
                continue

            u_j, reward = self.machine.transition(props, u_i)

            if u_j in self.machine.F or terminated:
                self.Q[u_i][o][action] += self.learning_rate * (
                    reward - self.Q[u_i][o][action]
                )
            else:
                self.Q[u_i][o][action] += self.learning_rate * (
                    reward
                    + self.discount_factor * np.max(self.Q[u_j][next_o])
                    - self.Q[u_i][o][action]
                )
        next_u, _ = self.machine.transition(props, self.u)
        self.u = next_u


class RewardMachineRSAgent(RewardMachineAgent):
    def __init__(
        self,
        machine,
        learning_rate,
        initial_epsilon,
        epsilon_decay,
        final_epsilon,
        discount_factor,
        action_space,
    ):
        super().__init__(
            machine,
            learning_rate,
            initial_epsilon,
            epsilon_decay,
            final_epsilon,
            discount_factor,
            action_space,
        )

        states = self.machine.U + self.machine.F
        actions = set()

        for k in self.machine.delta_u:
            actions.add(k[0])

        self.V = {s: 0 for s in states}
        delta = 1

        while delta > 1e-6:
            delta = 0

            for s in self.V:
                v = self.V[s]
                v_prime = 0

                if s in self.machine.F:
                    # Terminal states have value zero
                    continue

                for a in actions:
                    s_prime = self.machine.delta_u[(a, s)]
                    r = self.machine.delta_r[(a, s)]
                    v_prime = max(v_prime, (r + 0.99 * self.V[s_prime]))

                self.V[s] = v_prime
                delta = max(delta, abs(v - v_prime))

    def update(self, obs, action, next_obs, reward, terminated):
        props = next_obs[3]
        u_next, reward = self.machine.transition(props, self.u)

        shaped_reward = (
            reward + self.discount_factor * (self.V[u_next]) - (self.V[self.u])
        )

        # Remove propositions
        o = obs[:2]
        next_o = next_obs[:2]

        if u_next in self.machine.F or terminated:
            self.Q[self.u][o][action] += self.learning_rate * (
                shaped_reward - self.Q[self.u][o][action]
            )
        else:
            self.Q[self.u][o][action] += self.learning_rate * (
                shaped_reward
                + self.discount_factor * np.max(self.Q[u_next][next_o])
                - self.Q[self.u][o][action]
            )

        self.u = u_next


class RewardMachineRSCRMAgent(RewardMachineRSAgent):
    def update(self, obs, action, next_obs, reward, terminated):
        props = next_obs[3]
        o = obs[:2]
        next_o = next_obs[:2]

        for u_i in self.machine.U:
            if (props, u_i) not in self.machine.delta_u:
                continue

            u_j, reward = self.machine.transition(props, u_i)

            shaped_reward = (
                reward + self.discount_factor * (self.V[u_j]) - (self.V[u_i])
            )

            if u_j in self.machine.F or terminated:
                self.Q[u_i][o][action] += self.learning_rate * (
                    shaped_reward - self.Q[u_i][o][action]
                )
            else:
                self.Q[u_i][o][action] += self.learning_rate * (
                    shaped_reward
                    + self.discount_factor * np.max(self.Q[u_j][next_o])
                    - self.Q[u_i][o][action]
                )
        next_u, _ = self.machine.transition(props, self.u)
        self.u = next_u
