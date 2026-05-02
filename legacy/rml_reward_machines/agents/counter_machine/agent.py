from collections import defaultdict
import numpy as np
import random
import torch
import copy

class CounterMachineAgent:
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
        self.Q = defaultdict(lambda: np.zeros(self.action_space.n))
        self.epsilon = self.initial_epsilon

    def reset(self):
        self.u = self.machine.u_0
        self.counters = tuple(0 for _ in list(self.machine.delta_u.keys())[0][2])

    def get_action(self, obs):
        # Remove propositions
        o = obs[:2]
        machine_state = o + (self.u,) + (self.counters,)

        if np.random.random() < self.epsilon or np.all(self.Q[machine_state] == 0):
            return self.action_space.sample()
        else:
            return np.argmax(self.Q[machine_state])

    def update(self, obs, action, next_obs, reward, terminated):
        props = next_obs[3]
        # Remove propositions
        o = obs[:2]
        next_o = next_obs[:2]

        machine_state = o + (self.u,) + (self.counters,)

        next_u, next_counters, reward = self.machine.transition(
            props, self.u, self.counters
        )
        if next_u in self.machine.F:
            self.Q[machine_state][action] += self.learning_rate * (
                reward - self.Q[machine_state][action]
            )
        else:
            next_machine_state = next_o + (next_u,) + (next_counters,)

            self.Q[machine_state][action] += self.learning_rate * (
                reward
                + self.discount_factor * np.max(self.Q[next_machine_state])
                - self.Q[machine_state][action]
            )

        self.u = next_u
        self.counters = next_counters


    def get_greedy_action(self, obs):
        # Remove propositions
        o = obs[:2]
        machine_state = o + (self.u,) + (self.counters,)
        return np.argmax(self.Q[machine_state])

    def step(self, next_obs):
        props = next_obs[3]
        next_u, next_counters, _ = self.machine.transition(props, self.u, self.counters)
        self.u = next_u
        self.counters = next_counters

    def decay_epsilon(self):
        #self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)
        self.epsilon *= self.epsilon_decay

    def terminated(self):
        return self.u in self.machine.F
    
    def get_evaluation_action(self,obs):
        # Remove propositions
        o = obs[:2]
        machine_state = o + (self.u,) + (self.counters,)

        if np.all(self.Q[machine_state] == 0):
            return 0, True
        else:
            return np.argmax(self.Q[machine_state]), False

    def evaluation_update(self, next_obs):
        """
        Update the internal state and counters without learning (i.e., without updating Q-values).
        This function is used during deterministic evaluation to track the agent's state correctly.
        """
        # Extract propositions or features from the next observation
        props = next_obs[3]
        
        # Transition to the next state based on the current state and counters
        next_u, next_counters, _ = self.machine.transition(props, self.u, self.counters)
        
        # Update the agent's internal state and counters
        self.u = next_u
        self.counters = next_counters



class CounterMachineCRMAgent(CounterMachineAgent):
    def reset_training(self):
        self.Q = defaultdict(lambda: np.zeros(self.action_space.n))
        self.observed_counters = defaultdict(lambda: set())
        self.epsilon = self.initial_epsilon

    def update(self, obs, action, next_obs, reward, terminated):
        props = next_obs[3]
        o = obs[:2]
        next_o = next_obs[:2]

        # Store observed counter states
        self.observed_counters[self.u].add(self.counters)

        for u_i in self.machine.U:
            for c_j in self.observed_counters[u_i]:
                counterfactual_state = o + (u_i,) + (c_j,)
                u_k, c_k, r_k = self.machine.transition(props, u_i, c_j)
                if u_k in self.machine.F:
                    self.Q[counterfactual_state][action] += self.learning_rate * (
                        r_k - self.Q[counterfactual_state][action]
                    )
                else:
                    next_counterfactual_state = next_o + (u_k,) + (c_k,)

                    self.Q[counterfactual_state][action] += self.learning_rate * (
                        r_k
                        + self.discount_factor
                        * np.max(self.Q[next_counterfactual_state])
                        - self.Q[counterfactual_state][action]
                    )

        next_u, next_counters, _ = self.machine.transition(props, self.u, self.counters)
        self.u = next_u
        self.counters = next_counters

class CounterMachineCRMAgent_NewAction(CounterMachineCRMAgent):
    def get_action(self, obs, env,Actions):
        # Remove propositions
        o = obs[:2]
        machine_state = o + (self.u,) + (self.counters,)

        valid_actions_list = [a for a in Actions if (o[0], o[1], a) not in env.forbidden_transitions]

        valid_actions = []
        for i in range(4):
            if list(Actions)[i] in valid_actions_list:
                valid_actions.append(i)

        if np.random.random() < self.epsilon or np.all(self.Q[machine_state] == 0):
            return random.choice(valid_actions)  #self.action_space.sample()
        else:
            best_action = np.argmax(self.Q[machine_state])

            # If the best action is not valid, select the next best valid action
            if best_action not in valid_actions:
                valid_action_values = [(a, self.Q[machine_state][a]) for a in valid_actions]
                best_action = max(valid_action_values, key=lambda x: x[1])[0]
            
            return best_action
        
class CounterMachineCRMAgent_NewAction_ExtraReward(CounterMachineCRMAgent_NewAction):
    def update(self, obs, action, next_obs, reward, terminated):
        props = next_obs[3]
        o = obs[:2]
        next_o = next_obs[:2]

        # Current state and counters
        current_state = o + (self.u,) + (self.counters,)

        # Machine transition for actual current state
        next_u, next_counters, r = self.machine.transition(props, self.u, self.counters)

        # Next state and counters
        next_state = next_o + (next_u,) + (next_counters,)

        # Check if the next state is new
        first_time_next_state = next_state not in self.Q

        # Combine machine reward with environment reward
        total_reward = reward + r

        # Add small positive reward if it's the first time observing the next state
        if first_time_next_state:
            small_positive_reward = 0.1  # Adjust as needed
            total_reward += small_positive_reward

        # Update Q-value for actual current state
        if next_u in self.machine.F:
            # Terminal state: update Q-value with immediate total reward
            self.Q[current_state][action] += self.learning_rate * (
                total_reward - self.Q[current_state][action]
            )
        else:
            # Non-terminal: update Q-value with discounted future reward
            self.Q[current_state][action] += self.learning_rate * (
                total_reward
                + self.discount_factor * np.max(self.Q[next_state])
                - self.Q[current_state][action]
            )

        # Update observed counters
        self.observed_counters[self.u].add(self.counters)

        # Update agent's state and counters
        self.u = next_u
        self.counters = next_counters

        # Counterfactual Updates
        for u_i in self.machine.U:
            for c_j in self.observed_counters[u_i]:
                counterfactual_state = o + (u_i,) + (c_j,)
                u_k, c_k, r_k = self.machine.transition(props, u_i, c_j)
                next_counterfactual_state = next_o + (u_k,) + (c_k,)

                # Skip if counterfactual next state is the actual next state to avoid double counting the reward
                if next_counterfactual_state == next_state:
                    r_k_adjusted = r_k
                    if first_time_next_state:
                        # Add small positive reward only once
                        r_k_adjusted += small_positive_reward
                else:
                    r_k_adjusted = r_k

                if u_k in self.machine.F:
                    # Terminal state: update Q-value with immediate reward
                    self.Q[counterfactual_state][action] += self.learning_rate * (
                        r_k_adjusted - self.Q[counterfactual_state][action]
                    )
                else:
                    self.Q[counterfactual_state][action] += self.learning_rate * (
                        r_k_adjusted
                        + self.discount_factor * np.max(self.Q[next_counterfactual_state])
                        - self.Q[counterfactual_state][action]
                    )

class DeepQ_CounterMachineAgent:
    def __init__(
        self,
        machine,
        model,
        optimizer,
        buffer,
        learning_rate,
        initial_epsilon,
        epsilon_decay,
        final_epsilon,
        discount_factor,
        action_space
    ):
        self.machine = machine
        self.learning_rate = learning_rate
        self.initial_epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.discount_factor = discount_factor
        self.action_space = action_space
        self.model = model
        self.target_model = copy.deepcopy(self.model)
        self.optimizer = optimizer
        self.buffer = buffer
        self.reset_training()

    def update_target_network(self):
        """Update the target network by copying the weights from the main network."""
        self.target_model.load_state_dict(self.model.state_dict())

    def reset_training(self):
        self.model.reset_weights()    # Model needs a reset weights function to be compatible
        self.target_model.load_state_dict(self.model.state_dict())  # Sync target network initially
        self.epsilon = self.initial_epsilon
        if hasattr(self.buffer, 'clear'):
            self.buffer.clear()
        else:
            raise AttributeError("The buffer does not have a 'clear' method.")

    def reset(self):
        self.u = self.machine.u_0
        self.counters = tuple(0 for _ in list(self.machine.delta_u.keys())[0][2])
        
    def get_action(self, obs):
        o = list(obs[0]) + [obs[1]]

        machine_state = o + [self.u] + list(self.counters)  # List format

        # Combine the observation and machine state into a single input tensor

        full_state = torch.tensor(machine_state, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

        # Perform epsilon-greedy action selection
        if np.random.random() < self.epsilon:
            # Exploration: select a random action
            return self.action_space.sample()
        else:
            # Exploitation: use the model to predict action values or probabilities
            with torch.no_grad():
                action_values = self.model(full_state)  # Forward pass through the model
            
            # Select the action with the highest predicted value
            return torch.argmax(action_values).item()


    def update(self, batch_size):
        """
        Sample experiences from the buffer and update the Q-network.

        """
        if self.buffer.size() < batch_size:
            return  # Ensure there are enough experiences in the buffer

        # Sample a minibatch of experiences from the buffer
        minibatch = self.buffer.sample(batch_size)

        # Initialize lists to store components of the minibatch
        states, actions, rewards, next_states, dones = [], [], [], [], []

        # Unpack the minibatch
        for experience in minibatch:
            state, action, reward, next_state, done = experience
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
        
        # Convert lists to tensors
        state_batch = torch.tensor(states, dtype=torch.float32)
        action_batch = torch.tensor(actions, dtype=torch.long).unsqueeze(1)  # Unsqueeze for indexing
        reward_batch = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_state_batch = torch.tensor(next_states, dtype=torch.float32)
        done_batch = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

        # Get the Q-values for the current states from the main network
        current_q_values = self.model(state_batch).gather(1, action_batch)

        # Get the Q-values for the next states from the target network
        with torch.no_grad():
            next_q_values = self.target_model(next_state_batch).max(1)[0].unsqueeze(1)

        # Compute the target Q-values: r + γ * max(Q(s', a')) * (1 - done)
        target_q_values = reward_batch + (self.discount_factor * next_q_values * (1 - done_batch))

        # Compute the loss (MSE loss between current Q-values and target Q-values)
        loss = torch.nn.functional.mse_loss(current_q_values, target_q_values)

        # Perform backpropagation and update the model
        self.optimizer.zero_grad()  # Clear previous gradients
        loss.backward()  # Backpropagate the loss
        self.optimizer.step()  # Update model parameters


    def get_greedy_action(self, obs):
        # Remove propositions
        o = list(obs[0]) + [obs[1]]
        machine_state = o + [self.u] + list(self.counters)  # List format
        full_state = torch.tensor(machine_state, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        
        with torch.no_grad():
            action_values = self.model(full_state)  # Forward pass through the model
        
        # Select the action with the highest predicted value
        return torch.argmax(action_values).item()

    def step(self, obs, action, next_obs, done):
        """
        Updates the machine state and stores the experience in the replay buffer.

        """
        props = next_obs[3]
        
        next_u, next_counters, reward = self.machine.transition(props, self.u, self.counters)
        o = list(obs[0]) + [obs[1]]
        next_o = list(next_obs[0]) + [next_obs[1]]

        machine_state = o + [self.u] + list(self.counters)  # List format
        next_machine_state = next_o + [next_u] + list(next_counters)
        
        self.buffer.add(machine_state, action, reward, next_machine_state, done)
        
        # Update the machine's internal state and counters
        self.u = next_u
        self.counters = next_counters

        return reward
        


    def decay_epsilon(self):
        #self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)
        self.epsilon *= self.epsilon_decay

    def terminated(self):
        return self.u in self.machine.F

    def get_evaluation_action(self, obs):
        o = list(obs[0]) + [obs[1]]
        machine_state = o + [self.u] + list(self.counters)  # List format
        full_state = torch.tensor(machine_state, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            action_values = self.model(full_state)  # Forward pass through the model
        
        if torch.all(action_values == 0):
            return 0, True  # Return default action 0 if all values are 0, along with True
        else:
            return torch.argmax(action_values).item(), False

