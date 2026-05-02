import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.dqn.policies import DQNPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces
import numpy as np
from stable_baselines3 import DQN


class RandomObjects_MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(RandomObjects_MLP, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        
        # Create a list of fully connected layers
        layers = []
        
        # First input to first hidden layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        
        # Create hidden layers dynamically
        for i in range(1, len(hidden_sizes)):
            layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
        
        # Output layer
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        
        # Assign layers to a ModuleList so they are recognized by PyTorch
        self.layers = nn.ModuleList(layers)

        # Initialize weights
        self.reset_weights()

    def forward(self, x):
        for layer in self.layers[:-1]:  # Apply activation to all layers except the last one
            x = F.relu(layer(x))
        q_values = self.layers[-1](x)  # No activation for the output layer
        return q_values

    def reset_weights(self):
        for layer in self.layers:
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            nn.init.zeros_(layer.bias)

class RandomObjects_MLP_RML(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(RandomObjects_MLP_RML, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        
        # Create a list of fully connected layers
        layers = []
        
        # First input to first hidden layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        
        # Create hidden layers dynamically
        for i in range(1, len(hidden_sizes)):
            layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
        
        # Output layer
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        
        # Assign layers to a ModuleList so they are recognized by PyTorch
        self.layers = nn.ModuleList(layers)

        # Initialize weights
        self.reset_weights()

    def forward(self, x):
        position = x['position']  
        monitor = x['monitor'] 
        x = torch.cat([position, monitor], dim=-1).float()

        for layer in self.layers[:-1]:  # Apply activation to all layers except the last one
            x = F.relu(layer(x))
        q_values = self.layers[-1](x)  # No activation for the output layer
        return q_values

    def reset_weights(self):
        for layer in self.layers:
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            nn.init.zeros_(layer.bias)

    def set_training_mode(self, mode: bool):
        """
        Set the module to training mode (mode=True) or evaluation mode (mode=False).
        This is used by Stable Baselines3 to toggle between modes.
        """
        self.train(mode)


class CustomMultiInputPolicy(DQNPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, net_arch=None, activation_fn=nn.ReLU, device="auto"):

        super(CustomMultiInputPolicy, self).__init__(observation_space, action_space, lr_schedule, net_arch, activation_fn)
        
        input_dims = self.get_input_dims(observation_space)

        self.q_net = RandomObjects_MLP_RML(input_size=input_dims, hidden_sizes=[64,128,64], output_size=action_space.n)
        self.q_net_target = RandomObjects_MLP_RML(input_size=input_dims, hidden_sizes=[64,128,64], output_size=action_space.n)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr_schedule(1))

        self.to(self.device)

    def forward(self, obs: torch.Tensor):
        print(self.q_net(obs))
        return self.q_net(obs)

    def _predict(self, obs: dict, deterministic: bool = False):
        q_values = self.q_net(obs)

        if deterministic:
            # Select the action with the highest Q-value for deterministic predictions
            return torch.argmax(q_values, dim=1)
        else:
            # Sample actions based on Q-values (for exploration)
            return torch.multinomial(F.softmax(q_values, dim=1), num_samples=1)

    def get_input_dims(self, observation_space):
        """
        This method extracts the total input size from the Dict observation space.
        It concatenates the sizes of all the input components in the dictionary.
        """
        input_dims = 0
        for space in observation_space.spaces.values():
            if isinstance(space, spaces.Box):
                input_dims += np.prod(space.shape)
        return input_dims

    def _get_constructor_parameters(self):
        return dict(
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            lr_schedule=self.lr_schedule
        )

    def compute_loss(self, replay_data):
        """
        Override the loss function for DQN.
        """
        # Get current Q-values from the main network
        q_values = self.q_net(replay_data.observations).gather(1, replay_data.actions.long())

        # Get the target Q-values from the target network
        with torch.no_grad():
            target_q_values = self.q_net_target(replay_data.next_observations).max(1)[0].unsqueeze(1)

        # Compute the target for the loss function
        target = replay_data.rewards + self.gamma * target_q_values * (1 - replay_data.dones)

        # Custom loss function: MSE, Huber loss, etc.
        loss = F.mse_loss(q_values, target)

        return loss
    
