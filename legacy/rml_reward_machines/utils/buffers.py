import numpy as np
import torch
import gym
from typing import Optional, Dict, Union
from stable_baselines3.common.buffers import ReplayBuffer, ReplayBufferSamples
from stable_baselines3.common.vec_env import VecNormalize


class SequentialDictReplayBuffer(ReplayBuffer):
    """
    A replay buffer for Dict observation spaces that returns samples in sequential
    order (chronological) rather than random order.

    Inherits from stable_baselines3's ReplayBuffer so that we satisfy the methods
    that OffPolicyAlgorithm expects (_init_buffer, _get_samples, sample, etc.).
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: gym.spaces.Dict,
        action_space: gym.spaces.Space,
        device: Union[torch.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        **kwargs
    ):
        """
        :param buffer_size: Maximum number of transitions in the buffer
        :param observation_space: Dict space, e.g. {'position': Box(...), 'monitor': Box(...)}
        :param action_space: Action space (Discrete or Box)
        :param device: PyTorch device or "auto"
        :param n_envs: Number of parallel environments
        :param optimize_memory_usage: Whether to reduce memory usage
        :param handle_timeout_termination: Whether to treat timeouts as terminals
        :param kwargs: Extra arguments passed to ReplayBuffer
        """
        # We call super().__init__, but we'll override _init_buffer below.
        """super().__init__(
            buffer_size=buffer_size,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            n_envs=n_envs,
            optimize_memory_usage=optimize_memory_usage,
            handle_timeout_termination=handle_timeout_termination,
            **kwargs
        )"""

        # We'll use this rolling index to pick sequential slices in self.sample().
        self.current_index = 0

    def _init_buffer(self) -> None:
        """
        Create arrays to store observations, actions, rewards, etc. for a Dict obs space.
        We replicate what SB3's `DictReplayBuffer` does, but with sequential sampling.
        """
        # We assume `self.observation_space` is a gym.spaces.Dict
        if not isinstance(self.observation_space, gym.spaces.Dict):
            raise ValueError(
                "SequentialDictReplayBuffer requires a Dict observation space!"
            )

        # Extract each sub-space
        self.observation_keys = list(self.observation_space.spaces.keys())

        # For each sub-key in the dict, allocate separate arrays for obs / next_obs
        self.observations = {}
        self.next_observations = {}
        for key, subspace in self.observation_space.spaces.items():
            # Shape and dtype
            subspace_shape = subspace.shape
            subspace_dtype = subspace.dtype

            # observations[key]: shape = (buffer_size, n_envs, *subspace_shape)
            self.observations[key] = np.zeros(
                (self.buffer_size, self.n_envs, *subspace_shape), dtype=subspace_dtype
            )
            self.next_observations[key] = np.zeros(
                (self.buffer_size, self.n_envs, *subspace_shape), dtype=subspace_dtype
            )

        # Actions (super() already handles discrete vs continuous, but let's re-init)
        if self.action_space.is_discrete:
            self.actions = np.zeros((self.buffer_size, self.n_envs), dtype=np.int32)
        else:
            # For continuous actions
            self.actions = np.zeros(
                (self.buffer_size, self.n_envs, *self.action_dim), dtype=np.float32
            )

        # Rewards and dones
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        if self.handle_timeout_termination:
            self.timeouts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

    def _get_samples(
        self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None
    ) -> ReplayBufferSamples:
        """
        Given a set of indices, return a ReplayBufferSamples dataclass
        with Torch tensors on the correct device. This is how SB3 obtains
        the actual data from the buffer.

        We'll slice each sub-key of our Dict observations accordingly.
        """
        # Slicing observations
        obs_dict = {}
        next_obs_dict = {}
        for key in self.observation_keys:
            obs_dict[key] = self.observations[key][batch_inds, 0, ...]
            next_obs_dict[key] = self.next_observations[key][batch_inds, 0, ...]

        # If we have n_envs>1, the above might be [batch_size, n_envs,...]
        # We take only env=0 or flatten across envs, depending on your usage.
        # For simplicity, let's flatten if n_envs>1:
        if self.n_envs > 1:
            for key in self.observation_keys:
                # shape = (len(batch_inds), n_envs, *obs_shape)
                # Flatten to (len(batch_inds)*n_envs, *obs_shape)
                obs_dict[key] = obs_dict[key].reshape((-1,) + obs_dict[key].shape[2:])
                next_obs_dict[key] = next_obs_dict[key].reshape(
                    (-1,) + next_obs_dict[key].shape[2:]
                )

        actions = self.actions[batch_inds, ...]
        rewards = self.rewards[batch_inds, ...]
        dones = self.dones[batch_inds, ...]

        if self.n_envs > 1:
            actions = actions.reshape(-1, *actions.shape[2:])
            rewards = rewards.reshape(-1)
            dones = dones.reshape(-1)

        # Convert to torch
        obs_torch = {
            key: torch.as_tensor(val, dtype=torch.float32, device=self.device)
            for (key, val) in obs_dict.items()
        }
        next_obs_torch = {
            key: torch.as_tensor(val, dtype=torch.float32, device=self.device)
            for (key, val) in next_obs_dict.items()
        }

        # Actions
        if self.action_space.is_discrete:
            actions_torch = torch.as_tensor(actions, device=self.device, dtype=torch.long)
        else:
            actions_torch = torch.as_tensor(actions, device=self.device, dtype=torch.float32)

        rewards_torch = torch.as_tensor(rewards, device=self.device, dtype=torch.float32)
        dones_torch = torch.as_tensor(dones, device=self.device, dtype=torch.float32)

        return ReplayBufferSamples(
            observations=obs_torch,
            actions=actions_torch,
            next_observations=next_obs_torch,
            dones=dones_torch,
            rewards=rewards_torch,
        )

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        """
        Return a batch of data in sequential order instead of random.
        We maintain a rolling 'self.current_index'.
        """
        current_size = self.size()  # how many valid transitions are actually stored
        if batch_size > current_size:
            raise ValueError(
                f"Not enough samples in the buffer! Requested {batch_size}, but only have {current_size}."
            )

        # If we don't have enough space from current_index -> end, wrap to 0
        if self.current_index + batch_size > current_size:
            self.current_index = 0

        # Indices is [current_index, ..., current_index + batch_size - 1]
        indices = np.arange(self.current_index, self.current_index + batch_size)
        self.current_index += batch_size

        # Use our custom _get_samples() to build the final ReplayBufferSamples
        return self._get_samples(indices, env=env)



"""
class ReplayBuffer:
    def __init__(self, max_size):
        """
        #Initialize the replay buffer.
        
        #:param max_size: Maximum size of the buffer (how many experiences it can store).
"""      

        self.buffer = deque(maxlen=max_size)  # A deque that discards the oldest experiences when full
    
    def add(self, state, action, reward, next_state, done):
        """
        #Add a new experience to the buffer.
        
        #:param state: The current state.
        #:param action: The action taken in the current state.
        #:param reward: The reward received after taking the action.
        #:param next_state: The next state observed after taking the action.
        #:param done: A boolean indicating if the episode is done (i.e., terminal state).
"""
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        """
        #Sample a batch of experiences from the buffer.
        
        #:param batch_size: The number of experiences to sample.
        #:return: A list of experiences, where each experience is a tuple (state, action, reward, next_state, done).
"""
        return random.sample(self.buffer, batch_size)
    
    def size(self):
        """
        #Return the current size of the buffer.
"""
        return len(self.buffer)

    def clear(self):
        """#Clear the buffer."""
        #self.buffer.clear()




