import numpy as np
from stable_baselines3.common.buffers import DictReplayBuffer
from stable_baselines3.common.vec_env import VecNormalize


class SequentialDictReplayBuffer(DictReplayBuffer):
    def sample(self, batch_size: int, env: VecNormalize = None):
        """
        Override sample method to implement non-randomized sequential sampling.
        This method fetches the most recent `batch_size` transitions in sequential order.
        """
        current_size = self.size()  # Get the number of valid transitions in the buffer

        # Ensure we do not sample out of bounds
        start_index = max(0, self.pos - batch_size)

        # Generate indices for sequential sampling
        indices = np.arange(start_index, self.pos)

        # Fetch the samples for these indices using the parent class's _get_samples method
        return self._get_samples(indices, env)
