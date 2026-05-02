from abc import abstractmethod


class Agent:
    @abstractmethod
    def reset_training(self):
        ...

    @abstractmethod
    def reset(self):
        ...

    @abstractmethod
    def get_action(self, obs):
        ...

    @abstractmethod
    def update(self, obs, action, next_obs, reward, terminated):
        ...

    @abstractmethod
    def decay_epsilon(self):
        ...

    @abstractmethod
    def terminated(self):
        ...
