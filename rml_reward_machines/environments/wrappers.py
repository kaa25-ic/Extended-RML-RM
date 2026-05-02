from gymnasium import ObservationWrapper


class MarkovWrapper(ObservationWrapper):
    """Wrapper used to create Markov states from POMDP observations.

    This wrapper class stores additional trajectory information. This information
    is appended to the POMDP observation produced by the environment to create a
    Markov state. The Markov state is returned by the `observation` method.
    Specifically, the history of propositions observed by the agent is
    appended to the POMDP observation.
    """

    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        obs = self.env.reset()
        self.observed_propositions = ""
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._last_obs = obs

        if obs[2] != "_":
            self.observed_propositions += obs[2]
        return obs, reward, terminated, truncated, info

    def observation(self, observation):
        return observation + (self.observed_propositions,)


class LabellingFunctionWrapper(ObservationWrapper):
    """Wrapper used to append labelling function results to observations."""

    def __init__(self, env):
        super().__init__(env)

    def _labelling_function(self, observation):
        prop = observation[2]

        if prop == "_":
            return ()
        else:
            return (prop,)

    def reset(self, **kwargs):
        obs = self.env.reset()
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, info

    def observation(self, observation):
        return observation + (self._labelling_function(observation),)


class IdentityWrapper(ObservationWrapper):
    """Wrapper used for compatibility."""

    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        obs = self.env.reset()
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, info

    def observation(self, observation):
        return observation
