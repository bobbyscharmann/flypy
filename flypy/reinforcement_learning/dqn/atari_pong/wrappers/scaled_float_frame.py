
import gym
import numpy as np


class ScaledFloatFrame(gym.ObservationWrapper):
    """Converts data from bytes to floats and scales in range [0.0, 1.0]"""

    def observation(self, observation):
        return np.array(observation).astype(np.float32) / 255.0
