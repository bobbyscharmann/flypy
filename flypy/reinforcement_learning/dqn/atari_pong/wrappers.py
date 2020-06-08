"""DQN on Pong"""
import cv2
import gym
import gym.spaces
import numpy as np
import collections


class FireResetEnv(gym.Wrapper):
    """Wrapper used to press the FIRE button for Atari games that require it to start."""
    def __init__(self, env=None):
        super(FireResetEnv, self).__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) > 3

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        self.env.reset()

        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset()

        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset()

        return obs
