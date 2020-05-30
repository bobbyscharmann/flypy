"""This function is used to implement the cross entropy method for Cartpole"""

import gym
import numpy as np
import torch
from collections import namedtuple

# COnfiguration options
HIDDEN_SIZE = 128
BATCH_SIZE = 16
PERCENTILE = 70

# Define a neural network to map observations to actions
class CrossEntropy(torch.nn.Module):

    def __init__(self, obs_size, hidden_size, n_actions):
        super(CrossEntropy, self).__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(obs_size, hidden_size),
                                       torch.nn.ReLU(),
                                       torch.nn.Linear(hidden_size, n_actions))


    def forward(self, x):
        y_hat = self.net(x)
        return y_hat

# Represents a single episode stores as the total undiscounted reward
Episode = namedtuple('Episode', field_names=['reward', 'steps'])

# Represents a single step that our agent made in the episode aong with the observation from the environment and what
# action was completed
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])


