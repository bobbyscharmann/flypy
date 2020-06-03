"""Q Learning value iteration implementation of OpenAI Gym Frozen Lake"""

import numpy as np
import collections
from tensorboardX import SummaryWriter
import gym

# Key is "source state" + "action" + "target state" and the value is the immediate reward
reward_table = dict()

# Key is the "state" + "action" and the value is a dict that maps the target state into a count of the times we have
# seen it such that if we implement action A1 ten times and three times that leads us to state 4 and 7 times it leads us
# to state 5, this dict will be {4: 3, 5: 7}. This table is used to estimate the probabilities of transitions.
transition_table = dict()

# Dictionary that maps the state into the value for the state
value_table = dict()

GAMMA = 0.9
TEST_EPISODES = 20
