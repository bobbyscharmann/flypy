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


class Agent:
    def __init__(self):
        self.env = gym.make('FrozenLake-v0')
        self.state = self.env.reset()

        # defaultdict is basically a Python dictionary that does not raise a KeyError and instead returns a default
        self.reward_table = collections.defaultdict(float)
        self.transition_table = collections.defaultdict(collections.Counter)
        self.value_table = collections.defaultdict(float)

    # This function is used to gather random experience from the environment in order to prime the reward and transition
    # tables.
    def play_n_random_steps(self, count):
        for _ in range(count):
            action = self.env.action_space.sample()
            new_state, reward, is_done, _ = self.env.step(action)
            self.reward_table[(self.state, action, new_state)] = reward_table
            self.transition_table[(self.state, action)][new_state] += 1
            self.state = self.env.reset() if is_done else new_state

    def calc_action_value(self, state, action):
        target_counts = self.transition_table[(state, action)]
        total = sum(target_counts.values())
        action_value = 0.0
        for tgt_state, count in target_counts.items():
            reward = self.reward_table[(state, action, tgt_state)]
            val = reward + GAMMA * self.value_table[tgt_state]
            action_value += (count / total) * val

        return action_value
