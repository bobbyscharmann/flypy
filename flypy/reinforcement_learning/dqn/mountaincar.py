import gym
import numpy as np

env = gym.make("MountainCar-v0")
env.reset()

NUM_POSSIBLE_ACTIONS = env.action_space.n
NUM_POSSIBLE_STATES = len(env.observation_space.high)
q_space = np.zeros((NUM_POSSIBLE_STATES, NUM_POSSIBLE_ACTIONS))

# Ratio between exploring and exploiting the environment
EPSILON = 0.25

done = False

while not done:
    print(env.observation_space.high)
    print(env.observation_space.low)
    action = 2
    new_state, reward, done, _ = env.step(action)
    env.render()

env.close()