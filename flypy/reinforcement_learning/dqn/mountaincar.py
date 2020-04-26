import gym
import numpy as np
import random

env = gym.make("MountainCar-v0")
env.reset()

NUM_POSSIBLE_ACTIONS = env.action_space.n
NUM_POSSIBLE_STATES = 20 * len(env.observation_space.high)
q_space = np.zeros((NUM_POSSIBLE_STATES, NUM_POSSIBLE_ACTIONS))

# Ratio between exploring and exploiting the environment
EPSILON = 0.25
LEARNING_RATE = 0.1

# Discount factor
GAMMA = 0.8
done = False

action = 2
while not done:
    print(env.observation_space.high)
    print(env.observation_space.low)
    new_state, reward, done, _ = env.step(action)
    print(f"new_state: {new_state}")
    env.render()
    # Bellman Equation
    state = round(np.abs(new_state[0] / NUM_POSSIBLE_STATES)) + 1
    q_space[state, action] = q_space[state, action] + LEARNING_RATE * (reward + GAMMA * np.max(q_space[new_state, :]) - q_space[state, action])
    if random.uniform(0, 1) < EPSILON:
        # Explore: Select a random action
        action = random.choice([0, 1, 2])
    else:
        # Exploit: Select the best action (highest reward)
        np.max(q_space[state, :])
env.close()