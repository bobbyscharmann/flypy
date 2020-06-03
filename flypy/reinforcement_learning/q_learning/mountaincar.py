import gym
import numpy as np
import random

env = gym.make("MountainCar-v0")
NUM_POSSIBLE_ACTIONS = env.action_space.n  # Position and Velocity

num_states = (env.observation_space.high - env.observation_space.low) * np.array([10, 100]) #np.zeros((NUM_POSSIBLE_STATES, NUM_POSSIBLE_ACTIONS))
num_states = np.round(num_states, 0).astype(int) + 1
print(num_states)
Q = np.random.uniform(low=-1, high=1, size=(num_states[0], num_states[1], NUM_POSSIBLE_ACTIONS))

# Ratio between exploring and exploiting the environment
EPSILON = 0.02
LEARNING_RATE = 0.1

# Discount factor
GAMMA = 0.8
done = False

for i in range(50):
    action = 2
    done = False
    state = env.reset()
    state_adj = (state - env.observation_space.low) * np.array([10, 100])
    state_adj = np.round(state, 0).astype(int)
    total_reward = 0
    while not done:
        #print(env.observation_space.high)
        #print(env.observation_space.low)

        if i > 0:
            env.render()
        if random.uniform(0, 1) < EPSILON:
            # Explore: Select a random action
            action = random.choice([0, 1, 2])
        else:
            # Exploit: Select the best action (highest reward)
            action = np.argmax(Q[state_adj[0], state_adj[1]])

        state2, reward, done, _ = env.step(action)

        # Discretize state
        state2_adj = (state2 - env.observation_space.low) * np.array([10, 100])
        state2_adj = np.round(state2_adj, 0).astype(int)

        # Bellman Equation
        delta = LEARNING_RATE * (reward +
                           GAMMA * np.max(Q[state2_adj[0],
                                            state2_adj[1]]) -
                            Q[state_adj[0], state_adj[1], action])
        Q[state_adj[0], state_adj[1], action] += delta
        #Q[state[0], state[1], action] = q_space[state[0], state[1], action] + LEARNING_RATE * (reward + GAMMA * np.max(q_space[new_state[0], new_state[1]]) - q_space[state[0], state[1], action])
        state_adj = state2_adj
        total_reward += reward
    print(f"Eposode: {i}, Reward: {total_reward}")
env.close()