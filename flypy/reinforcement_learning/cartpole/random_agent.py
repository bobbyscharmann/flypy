"""Implementation of the OpenAI Gym CartPole exercise with a random sampling of the action space.
In other words, a vey simplistic (or dumb) agent)"""
import gym

env = gym.make("CartPole-v0")
env.reset()

# Couple of variables for knowing when the episode is over
done: bool = False

# Keeping track of total aware and steps
total_reward: float = 0
steps: int = 0

# Iterate until the episode is over
while not done:
    # Step the environment choosing a random action from the Environment action space
    state, reward, done, info = env.step(action=env.action_space.sample())

    # Accumulate steps and awards
    steps += 1
    total_reward += reward
    env.render()

# Print some general information about the episode
print(f"Total Reward in {total_reward} in {steps} steps.")