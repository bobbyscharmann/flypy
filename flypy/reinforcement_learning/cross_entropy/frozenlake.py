"""This function is used to implement the cross entropy method for Frozen Lake

The idea here is to solve the Frozen Lake OpenAI Gym environment using the Cross Entropy RL Method. Essentially this is
a model-free (no apriori knowledge of the environment is assumed), policy-based (select next action based on probability
distribution of the action space), and on-policy which means it uses fresh data to infer the next action.

A Simple neural network is trained to learn the behavior to map observations to actions with the 30% of actions
resulting in the most reward being selected for model training.

Author: Bob Scharmann
Reference: Deep Reinforcement Learning Hands-On by Maxim Lapan
"""

import gym
import numpy as np
import torch
from collections import namedtuple

# Configuration options
HIDDEN_SIZE = 128
BATCH_SIZE = 100
PERCENTILE = 70

# Represents a single episode stores as the total undiscounted reward
Episode = namedtuple('Episode', field_names=['reward', 'steps'])

# Represents a single step that our agent made in the episode aong with the observation from the environment and what
# action was completed
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])


# Define a neural network to map observations to actions
class CrossEntropyNN(torch.nn.Module):

    def __init__(self, obs_size, hidden_size, n_actions):
        """obs_size - number of elements in this Gym's Observation space
           hidden_size - number of neurons in the hidden layer
           n_actions - size of the possible action space within this Gym environment"""

        super(CrossEntropyNN, self).__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(obs_size, hidden_size),
                                       torch.nn.ReLU(),
                                       torch.nn.Linear(hidden_size, n_actions))

    def forward(self, x):
        y_hat = self.net(x)
        return y_hat


# Create a one hot encode wrapper for the discrete observation space
class DiscreteOneHotWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(DiscreteOneHotWrapper, self).__init__(env)
        # Ensure observation space is discrete
        assert isinstance(env.observation_space, gym.spaces.Discrete)

        shape = (env.observation_space.n, )
        self.observation_space = gym.spaces.Box(0.0, 1.0, shape, dtype=np.float)

    def observation(self, observation):
        res = np.copy(self.observation_space.low)
        res[observation] = 1.0
        return res


# Generator function to iterate over batches
def iterate_batches(env, net, batch_size):
    # Initialize common variables and reset the Gym environment
    batch = []
    episode_reward = 0.0
    episode_steps = []
    obs = env.reset()

    # Softmax will convert logits to a probabilistic distribution from which the agent will select the next action.
    sm = torch.nn.Softmax(dim=1)

    # Generator (uses yield) to relinquish control to calling function so don't worry about an infinite loop
    while True:
        # Observation vector
        obs_v = torch.FloatTensor([obs])

        # Run observations through the NN and then apply Softmax to logits. This results in a vector of probabilities
        # for each possible action
        act_probs_v = sm(net(obs_v))

        # Both the NN and softmax return tensor that compute gradients so use .data to convert to numpy array
        act_probs = act_probs_v.data.numpy()[0]

        # Select the next action based on the probability distributions
        action = np.random.choice(len(act_probs), p=act_probs)

        # Step in the environment
        next_obs, reward, is_done, _ = env.step(action)

        # Accumulate reward
        episode_reward += reward

        # Build and log the step
        step = EpisodeStep(observation=obs, action=action)
        episode_steps.append(step)

        # Once the episode is over (reached max steps allowable by OpenAI Gym or succeeded)
        if is_done:

            # log this episode
            e = Episode(reward=episode_reward, steps=episode_steps)
            batch.append(e)

            # Reset variables between each episode
            episode_reward = 0.0
            episode_steps = []
            next_obs = env.reset()

            # Once we've ran through the batches, yield control from the generator to the training process
            if len(batch) == batch_size:
                yield batch
                batch = []

        # Update the observation the step
        obs = next_obs


# This function will filter a batch of data using a percentile threshold
def filter_batch(batch, percentile):
    # Take the batch namedtuple and extract out a list of the rewards
    rewards = list(map(lambda s: s.reward, batch))

    # Compute the reward value of which PERCENTILE (say 70%) of episodes were lower than
    reward_bound = np.percentile(rewards, percentile)

    # Find the mean reward value - useful for determining if the Agent is getting better or worse
    reward_mean = float(np.mean(rewards))

    train_obs = []
    train_act = []

    # Look at each example and see if it should be included or removed
    for example in batch:
        # If it's less than the desired percentile boundary, ignore it (not a good example to train on)
        if example.reward < reward_bound:
            continue

        # Add the example to the observation and action space for training
        train_obs.extend(map(lambda step: step.observation, example.steps))
        train_act.extend(map(lambda step: step.action, example.steps))

    # Convert these to torch types for use in training (32-bit float, actions or 64-bit signed integer)
    train_obs_v = torch.FloatTensor(train_obs)
    train_act_v = torch.LongTensor(train_act)
    return train_obs_v, train_act_v, reward_bound, reward_mean


if __name__ == "__main__":
    # Create the environment and clear it
    env = DiscreteOneHotWrapper(gym.make("FrozenLake-v0"))
    env.reset()
    episode_id = [40]
    env = gym.wrappers.Monitor(env, directory="mon", video_callable=lambda episode_id: True,force=True)

    # Find out the size of the observation and action space
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # Create our neural network and define the loss and optimizer functions
    net = CrossEntropyNN(obs_size, HIDDEN_SIZE, n_actions)
    objective = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=net.parameters(), lr=0.01)

    # For each batch
    for iter_no, batch in enumerate(iterate_batches(env, net, BATCH_SIZE)):
        # Filter out the undesirable examples
        obs_v, act_v, reward_b, reward_m = filter_batch(batch, PERCENTILE)
        # Zero gradients and then run observations through the NN to get actions
        optimizer.zero_grad()
        action_scores_v = net(obs_v)

        # Compute the loss
        loss_v = objective(action_scores_v, act_v)

        # Backwards propagate and step the optimizer (hopefully will learn!)
        loss_v.backward()
        optimizer.step()
        print(f"Episode: {iter_no}, Reward: {reward_m}, \tLoss: {loss_v}")

        # Reward of 200 in OpenAI Gym implies success
        if reward_m > 199:
            print(f"SOLVED: {reward_m}")
            break

