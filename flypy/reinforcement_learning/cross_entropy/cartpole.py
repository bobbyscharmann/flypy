"""This function is used to implement the cross entropy method for Cartpole.

The idea here is to balance the cartpole OpenAI Gym environment using the Cross Entropy RL Method. Essentially this is
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
BATCH_SIZE = 16
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
        obs_v = torch.FloatTensor([obs])
        act_probs_v = sm(net(obs_v))

        # Both the NN and softmax return tensor that compute gradients so use .data to convert to numpy array
        act_probs = act_probs_v.data.numpy()[0]

        action = np.random.choice(len(act_probs), p=act_probs)
        next_obs, reward, is_done, _ = env.step(action)

        episode_reward += reward
        step = EpisodeStep(observation=obs, action=action)
        episode_steps.append(step)

        if is_done:
            e = Episode(reward=episode_reward, steps=episode_steps)
            batch.append(e)
            episode_reward = 0.0
            episode_steps = []
            next_obs = env.reset()

            if len(batch) == batch_size:
                yield batch
                batch = []

        obs = next_obs


# This function will filter a batch of data using a percentile threshold
def filter_batch(batch, percentile):
    rewards = list(map(lambda s: s.reward, batch))
    reward_bound = np.percentile(rewards, percentile)
    reward_mean = float(np.mean(rewards))

    train_obs = []
    train_act = []

    for example in batch:
        if example.reward < reward_bound:
            continue
        train_obs.extend(map(lambda step: step.observation, example.steps))
        train_act.extend(map(lambda step: step.action, example.steps))

    train_obs_v = torch.FloatTensor(train_obs)
    train_act_v = torch.LongTensor(train_act)
    return train_obs_v, train_act_v, reward_bound, reward_mean


if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    env.reset()

    # env = gym.wrappers.Monitor(env, directory="mon", force=True)

    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    net = CrossEntropyNN(obs_size, HIDDEN_SIZE, n_actions)
    objective = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=net.parameters(), lr=0.01)
    #writer = torch.utils.tensorboard.SummaryWriter(comment="-cartpole")

    episode_number = 1
    for iter_no, batch in enumerate(iterate_batches(env, net, BATCH_SIZE)):
        obs_v, act_v, reward_b, reward_m = filter_batch(batch, PERCENTILE)
        optimizer.zero_grad()
        action_scores_v = net(obs_v)
        loss_v = objective(action_scores_v, act_v)
        loss_v.backward()
        optimizer.step()
        print(f"Episode: {iter_no}, Reward: {reward_m}, Loss: {loss_v}")

        if reward_m > 199:
            env.render()
            print(f"SOLVED: {reward_m}")
            break
        episode_number += 1

