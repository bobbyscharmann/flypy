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
class CrossEntropyNN(torch.nn.Module):

    def __init__(self, obs_size, hidden_size, n_actions):
        super(CrossEntropyNN, self).__init__()
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

episode_steps = []
episode_reward = 0.0
batch = []


def iterate_batches(env, net, batch_size):
    batch = []
    episode_reward = 0.0
    episode_steps = []
    obs = env.reset()
    sm = torch.nn.Softmax(dim=1)

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


def filter_batch(batch, percentile):
    rewards = list(map(lambda s: s.reward, batch))
    reward_bound = np.percentile(rewards, percentile)
    reward_mean = float(np.mean(rewards))

    train_obs = []
    train_act = []

    for reward, steps in batch:
        if reward > reward_bound:
            continue
        train_obs.extend(map(lambda step: step.observation, steps))
        train_act.extend(map(lambda step: step.action, steps))

        train_obs_v = torch.FloatTensor(train_obs)
        train_act_v = torch.LongTensor(train_act)
        return train_obs_v, train_act_v, reward_bound, reward_mean


if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    env.reset()

    # env = gym.wrappers.Monitor(env, directory="mon", force=True)

    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    net = CrossEntropyNN(4, HIDDEN_SIZE, n_actions)
    objective = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=net.parameters(), lr=0.01)
    #writer = torch.utils.tensorboard.SummaryWriter(comment="-cartpole")

    for iter_no, batch in enumerate(iterate_batches(env, net, BATCH_SIZE)):
        obs_v, act_v, reward_b, reward_m = filter_batch(batch, PERCENTILE)
        optimizer.zero_grad()
        action_scores_v = net(obs_v)
        loss_v = objective(action_scores_v, act_v)
        loss_v.backward()
        optimizer.step()
        print(f"Reward: {reward_m}")

        if reward_m > 199:
            print(f"SOLVED: {reward_m}")
            break

