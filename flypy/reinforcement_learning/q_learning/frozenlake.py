"""Tabular Q Learning implementation of OpenAI Gym Frozen Lake"""
import collections
from tensorboardX import SummaryWriter
import gym

# Discount Factor
GAMMA = 0.9

# Learning Rate - helps the Q table converge smoothly
ALPHA = 0.2
TEST_EPISODES = 20


class Agent:
    def __init__(self):
        self.env = gym.make('FrozenLake-v0')
        self.state = self.env.reset()

        # Dictionary that maps the state into the value for the state
        self.values = collections.defaultdict(float)

    def sample_env(self):
        action = self.env.action_space.sample()
        old_state = self.state
        new_state, reward, is_done, _ = self.env.step(action)
        self.state = self.env.reset() if is_done else new_state
        return old_state, action, reward, new_state

    def best_value_and_action(self, state):
        """This function is used to select the best action (that with the highest value) from the current state
        :param state: The current state.
        """
        best_action, best_value = None, None

        # For each possible action we can transition to from this state
        for action in range(self.env.action_space.n):
            # Compute the action value
            action_value = self.values[(state, action)]

            # Update the best value and action if this is the best one
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_value, best_action

    def value_update(self, state, action, reward, next_state):
        """Update the vaue table from taking a step from the environment and applying the learning rate."""
        best_value, _ = self.best_value_and_action(next_state)
        new_value = reward + GAMMA * best_value
        old_value = self.values[(state, action)]
        self.values[(state, action)] = old_value * (1 - ALPHA) + new_value * ALPHA

    def play_episode(self, env):
        """Play an episode in the environment. This initial state is set based on resetting the environment.

        :param env: The environment to play the episode in.
        :return:
        """
        total_reward = 0.0
        state = env.reset()
        while True:
            # Select the best action for the current state
            _, action = self.best_value_and_action(state)

            # Take that action
            new_state, reward, is_done, _ = env.step(action)

            # Accumulate the total reward
            total_reward += reward
            if is_done:
                break

            # Update the state
            state = new_state

        return total_reward



if __name__ == "__main__":
    test_env = gym.make('FrozenLake-v0')
    agent = Agent()
    writer = SummaryWriter(comment="-q-learning")
    iter_no = 0
    best_reward = 0.0
    while True:
        iter_no += 1
        agent.sample_env()
        state, action, reward, next_state = agent.sample_env()
        agent.value_update(state, action, reward, next_state)

        reward = 0.0
        for _ in range(TEST_EPISODES):
            reward += agent.play_episode(test_env)
        reward /= TEST_EPISODES
        writer.add_scalar("reward", reward, iter_no)
        if reward > best_reward:
            print(f"Best reward updated from  {best_reward}->{reward}")
            best_reward = reward

        if reward > 0.80:
            print(f"Solved in {iter_no} iterations.")
            iter_no = 0

        writer.close()

