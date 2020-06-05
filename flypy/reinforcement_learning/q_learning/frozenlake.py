"""Q Learning value iteration implementation of OpenAI Gym Frozen Lake"""
import collections
from tensorboardX import SummaryWriter
import gym

GAMMA = 0.9
TEST_EPISODES = 20


class Agent:
    def __init__(self):
        self.env = gym.make('FrozenLake-v0')
        self.state = self.env.reset()

        # Key is "source state" + "action" + "target state" and the value is the immediate reward
        # defaultdict is basically a Python dictionary that does not raise a KeyError and instead returns a default
        self.reward_table = collections.defaultdict(float)

        # Key is the "state" + "action" and the value is a dict that maps the target state into a count of the times we
        # have seen it such that if we implement action A1 ten times and three times that leads us to state 4 and 7
        # times it leads to state 5, this dict will be {4: 3, 5: 7}. This table is used to estimate the probabilities of
        # transitions.
        self.transition_table = collections.defaultdict(collections.Counter)

        # Dictionary that maps the state into the value for the state
        self.value_table = collections.defaultdict(float)

    def play_n_random_steps(self, count):
        """ This function is used to gather random experience from the environment in order to prime the reward and
        transition tables.

        :param count:
        :return:
        """
        # For each step
        for _ in range(count):
            # Sample the actions space
            action = self.env.action_space.sample()

            # Execute the action to get the reward
            new_state, reward, is_done, _ = self.env.step(action)

            # Update the reward and transition table
            self.reward_table[(self.state, action, new_state)] = reward
            self.transition_table[(self.state, action)][new_state] += 1
            self.state = self.env.reset() if is_done else new_state

    def calc_action_value(self, state, action):
        """
        This function is used to compute value from the state and action provided. This wil use the Bellman equation to
        compute the future rewards.
        :param state: The current state.
        :param action:
        :return:
        """
        # Number of times we have transition from this state in the past
        target_counts = self.transition_table[(state, action)]

        # Total number of times we have transitioned from state inthe past
        total = sum(target_counts.values())
        action_value = 0.0

        # For each target state we have transitioned to from state
        for tgt_state, count in target_counts.items():
            # Look up the reward
            reward2 = self.reward_table[(state, action, tgt_state)]

            # Compute the discounted reward
            val = reward2 + GAMMA * self.value_table[tgt_state]

            # Compute the fraction of the reward for each state it can transition to due to each discounted value
            action_value += (count / total) * val

        return action_value

    def select_action(self, state):
        """This function is used to select the best action (that with the highest value) from the current state
        :param state: The current state.
        """
        best_action, best_value = None, None

        # For each possible action we can transition to from this state
        for action in range(self.env.action_space.n):
            # Compute teh action value
            action_value = self.calc_action_value(state, action)

            # Update the best value and action if this is the best one
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_action

    def play_episode(self, env):
        """Play an episode in the environment. This initial state is set based on resetting the environment.

        :param env: The environment to play the episode in.
        :return:
        """
        total_reward = 0.0
        state = env.reset()
        while True:
            # Select the best action for the current state
            action = self.select_action(state)

            # Take that action
            new_state, reward, is_done, _ = env.step(action)

            # Update the reward table
            self.reward_table[(state, action, new_state)] = reward
            self.transition_table[state, reward][new_state] += 1

            # Accumulate the total reward
            total_reward += reward
            if is_done:
                break

            # Update the state
            state = new_state

        return total_reward

    def value_iteration(self):
        for state in range(self.env.observation_space.n):
            state_values = [self.calc_action_value(state, action) for action in range(self.env.action_space.n)]
            self.value_table[state] = max(state_values)


if __name__ == "__main__":
    test_env = gym.make('FrozenLake-v0')
    agent = Agent()
    writer = SummaryWriter(comment="-v-iteration")
    iter_no = 0
    best_reward = 0.0
    while True:
        iter_no += 1
        agent.play_n_random_steps(100)
        agent.value_iteration()
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

    writer.close()

