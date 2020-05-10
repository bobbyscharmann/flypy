import random


class Environment(object):
    steps_remaining: int

    def __init__(self):
        self.steps_remaining = 10
        return

    def action(self, action: int) -> float:
        """Handles the Agent's action and returns the reward for said action"""
        reward = random.random()
        if self.is_done():
            raise Exception("Game over!")
        self.steps_remaining -= 1
        return reward

    def get_observation(self):
        """Environments current observations for the Agent. This could be positions, velocities, etc."""
        return [0.0, 0.0, 0.0]

    @staticmethod
    def get_actions(self):
        """List of actions available for execution"""
        return [0, 1]

    def is_done(self) -> bool:
        """Done with this episode? Number of steps defined in steps_remaining class variable"""
        return self.steps_remaining == 0


class Agent(object):
    def __init__(self, env: Environment):
        self.environment = env
        return

    def step(self):
        return env.action(random.choice(self.environment.get_observation()))


if __name__ == "__main__":
    env = Environment()
    agent = Agent(env)

    while not env.is_done():
        print(agent.step())
