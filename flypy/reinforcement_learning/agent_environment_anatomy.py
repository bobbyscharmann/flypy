import random


class Environment(object):
    steps_remaining: int

    def __init__(self):
        self.steps_remaining = 10
        return

    def action(self, action: int) -> float:
        reward = random.random()
        self.steps_remaining -= 1
        return reward

    def get_observations(self):
        return [0.0, 0.0, 0.0]

    @staticmethod
    def get_actions(self):
        return [0, 1]

    def is_done(self) -> bool:
        return self.steps_remaining == 0


class Agent(object):
    def __init__(self, env: Environment):
        self.environment = env
        return

    def step(self):
        return env.action(random.choice(self.environment.get_observations()))


if __name__ == "__main__":
    env = Environment()
    agent = Agent(env)

    while not env.is_done():
        print(agent.step())
