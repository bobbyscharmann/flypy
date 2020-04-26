import gym

env = gym.make("MountainCar-v0")
env.reset()

done = False

while not done:
    print(env.observation_space.high)
    print(env.observation_space.low)
    print(env.action_space.n)
    action = 2
    new_state, reward, done, _ = env.step(action)
    env.render()

env.close()