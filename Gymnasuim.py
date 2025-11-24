import gymnasium as gym
for i in gym.envs.registry.keys():
    print(i)

env = gym.make('FrozenLake-v1',
               render_mode='human')

while True:
    observation, info = env.reset()

