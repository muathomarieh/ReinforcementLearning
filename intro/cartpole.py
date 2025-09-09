import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")
state, info = env.reset()

print(env.observation_space.shape)
discrete = isinstance(env.observation_space, gym.spaces.Discrete)
print(env.observation_space)
print(discrete)

a = (12,)
print(a[0])
