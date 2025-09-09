import gymnasium as gym
import ale_py

gym.register_envs(ale_py)

env = gym.make("ALE/Breakout-v5", render_mode="human")

print(env.observation_space)
print(env.observation_space.shape)
print(env.action_space)

obs, info = env.reset()
done = 0

while not done:
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    done = terminated
env.close()
