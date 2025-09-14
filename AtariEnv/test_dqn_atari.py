import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import DummyVecEnv
import ale_py

GAME_NAME = "ALE/Breakout-v5"
MODEL_PATH = "./models/dqn_atari.zip"


def make_env():
    env = gym.make(GAME_NAME, render_mode="human")
    env = AtariWrapper(env)  #
    return env


env = DummyVecEnv([make_env])

model = DQN.load(MODEL_PATH, env=env)

NUM_EPISODES = 5

for episode in range(NUM_EPISODES):
    obs = env.reset()
    done = False
    total_reward = 0

    while True:
        action, _states = model.predict(obs)

        obs, rewards, dones, infos = env.step(action)
        total_reward += rewards[0]

        if dones[0]:
            print(f"Episode {episode+1} finished with reward: {total_reward}")
            break

env.close()
