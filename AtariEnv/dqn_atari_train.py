# dqn_atari_train.py
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import DummyVecEnv
import os
import ale_py

gym.register_envs(ale_py)

# # ----- CONFIGURATION -----
GAME_NAME = "ALE/Breakout-v5"
# TOTAL_TIMESTEPS = 10_000_000
# MODEL_DIR = "./models"
# MODEL_NAME = "dqn_atari"

# os.makedirs(MODEL_DIR, exist_ok=True)


# # ----- CREATE ENVIRONMENT -----
def make_env():
    env = gym.make(GAME_NAME)
    env = AtariWrapper(env)  # Handles frame stack, grayscale, resizing, and clipping
    return env


env = DummyVecEnv([make_env])

# # ----- CREATE DQN MODEL -----
# model = DQN(
#     "CnnPolicy",
#     env,
#     learning_rate=1e-4,
#     buffer_size=100_000,
#     learning_starts=50_000,
#     batch_size=32,
#     gamma=0.99,
#     train_freq=4,
#     target_update_interval=10_000,
#     exploration_fraction=0.1,
#     exploration_final_eps=0.01,
#     verbose=1,
#     tensorboard_log="./tensorboard/",
# )

# # ----- TRAIN MODEL -----
# model.learn(total_timesteps=TOTAL_TIMESTEPS)

# # ----- SAVE MODEL -----
# model.save(os.path.join(MODEL_DIR, MODEL_NAME))
# print(f"Model saved to {os.path.join(MODEL_DIR, MODEL_NAME)}")

# env.close()

model = DQN(
    "CnnPolicy",
    env,
    learning_rate=1e-4,
    buffer_size=100_000,
    batch_size=32,
    gamma=0.99,
    train_freq=4,
    target_update_interval=10_000,
    exploration_fraction=0.2,
    exploration_final_eps=0.01,
    verbose=1,
    tensorboard_log="./tensorboard/",
    device="cuda",
)

model.learn(total_timesteps=50_000_000)  # train longer
model.save("./models/dqn_atari_enhanced")
