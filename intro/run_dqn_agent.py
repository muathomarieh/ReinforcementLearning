import gymnasium as gym
import numpy as np
from dqn_agent import DQNAgent
from train_dqn_agent import Env
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--env_name", type=str, default="FrozenLake-v1", help="Environment name"
)
parser.add_argument(
    "--model_path",
    type=str,
    default="dqn_model_keras_FL.keras",
    help="Trained model path",
)
parser.add_argument("--episodes", type=int, default=5, help="Number of run episodes")
args = parser.parse_args()


def run(env, model_path="dqn_model_keras_FL.keras", episodes=5):

    discrete = isinstance(env.observation_space, gym.spaces.Discrete)

    if discrete:
        state_shape = (env.observation_space.n,)
    else:
        state_shape = env.observation_space.shape

    agent = DQNAgent(state_shape, env.action_space.n, discrete)
    agent.load(model_path)

    for ep in range(episodes):
        state, _ = env.reset()
        state = agent.preprocess_state(state)
        total_reward = 0
        terminated = False
        truncated = False
        while not (terminated or truncated):
            # Greedy action only
            action = agent.predict_action(state)

            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = agent.preprocess_state(next_state)
            state = next_state
            total_reward += reward
        print(f"Episode {ep+1}, Reward: {total_reward}")


env_enum = Env.CartPole
env_instance = env_enum.make()
run(env=env_instance, model_path=f"{env_enum.env_name()}.keras")
