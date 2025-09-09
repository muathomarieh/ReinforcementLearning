import gymnasium as gym
import numpy as np
from dqn_agent import DQNAgent
import matplotlib.pyplot as plt
from enum import Enum


class Env(Enum):
    CartPole = "CartPole"
    LunarLander = "LunarLander-v3"
    FrozenLake8x8 = "FrozenLake-v1-8x8"
    FrozenLake4x4 = "FrozenLake-v1-4x4"
    MountainCar = "MountainCar"

    @staticmethod
    def _env_creators():
        return {
            "LunarLander-v3": lambda: gym.make(
                "LunarLander-v3",
                continuous=False,
                gravity=-10.0,
                enable_wind=False,
                wind_power=15.0,
                turbulence_power=1.5,
            ),
            "FrozenLake-v1-8x8": lambda: gym.make(
                "FrozenLake-v1", map_name="8x8", is_slippery=False
            ),
            "FrozenLake-v1-4x4": lambda: gym.make(
                "FrozenLake-v1", map_name="4x4", is_slippery=False
            ),
            "CartPole": lambda: gym.make("CartPole-v1", render_mode="human"),
            "MountainCar": lambda: gym.make(
                "MountainCar-v0", max_episode_steps=999, render_mode="human"
            ),
            "Breakout": lambda: gym.make("ALE/Breakout-v5"),
        }

    def make(self):
        creators = self._env_creators()
        if self.value in creators:
            return creators[self.value]()
        else:
            raise ValueError(f"No environment defined for {self.name}")

    def env_name(self):
        return self.name


class RewardWrapper(gym.Wrapper):
    def __init__(self, env):
        self.step_count = 0
        super().__init__(env)

    def reset(self, **kwargs):
        self.step_count = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)

        # More aggressive reward shaping
        if reward == 0 and not terminated:
            reward = -0.01  # Increased step penalty
        if terminated and reward == 0:  # Fell in hole
            reward = -1.0  # Much larger penalty for failure

        self.step_count += 1

        # Bonus for reaching goal efficiently
        if reward == 1:
            efficiency_bonus = max(0, (30 - self.step_count) / 30)
            reward = 1 + efficiency_bonus

        return state, reward, terminated, truncated, info


class MountainCarRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)

        position, velocity = state

        if terminated:
            reward = 100.0
        else:
            position_reward = position + 0.5

            velocity_reward = abs(velocity) * 0.1

            reward = position_reward + velocity_reward - 0.01

        return state, reward, terminated, truncated, info


def train(
    env,
    episodes=15000,
    max_steps=200,  # Reduced max steps -> 1000 for the MountainCar
    save_path="dqn_model_keras_FL_v0_8.keras",
):
    discrete = isinstance(env.observation_space, gym.spaces.Discrete)

    if discrete:
        if hasattr(env.unwrapped, "nS"):
            n_states = env.unwrapped.nS
        else:
            n_states = env.observation_space.n
        state_shape = (n_states,)
    else:
        state_shape = env.observation_space.shape

    # hyperparameters
    agent = DQNAgent(
        state_shape=state_shape,
        action_size=env.action_space.n,
        discrete=discrete,
        gamma=0.99,
        lr=0.001,
        epsilon=1.0,
        epsilon_decay=0.995,
        min_epsilon=0.01,
        tau=0.01,
        memory_size=10000,
        batch_size=32,
        update_frequency=4,
    )

    rewards = []
    losses = []
    dones = []
    for ep in range(episodes):
        state, _ = env.reset()
        state = agent.preprocess_state(state)
        total_reward = 0

        for step in range(max_steps):
            action = agent.get_action(state, env)
            next_state, reward, done, truncated, _ = env.step(action)
            next_state = agent.preprocess_state(next_state)

            agent.append_new_experience(state, action, reward, next_state, done)

            agent.global_step += 1
            if agent.start_training():
                experiences = agent.get_experiences_replay()
                agent.agent_learn(experiences)

            state = next_state
            total_reward += reward

            if done:
                dones.append(done)
            if truncated:
                print("Episode truncated (time limit)")
            if done or truncated:
                break

        epsilon = agent.update_epsilon()
        rewards.append(total_reward)

        # Check progress more frequently
        window = min(100, len(rewards))
        success_rate = sum(1 for r in rewards[-window:] if r > 0.5) / window
        done_rate = sum(dones[-window:]) / window

        if (ep + 1) % 500 == 0:
            print(f"Episode: {ep + 1}, Epsilon: {epsilon:.3f}")
            print(
                f"Average Reward over last {window} episodes: {np.mean(rewards[-window:]):.2f}"
            )
            print(f"Success rate: {success_rate:.2%}")
            print(f"done_rate: {done_rate:.2%}")
            print("-" * 50)

        # Early stopping with lower threshold initially
        if done_rate >= 0.8 and len(rewards) >= 1000:
            print(f"Solved in episode {ep + 1} with done_rate {done_rate:.2%}!")
            break
        # if success_rate >= 0.8 and len(rewards) >= 1000:
        #     print(f"Solved in episode {ep + 1} with success rate {success_rate:.2%}!")
        #     break

    agent.save(save_path)
    print(f"Model saved at {save_path}")

    # Plot training progress
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(rewards)
    plt.title("Rewards per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")

    plt.subplot(1, 2, 2)
    # Calculate rolling success rate
    window_size = 100
    success_rates = []
    for i in range(window_size, len(rewards)):
        window_rewards = rewards[i - window_size : i]
        success_rate = sum(1 for r in window_rewards if r > 0.5) / window_size
        success_rates.append(success_rate)

    plt.plot(success_rates)
    plt.title(f"Success Rate")
    plt.xlabel("Episode")
    plt.ylabel("Success Rate")
    plt.ylim(0, 1)

    plt.tight_layout()
    plt.show()

    return agent


# Test the agent after training
def test_agent(agent, env, episodes=100):
    successes = 0
    total_rewards = []
    dones_sum = 0

    for ep in range(episodes):
        state, _ = env.reset()
        state = agent.preprocess_state(state)
        total_reward = 0

        for step in range(200):
            action = agent.predict_action(state)  # No exploration
            next_state, reward, done, truncated, _ = env.step(action)
            state = agent.preprocess_state(next_state)
            total_reward += reward
            if done:
                dones_sum += 1
            if done or truncated:
                if reward > 0:
                    successes += 1
                break

        total_rewards.append(total_reward)

    success_rate = successes / episodes
    avg_reward = np.mean(total_rewards)

    print(f"Test Results over {episodes} episodes:")
    print(f"Success Rate: {success_rate:.2%}")
    print(f"Average Reward: {avg_reward:.3f}")

    return success_rate, avg_reward


# Run training
if __name__ == "__main__":
    # Create environment
    # env = RewardWrapper(gym.make("FrozenLake-v1", map_name="8x8", is_slippery=False))
    env_enum = Env.CartPole
    env_instance = env_enum.make()

    print(f"Training DQN on {env_enum.env_name()}: ")
    agent = train(env_instance, save_path=f"{env_enum.env_name()}.keras")
    test_agent(agent, env_instance)
