import gymnasium as gym
import numpy as np
from dqn_agent import DQNAgent
import matplotlib.pyplot as plt
from enum import Enum
import ale_py

gym.register_envs(ale_py)


class Env(Enum):
    CartPole = "CartPole"
    LunarLander = "LunarLander-v3"
    FrozenLake8x8 = "FrozenLake-v1-8x8"
    FrozenLake4x4 = "FrozenLake-v1-4x4"
    MountainCar = "MountainCar"
    Breakout = "ALE/Breakout-v5"

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
            "ALE/Breakout-v5": lambda: gym.make(
                "ALE/Breakout-v5", obs_type="grayscale", render_mode=None
            ),
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


class AtariWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(84, 84, 1), dtype=np.uint8
        )

    def observation(self, obs):
        import cv2

        if obs.ndim == 3 and obs.shape[-1] == 3:
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        elif obs.ndim == 2:
            pass
        else:
            obs = obs.squeeze()
        obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
        return np.expand_dims(obs, axis=-1)


def train(
    env,
    episodes=15000,
    max_steps=1000,  # Reduced max steps -> 1000 for the MountainCar
    save_path="dqn_model_keras.keras",
):

    if (
        hasattr(env.observation_space, "shape")
        and len(env.observation_space.shape) >= 2
    ):
        state_shape = (
            env.observation_space.shape + (1,)
            if len(env.observation_space.shape) == 2
            else env.observation_space.shape
        )
        is_atari = True

        agent = DQNAgent(
            state_shape=state_shape,
            action_size=env.action_space.n,
            gamma=0.99,
            lr=0.01,
            epsilon=1.0,
            epsilon_decay=0.9999,
            min_epsilon=0.05,
            tau=0.001,
            memory_size=50000,
            batch_size=64,
            update_frequency=16,
        )
    elif isinstance(env.observation_space, gym.spaces.Discrete):
        if hasattr(env.unwrapped, "nS"):
            n_states = env.unwrapped.nS
        else:
            n_states = env.observation_space.n
        state_shape = (n_states,)
        is_atari = False

        agent = DQNAgent(
            state_shape=state_shape,
            action_size=env.action_space.n,
        )

    else:
        state_shape = env.observation_space.shape
        is_atari = False

        agent = DQNAgent(
            state_shape=state_shape,
            action_size=env.action_space.n,
        )

    print(f"Environment: {type(env.unwrapped).__name__}")
    print(f"State shape: {state_shape}")
    print(f"Action size: {env.action_space.n}")
    print(f"Is Atari: {is_atari}")
    print("-" * 50)

    # discrete = isinstance(env.observation_space, gym.spaces.Discrete)

    # if discrete:
    #     if hasattr(env.unwrapped, "nS"):
    #         n_states = env.unwrapped.nS
    #     else:
    #         n_states = env.observation_space.n
    #     state_shape = (n_states,)
    # else:
    #     state_shape = env.observation_space.shape

    # hyperparameters
    # agent = DQNAgent(
    #     state_shape=state_shape,
    #     action_size=env.action_space.n,
    #     discrete=discrete,
    #     gamma=0.99,
    #     lr=0.001,
    #     epsilon=1.0,
    #     epsilon_decay=0.995,
    #     min_epsilon=0.01,
    #     tau=0.01,
    #     memory_size=10000,
    #     batch_size=32,
    #     update_frequency=4,
    # )

    rewards = []
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

            if done or truncated:
                break

        agent.update_epsilon()
        rewards.append(total_reward)

        # Progress reporting
        if (ep + 1) % 50 == 0:
            window = min(50, len(rewards))
            avg_reward = np.mean(rewards[-window:])
            print(
                f"Episode: {ep + 1}, Epsilon: {agent.epsilon:.4f}, Avg Reward: {avg_reward:.2f}"
            )

        # Early stopping for Atari (based on average reward)
        if is_atari and ep >= 100:
            recent_avg = np.mean(rewards[-100:])
            if recent_avg > 15:  # Good Breakout score threshold
                print(f"Solved! Episode {ep + 1}, Average reward: {recent_avg:.2f}")
                break

    agent.save(save_path)
    print(f"Model saved: {save_path}")

    # Plot results
    plt.figure(figsize=(10, 4))
    plt.plot(rewards)
    plt.title("Training Progress")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
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


if __name__ == "__main__":

    # env = RewardWrapper(gym.make("FrozenLake-v1", map_name="8x8", is_slippery=False))
    env_enum = Env.Breakout
    env_instance = env_enum.make()
    env_for_atari = AtariWrapper(env_instance)

    print(f"Training DQN on {env_enum.env_name()}: ")
    agent = train(env_for_atari, save_path=f"{env_enum.env_name()}.keras")
    test_agent(agent, env_for_atari)
