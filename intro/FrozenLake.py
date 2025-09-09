import gymnasium as gym
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.losses import MSE
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from collections import deque, namedtuple
import random
import matplotlib.pyplot as plt


import os
import warnings


os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")


env = gym.make(
    "FrozenLake-v1",
    desc=None,
    map_name="8x8",
    is_slippery=False,
    render_mode="rgb_array",
)
env.reset()


class RewardWrapper(gym.Wrapper):
    def __init__(self, env):
        self.step_count = 0
        super().__init__(env)

    def reset(self, **kwargs):
        self.step_count = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)

        # Example: give -0.001 for each step to encourage faster solving
        if reward == 0 and not terminated:
            reward = -0.001

        if terminated and reward == 0:
            reward = -0.1  # Example: penalize falling into a hole

        # Example: increase reward for reaching goal
        if reward == 1:
            efficiency_bonus = max(0, (50 - self.step_count) / 50)
            reward = 1 + efficiency_bonus

        return state, reward, terminated, truncated, info


env = RewardWrapper(env)

# hyperparameters
num_episodes = 20000
num_steps_for_episode = 200
memory_size = 100_000
batch_size = 64
update_frequency = 16
gamma = 0.995
epsilon = 1.0
epsilon_decay = 0.9999
min_epsilon = 0.05
TAU = 1e-3
learning_rate = 0.001  # 0.001

actions_size = env.action_space.n
state_size = env.observation_space.n
print(f"Number of actions: {actions_size}, Number of states: {state_size}")


def create_network():
    return Sequential(
        [
            Input(shape=(state_size,)),
            Dense(64, activation="relu"),
            Dense(64, activation="relu"),
            Dense(actions_size, activation="linear"),
        ]
    )


q_target_network = create_network()

q_network = create_network()

optimizer = Adam(learning_rate=learning_rate)


def compute_loss(experiences):
    states, actions, rewards, next_states, dones = experiences

    # Get the predicted Q-values from the Q-network
    q_values = q_network(states)
    q_values = tf.gather_nd(
        q_values,
        tf.stack([tf.range(q_values.shape[0]), tf.cast(actions, tf.int32)], axis=1),
    )

    # Get the maximum Q-values for the next states from the target network
    next_q_values = q_target_network(next_states)
    max_next_q_values = tf.reduce_max(next_q_values, axis=1)

    # Compute the target Q-values
    target_q_values = rewards + (1 - dones) * gamma * max_next_q_values

    # Compute the loss
    loss = MSE(target_q_values, q_values)
    return loss


Experience = namedtuple(
    "Experience", ["state", "action", "reward", "next_state", "done"]
)


def agent_learn(experiences, gamma):

    with tf.GradientTape() as tape:
        loss = compute_loss(experiences=experiences)

    grads = tape.gradient(loss, q_network.trainable_variables)
    optimizer.apply_gradients(zip(grads, q_network.trainable_variables))

    # update the weights of the target network soft update
    for target_param, param in zip(
        q_target_network.trainable_variables, q_network.trainable_variables
    ):
        target_param.assign(param * TAU + target_param * (1 - TAU))


def get_action(q_values, epsilon):
    if random.random() > epsilon:
        return np.argmax(q_values.numpy()[0])
    else:
        return random.choice(np.arange(actions_size))


def start_training(t, update_frequency, memory_buffer):
    return len(memory) > batch_size and (t + 1) % update_frequency == 0


def get_experiences(memory, batch_size):
    experiences = random.sample(memory, k=batch_size)
    states = tf.convert_to_tensor(
        np.array([tf.one_hot(e.state, state_size) for e in experiences]),
        dtype=tf.float32,
    )
    actions = tf.convert_to_tensor(
        np.array([e.action for e in experiences if e is not None]), dtype=tf.int32
    )
    rewards = tf.convert_to_tensor(
        np.array([e.reward for e in experiences if e is not None]), dtype=tf.float32
    )
    next_states = tf.convert_to_tensor(
        np.array([tf.one_hot(e.next_state, state_size) for e in experiences]),
        dtype=tf.float32,
    )
    done_vals = tf.convert_to_tensor(
        np.array([e.done for e in experiences if e is not None]).astype(np.uint8),
        dtype=tf.float32,
    )
    return (states, actions, rewards, next_states, done_vals)


memory = deque(maxlen=memory_size)
q_target_network.set_weights(q_network.get_weights())
all_rewards = []

for episode in range(num_episodes):
    state, _ = env.reset()
    episode_reward = 0
    for t in range(num_steps_for_episode):

        state_tensor = tf.expand_dims(tf.one_hot(state, state_size), axis=0)
        q_values = q_network(state_tensor)

        action = get_action(q_values, epsilon)
        next_state, reward, done, truncated, info = env.step(action)
        episode_reward += reward

        memory.append(Experience(state, action, reward, next_state, done))

        if start_training(t, update_frequency, memory):
            experiences = get_experiences(memory, batch_size)
            agent_learn(experiences, gamma)
        state = next_state
        if done:
            break

    all_rewards.append(episode_reward)
    epsilon = max(min_epsilon, epsilon * epsilon_decay)
    if (episode + 1) % 100 == 0:
        print(
            f"Episode: {episode + 1}, Reward: {episode_reward}, Epsilon: {epsilon:.3f}"
        )
        print(
            f"Average Reward over last 100 episodes: {np.mean(all_rewards[-100:]):.2f}"
        )
        print(
            f"success rate: {sum(1 for r in all_rewards[-100:] if r > 0) / len(all_rewards[-100:]):.2%}"
        )

    succes_rate = sum(1 for r in all_rewards[-100:] if r > 0) / len(all_rewards[-100:])
    if succes_rate >= 0.85:
        print(
            f"success rate: {sum(1 for r in all_rewards[-100:] if r > 0) / len(all_rewards[-100:]):.2%}"
        )
        print(f"Solved in episode {episode + 1}!")
        break


q_network.save("dqn_model_keras_FL.keras.keras")
