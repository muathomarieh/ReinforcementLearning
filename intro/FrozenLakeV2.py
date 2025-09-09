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

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")

env = gym.make('FrozenLake-v1', desc=None, map_name="8x8", is_slippery=True, render_mode="rgb_array")

class RewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.step_count = 0

    def reset(self, **kwargs):
        self.step_count = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        self.step_count += 1

        # Less harsh step penalty to encourage exploration
        if reward == 0 and not terminated:
            reward = -0.001  # Very small step penalty
        
        # Moderate penalty for falling into holes
        if terminated and reward == 0:
            reward = -0.1
        
        # Good reward for reaching goal, with bonus for efficiency
        if reward == 1:
            efficiency_bonus = max(0, (100 - self.step_count) / 100)
            reward = 1 + efficiency_bonus

        return state, reward, terminated, truncated, info

env = RewardWrapper(env)

# Improved hyperparameters
num_episodes = 5000
num_steps_for_episode = 200  # Reduced max steps
memory_size = 50_000
batch_size = 32
update_frequency = 10  # Less frequent updates
target_update_frequency = 100  # Update target network every 100 episodes
gamma = 0.99
epsilon = 1.0
min_epsilon = 0.05  # Higher minimum epsilon for continued exploration
epsilon_decay = 0.9995  # Slower decay
learning_rate = 0.0005  # Slower learning

actions_size = env.action_space.n
state_size = env.observation_space.n
print(f"Number of actions: {actions_size}, Number of states: {state_size}")

def create_network():
    return Sequential([
        Input(shape=(state_size,)),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(actions_size, activation='linear')
    ])

q_network = create_network()
q_target_network = create_network()
optimizer = Adam(learning_rate=learning_rate)

def compute_loss(experiences):
    states, actions, rewards, next_states, dones = experiences

    # Get the predicted Q-values from the Q-network
    q_values = q_network(states)
    q_values = tf.gather_nd(q_values, tf.stack([tf.range(q_values.shape[0]), tf.cast(actions, tf.int32)], axis=1))

    # Double DQN: use main network to select actions, target network to evaluate
    next_q_values_main = q_network(next_states)
    next_actions = tf.argmax(next_q_values_main, axis=1)
    
    next_q_values_target = q_target_network(next_states)
    max_next_q_values = tf.gather_nd(next_q_values_target, 
                                   tf.stack([tf.range(next_q_values_target.shape[0]), 
                                           tf.cast(next_actions, tf.int32)], axis=1))

    # Compute the target Q-values
    target_q_values = rewards + (1 - dones) * gamma * max_next_q_values

    # Huber loss for more stable training
    loss = tf.keras.losses.Huber()(target_q_values, q_values)
    return loss

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

def agent_learn(experiences):
    with tf.GradientTape() as tape:
        loss = compute_loss(experiences=experiences)

    grads = tape.gradient(loss, q_network.trainable_variables)
    # Gradient clipping for stability
    grads = [tf.clip_by_value(grad, -1.0, 1.0) for grad in grads]
    optimizer.apply_gradients(zip(grads, q_network.trainable_variables))
    
    return loss

def get_action(q_values, epsilon):
    if random.random() > epsilon:
        return np.argmax(q_values.numpy()[0])
    else:
        return random.choice(np.arange(actions_size))

def get_experiences(memory, batch_size):
    experiences = random.sample(memory, k=batch_size)
    states = tf.convert_to_tensor(
        np.array([tf.one_hot(e.state, state_size) for e in experiences]), dtype=tf.float32
    )
    actions = tf.convert_to_tensor(
        np.array([e.action for e in experiences]), dtype=tf.int32
    )
    rewards = tf.convert_to_tensor(
        np.array([e.reward for e in experiences]), dtype=tf.float32
    )
    next_states = tf.convert_to_tensor(
        np.array([tf.one_hot(e.next_state, state_size) for e in experiences]), dtype=tf.float32
    )
    done_vals = tf.convert_to_tensor(
        np.array([e.done for e in experiences]).astype(np.float32),
        dtype=tf.float32,
    )
    return (states, actions, rewards, next_states, done_vals)

# Training
memory = deque(maxlen=memory_size)
q_target_network.set_weights(q_network.get_weights())
all_rewards = []
success_count = 0
losses = []

print("Starting training...")

for episode in range(num_episodes):
    state, _ = env.reset()
    episode_reward = 0
    episode_loss = []
    
    for t in range(num_steps_for_episode):
        state_tensor = tf.expand_dims(tf.one_hot(state, state_size), axis=0)
        q_values = q_network(state_tensor)

        action = get_action(q_values, epsilon)
        next_state, reward, done, truncated, info = env.step(action)
        episode_reward += reward

        memory.append(Experience(state, action, reward, next_state, done or truncated))
        
        # Train less frequently but more consistently
        if len(memory) > batch_size and t % update_frequency == 0:
            experiences = get_experiences(memory, batch_size)
            loss = agent_learn(experiences)
            episode_loss.append(loss.numpy())
        
        state = next_state
        if done or truncated:
            break

    all_rewards.append(episode_reward)
    if episode_loss:
        losses.append(np.mean(episode_loss))
    
    # Update target network less frequently
    if episode % target_update_frequency == 0:
        q_target_network.set_weights(q_network.get_weights())
    
    # Slower epsilon decay
    epsilon = max(min_epsilon, epsilon * epsilon_decay)
    
    # Track success
    if episode_reward > 0.5:  # Reached goal
        success_count += 1
    
    if (episode + 1) % 100 == 0:
        recent_rewards = all_rewards[-100:]
        avg_reward = np.mean(recent_rewards)
        success_rate = sum(1 for r in recent_rewards if r > 0.5) / len(recent_rewards)
        
        print(f"Episode: {episode + 1}")
        print(f"  Avg Reward (last 100): {avg_reward:.3f}")
        print(f"  Success Rate (last 100): {success_rate:.2%}")
        print(f"  Epsilon: {epsilon:.3f}")
        if losses:
            print(f"  Avg Loss: {np.mean(losses[-100:]):.4f}")
        print()
        
        # More reasonable success criteria
        if success_rate >= 0.7:  # 70% success rate
            print(f"Environment solved! 70% success rate achieved in episode {episode + 1}")
            break

print(f"Training completed. Total successful episodes: {success_count}")

# Plot results
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(all_rewards)
plt.title('Episode Rewards')
plt.xlabel('Episode')
plt.ylabel('Reward')

plt.subplot(1, 2, 2)
window = 100
if len(all_rewards) >= window:
    moving_avg = [np.mean(all_rewards[i:i+window]) for i in range(len(all_rewards)-window+1)]
    plt.plot(range(window-1, len(all_rewards)), moving_avg)
    plt.title(f'Moving Average Reward (window={window})')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')

plt.tight_layout()
plt.show()

# Save the trained model
q_network.save("frozen_lake_model_improved.keras")
print("Model saved as 'frozen_lake_model_improved.keras'")