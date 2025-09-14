import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam
from collections import deque, namedtuple
import random


gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

Experience = namedtuple(
    "Experience", ["state", "action", "reward", "next_state", "done"]
)


class DQNAgent:
    def __init__(
        self,
        state_shape,
        action_size,
        gamma=0.99,
        lr=0.001,
        epsilon=1.0,
        epsilon_decay=0.995,
        min_epsilon=0.01,
        tau=0.01,
        memory_size=10000,
        batch_size=32,
        update_frequency=4,
    ):
        self.state_shape = state_shape
        self.action_size = action_size

        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.tau = tau
        self.batch_size = batch_size
        self.update_frequency = update_frequency

        self.memory = deque(maxlen=memory_size)

        # Networks
        self.q_network = self._build_network()
        self.target_network = self._build_network()
        self.target_network.set_weights(self.q_network.get_weights())

        self.optimizer = Adam(learning_rate=self.lr)
        self.loss_fn = Huber()

        # Global step counter
        self.global_step = 0

    def _build_network(self):
        """Optimal network architecture for Atari games"""
        if len(self.state_shape) == 3:
            return Sequential(
                [
                    Input(shape=self.state_shape),
                    Conv2D(32, (8, 8), strides=4, activation="relu", name="conv1"),
                    Conv2D(64, (4, 4), strides=2, activation="relu", name="conv2"),
                    Conv2D(64, (3, 3), strides=1, activation="relu", name="conv3"),
                    Flatten(),
                    Dense(512, activation="relu", name="fc1"),
                    Dense(self.action_size, activation="linear", name="q_values"),
                ]
            )
        elif len(self.state_shape) == 1 and self.state_shape[0] <= 64:  # Small discrete
            return Sequential(
                [
                    Input(shape=self.state_shape),
                    Dense(128, activation="relu"),
                    Dense(128, activation="relu"),
                    Dense(64, activation="relu"),
                    Dense(self.action_size, activation="linear"),
                ]
            )
        else:
            return Sequential(
                [
                    Input(shape=self.state_shape),
                    Dense(256, activation="relu"),
                    Dense(256, activation="relu"),
                    Dense(128, activation="relu"),
                    Dense(self.action_size, activation="linear"),
                ]
            )

    def append_new_experience(self, state, action, reward, next_state, done):
        self.memory.append(
            Experience(
                state.astype(np.float32),
                action,
                reward,
                next_state.astype(np.float32),
                done,
            )
        )

    def get_action(self, state, env):
        """Epsilon-greedy policy"""
        if np.random.rand() < self.epsilon:
            return env.action_space.sample()

        # if len(state.shape) == 1:
        state_input = np.expand_dims(state, axis=0)

        q_values = self.q_network(state_input, training=False)
        return np.argmax(q_values.numpy()[0])

    def predict_action(self, state):
        """Get action without exploration (for testing)"""
        state_input = np.expand_dims(state, axis=0)
        q_values = self.q_network(state_input, training=False)
        return int(np.argmax(q_values[0]))

    def preprocess_state(self, state):
        """Preprocess state for the network"""
        if len(self.state_shape) == 3:
            if len(state.shape) == 2:
                state = np.expand_dims(state, axis=-1)
            return state.astype(np.float32) / 255.0

        elif len(self.state_shape) == 1 and self.state_shape[0] <= 64:
            if isinstance(state, (int, np.integer)):
                one_hot = np.zeros(self.state_shape[0], dtype=np.float32)
                one_hot[state] = 1.0
                return one_hot
            return np.array(state, dtype=np.float32)
        else:
            return np.array(state, dtype=np.float32)

    def compute_loss(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        # Current Q-values
        current_q_values = self.q_network(states, training=True)
        current_q_values = tf.gather_nd(
            current_q_values,
            tf.stack(
                [tf.range(current_q_values.shape[0]), tf.cast(actions, tf.int32)],
                axis=1,
            ),
        )

        # Double DQN: Use main network to select actions, target network to evaluate
        next_actions = tf.argmax(self.q_network(next_states, training=False), axis=1)
        next_q_values = self.target_network(next_states, training=False)
        max_next_q_values = tf.gather(next_q_values, next_actions, batch_dims=1)

        # Compute target Q-values
        target_q_values = rewards + (1.0 - dones) * self.gamma * max_next_q_values

        target_q_values = tf.stop_gradient(target_q_values)

        return self.loss_fn(target_q_values, current_q_values)

    @tf.function
    def agent_learn(self, experiences):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(experiences=experiences)

        grads = tape.gradient(loss, self.q_network.trainable_variables)
        clipped_grads = [tf.clip_by_norm(grad, 1.0) for grad in grads]
        self.optimizer.apply_gradients(
            zip(clipped_grads, self.q_network.trainable_variables)
        )

        # Soft update target network
        for target_param, param in zip(
            self.target_network.trainable_variables,
            self.q_network.trainable_variables,
        ):
            target_param.assign(param * self.tau + target_param * (1.0 - self.tau))

    def start_training(self):
        return (
            len(self.memory) >= self.batch_size
            and self.global_step % self.update_frequency == 0
        )

    def update_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        return self.epsilon

    def get_experiences_replay(self):
        experiences = random.sample(self.memory, self.batch_size)

        states = tf.convert_to_tensor(
            np.array([e.state for e in experiences]), dtype=tf.float32
        )
        actions = tf.convert_to_tensor(
            np.array([e.action for e in experiences]), dtype=tf.int32
        )
        rewards = tf.convert_to_tensor(
            np.array([e.reward for e in experiences]), dtype=tf.float32
        )
        next_states = tf.convert_to_tensor(
            np.array([e.next_state for e in experiences]), dtype=tf.float32
        )
        dones = tf.convert_to_tensor(
            np.array([e.done for e in experiences]), dtype=tf.float32
        )

        return (states, actions, rewards, next_states, dones)

    def save(self, path):
        self.q_network.save(path)

    def load(self, path):
        self.q_network = tf.keras.models.load_model(path)
        self.target_network = tf.keras.models.clone_model(self.q_network)
        self.target_network.set_weights(self.q_network.get_weights())
