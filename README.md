# 🎮 Deep Reinforcement Learning — Double DQN Agents

![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow&logoColor=white)
![Gymnasium](https://img.shields.io/badge/Gymnasium-Farama-0081A5)
![SB3](https://img.shields.io/badge/Stable--Baselines3-DQN-blueviolet)

A **Double Deep Q-Network (DDQN) agent implemented from scratch** in TensorFlow and trained on several classic control and Atari environments, plus a large-scale Atari **Breakout** agent trained with Stable-Baselines3.

## ✨ Highlights

- **From-scratch DDQN** (`DDQN/dqn_agent.py`)
  - Experience replay buffer (`deque`, 10k transitions) and mini-batch updates
  - **Double DQN target**: online network *selects* the action, target network *evaluates* it — reduces Q-value overestimation
  - **Soft target updates** (Polyak averaging, τ = 0.01)
  - ε-greedy exploration with exponential decay
  - Huber loss + Adam, custom `tf.GradientTape` training step
  - Dense network for vector states, **CNN** (Conv2D + MaxPooling) for image states
- **One training script, many environments** (`DDQN/train_dqn_agent.py`) — CartPole, LunarLander, MountainCar, FrozenLake 4×4 / 8×8, Breakout
- **Evaluation** over 100 test episodes with success-rate reporting
- **Atari Breakout** (`AtariEnv/`) — Stable-Baselines3 DQN with `AtariWrapper`, frame preprocessing, **50M timesteps**, TensorBoard logging

## 📁 Structure

```
DDQN/
├── dqn_agent.py          # DQNAgent: replay buffer, Double-DQN loss, soft updates
├── train_dqn_agent.py    # train + test on any supported env
├── run_dqn_agent.py      # watch a trained agent play
└── *.keras               # trained models (CartPole, MountainCar, FrozenLake…)
AtariEnv/
├── dqn_atari_train.py    # SB3 DQN on ALE/Breakout-v5
└── test_dqn_atari.py     # render the trained agent
```

## 🚀 Run it

```bash
pip install tensorflow gymnasium[box2d,atari] ale-py stable-baselines3 matplotlib
cd DDQN
python train_dqn_agent.py      # choose env via Env enum in __main__
python run_dqn_agent.py        # watch the trained agent
```

## 📚 Concepts Practised

Markov Decision Processes · Bellman equation · Q-learning · function approximation · experience replay · target networks · exploration vs. exploitation

---
👤 **Muath Omarieh** — CS student @ University of Jordan · [GitHub](https://github.com/muathomarieh) · [LinkedIn](https://www.linkedin.com/in/muath-omarieh-550679297/)
