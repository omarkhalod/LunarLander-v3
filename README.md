# Deep Q-Learning LunarLander-v3

This repository implements Deep Q-Learning (DQN) on the OpenAI Gymnasium `LunarLander-v3` environment with three replay strategies:

1. **Linear Replay (Uniform)**: Standard experience replay drawing uniformly from past transitions.
2. **Prioritized Replay (PER)**: Samples high‑TD‑error transitions more frequently.
3. **Mixed Replay**: Blends PER and uniform replay by a tunable ratio (`ETA`).

---

## Features

* DQN architecture with two hidden layers (64 units each).
* Epsilon-greedy exploration with configurable decay.
* Soft target-network updates.
* Gradient clipping to stabilize training.
* Three replay buffer implementations (`linear_replay.py`, `priority_replay.py`, `MixedReplay.py`).
* Training curves comparison via `plot_comparison()`.
* Video recording of trained agents.

---

## Installation

1. **Clone the repository**

   ```bash
   git clone <your-repo-url>
   cd <repo-folder>
   ```

2. **Create and activate a virtual environment** (recommended)

   ```bash
   python -m venv venv
   source venv/bin/activate   # on macOS/Linux
   venv\\Scripts\\activate    # on Windows
   ```

3. **Install required Python packages**

   ```bash
   pip install --upgrade pip
   pip install \
     gymnasium \
     numpy \
     torch \
     matplotlib \
     imageio \
     pyglet      # for rendering in Gymnasium
   ```

    Installing Gymnasium:

```bash
    pip install gymnasium
    pip install "gymnasium[atari, accept-rom-license]"
    pip install gymnasium[box2d]
    ```

    
4. **Local modules**

   * Ensure the files `Visualization.py`, `priority_replay.py`, `linear_replay.py`, and `MixedReplay.py` are in the project root or in your Python path.

---

## Usage

1. **Run training and evaluation**

   ```bash
   python DQN_Agent.py
   ```

   This script will:

   * Train three agents (PER, Linear, Mixed) sequentially.
   * Save model checkpoints under `Models/` when the environment is solved.
   * Display and save videos of each agent’s performance.
   * Plot a comparison of learning curves.

2. **Configure hyperparameters**

   * At the top of `DQN_Agent.py`, adjust constants like `LEARNING_RATE`, `ETA`, `EPSILON_DECAY`, etc.

---

## File Structure

```
├── DQN_Agent.py         # Main training and evaluation script
├── MixedReplay.py       # Mixed replay buffer implementation
├── linear_replay.py     # Uniform replay buffer implementation
├── priority_replay.py   # Prioritized replay buffer implementation
├── Visualization.py     # Plotting and video utilities
├── Models/              # Saved model checkpoints
└── README.md            # This file
```

---

## Summary

This project demonstrates how different replay strategies affect DQN performance on LunarLander-v3. You can easily switch between linear, prioritized, or mixed replay by adjusting the `ETA` parameter before training. The included plotting utilities help visualize and compare the learning curves of each approach.

Enjoy experimenting with replay methods and hyperparameter tuning!
