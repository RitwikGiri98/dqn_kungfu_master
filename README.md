# 🧠 Deep Q-Learning Agent for Atari: Kung Fu Master

**Course:** Reinforcement Learning and AI Agents  
**Assignment:** LLM Agents & Deep Q-Learning with Atari Games  
**Prepared By:** Ritwik Giri  
**Submission Date:** November 9, 2025

* * *

## 🎯 Project Overview

This project implements a **Deep Q-Learning (DQN)** agent in the **Atari Kung Fu Master** environment using **OpenAI Gymnasium**.  
The goal is to train an agent to maximize in-game score by learning optimal policies through **value-based reinforcement learning**.

The assignment explores how agents learn via **interaction and feedback**, emphasizing:

*   Environment exploration–exploitation balance
*   Bellman equation parameter sensitivity
*   Reward shaping and target network stability
*   Theoretical connections to **LLM-based reinforcement** (RLHF)

* * *

## ⚙️ Environment Setup

### 1️⃣ Create and Activate Virtual Environment

    # Create virtual environment
    python3 -m venv .venv
    
    # Activate environment (Linux/Mac)
    source .venv/bin/activate
    
    # Activate environment (Windows)
    # .venv\Scripts\activate
    

### 2️⃣ Install Dependencies

    # Upgrade pip
    pip install --upgrade pip
    
    # Install required packages
    pip install "gymnasium[atari,accept-rom-license]" ale-py
    pip install torch torchvision numpy opencv-python matplotlib pandas tqdm
    

### 3️⃣ Verify Environment

    import gymnasium as gym
    
    # Create environment
    env = gym.make("ALE/KungFuMaster-v5")
    obs, info = env.reset()
    
    # Check observation and action spaces
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    

**Expected Output:**

    Observation shape: (210, 160, 3)
    Action space: Discrete(14)
    

* * *

## 📁 Project Structure

    dqn_kungfu_master/
    │
    ├── src/
    │   ├── train_dqn.py           # Training loop with hyperparameter tuning
    │   ├── evaluate_dqn.py        # Evaluation + video recording
    │   ├── replay_buffer.py       # Experience replay implementation
    │   ├── q_network.py           # PyTorch CNN-based Q-network
    │   ├── utils.py               # Logging, metrics, and plotting helpers
    │   ├── wrappers.py            # Preprocessing: grayscale, frame stack
    │
    ├── notebooks/
    │   └── DeepQLearning_KungFuMaster.ipynb  # Analysis + plots + Q&A
    │
    ├── outputs/
    │   ├── metrics_baseline.csv
    │   ├── metrics_gamma095_lr1e4.csv
    │   ├── metrics_epsdecay25k.csv
    │   ├── metrics_policy_softmax.csv
    │   ├── metrics_reward_clipped.csv
    │   ├── metrics_tsync_10k.csv
    │   ├── trained_model.pth
    │   ├── demo_trained.mp4
    │   ├── demo_baseline.mp4
    │   ├── comparison_baseline_vs_trained.mp4
    │
    ├── requirements.txt
    └── README.md
    
    

* * *

## 🚀 Quick Start

### 1️⃣ Train the Agent

    python -m src.train_dqn --episodes 100 --gamma 0.95 --lr 1e-4 --target_sync 10000
    

### 2️⃣ Evaluate and Record Gameplay

    python -m src.evaluate_dqn --weights outputs/trained_model.pth --episodes 5 --record outputs/demo_trained.mp4
    

* * *

## 📊 Key Experiments

| Experiment | Configuration | Result |
| --- | --- | --- |
| Bellman parameters | γ=0.95, lr=1e−4 | Stable learning |
| Exploration tuning | Faster ε-decay | Improved convergence |
| Reward clipping | [-1, 1] range | Reduced variance |
| Target sync | 10k steps | Balanced update frequency |

* * *

## 🧠 Project Highlights

*   Built using **PyTorch + Gymnasium + ALE-py**
*   End-to-end DQN training and evaluation pipeline
*   Includes gameplay recordings and experiment logs
*   Links classical RL to LLM-based reinforcement concepts (RLHF)

* * *

## ⚖️ Code Attribution

### ✅ Original Code

*   **`train_dqn.py`** — Main training logic, epsilon scheduling, reward clipping, and target sync
*   **`evaluate_dqn.py`** — Evaluation, greedy policy toggle, and MP4 recording
*   **`replay_buffer.py`** — Replay memory built from scratch
*   **`q_network.py`** — PyTorch CNN architecture tailored for Kung Fu Master
*   **`utils.py`** — Metrics, plotting, CSV logging
*   **Notebook** — All parameter sweeps, plots, and analysis

### 🧩 Adapted & Referenced Sources

*   [OpenAI Gymnasium](https://gymnasium.farama.org/) (environment and wrapper templates)
*   [PyTorch DQN Tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html) (initial structure for Q-network updates)
*   [ALE-py](https://github.com/mgbellemare/Arcade-Learning-Environment) (Arcade Learning Environment backend)

* * *

## 🏁 Conclusion

This project demonstrates a successful end-to-end implementation of a Deep Q-Learning agent for Atari's Kung Fu Master. Through iterative experimentation and parameter optimization, the agent displayed measurable learning stability consistent with core RL theory. The study also bridges the conceptual gap between traditional reinforcement learning and modern LLM optimization, reinforcing how reward-driven learning generalizes across AI paradigms.


* * *

## 👏 Acknowledgments

Special thanks to:

*   Professor & TA team for providing clear assignment rubrics and evaluation structure
*   OpenAI Gymnasium & PyTorch for open educational resources

* * *

## 📄 License

This project was developed as part of academic coursework for educational purposes.

* * *

<p align="center"> <strong>🎮 Built with Deep Reinforcement Learning 🤖</strong> </p>
