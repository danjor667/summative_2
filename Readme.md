# DQN Agent for Atari Breakout

This project implements a Deep Q-Network (DQN) agent to play the Atari game **Breakout** using Stable Baselines3 and Gymnasium. The agent is trained using image-based inputs and reinforcement learning techniques, then evaluated by running it in the game environment.

## Project Structure

```

summative\_2/
├── train.py                # Trains the agent and saves the model
├── play.py                 # Loads and plays the trained agent
├── utils/
│   ├── hyperparameter\_tuning.py
│   └── policy\_comparison.py
├── models/                 # Saved models
├── logs/                   # Training logs
├── videos/                 # Optional: recorded gameplay
└── README.md

````

## How to Run

### 1.Train the Agent

```bash
python train.py
````

### 2. Watch the Agent Play

```bash
python play.py
```
---

##  Video Demonstration

Watch the trained agent in action:

**\[Insert Your YouTube Link Here]**

---

##  Hyperparameter Tuning Results

| Set | Learning Rate | Gamma | Batch Size | Epsilon Start | Epsilon End | Epsilon Decay | Noted Behavior                              |
| --- | ------------- | ----- | ---------- | ------------- | ----------- | ------------- | ------------------------------------------- |
| 1   | 0.0001        | 0.99  | 32         | 1.0           | 0.05        | 0.1           | Stable reward increase, good convergence    |
| 2   | 0.0005        | 0.98  | 64         | 1.0           | 0.1         | 0.2           | Faster learning, slightly unstable          |
| 3   | 0.0001        | 0.99  | 16         | 1.0           | 0.01        | 0.05          | Slower learning, better long-term stability |

**Selected Set:** Set 1
It produced consistent reward improvements and steady performance during evaluation.

---

##  Group Collaboration and Contributions

| Member Name | Contribution                                                                      |
| ----------- | --------------------------------------------------------------------------------- |
| Jordan Steve Lopez Nguepi | Implemented and ran `train.py`, logged training results, tuned hyperparameters    |
| Jules Gatete | Developed `play.py`, tested agent evaluation, recorded and prepared demo video    |
| Kevin Kenny Mugisha | Ran tuning experiments (`hyperparameter_tuning.py`), compared CNN vs MLP policies |

Work was divided fairly, with code committed through GitHub and team coordination maintained through regular communication.

---

## Requirements

Install all required libraries using:

```bash
pip install stable-baselines3[extra] gymnasium[atari] ale-py autorom opencv-python
AutoROM --accept-license
``
