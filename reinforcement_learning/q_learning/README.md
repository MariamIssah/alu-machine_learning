# Q-Learning Project

This project implements Q-learning, a fundamental reinforcement learning algorithm, using OpenAI's FrozenLake environment.

## Description

Q-learning is a model-free reinforcement learning algorithm that learns the value of an action in a particular state. The project includes:

1. **0-load_env.py**: Load the FrozenLake environment from OpenAI Gym
2. **1-q_init.py**: Initialize the Q-table for the reinforcement learning agent
3. **2-epsilon_greedy.py**: Implement the epsilon-greedy exploration strategy
4. **3-q_learning.py**: Train the agent using Q-learning algorithm
5. **4-play.py**: Play the game using the trained agent

## Concepts

- **Markov Decision Process (MDP)**: A mathematical framework for decision-making in environments with states, actions, and rewards
- **State**: A specific configuration in the environment
- **Action**: A choice the agent can make at each state
- **Reward**: Feedback signal returned by the environment
- **Q-Value**: The expected cumulative reward for taking an action in a state
- **Epsilon-Greedy**: Exploration strategy that chooses random actions with probability epsilon
- **Bellman Equation**: Fundamental equation for updating Q-values
- **Discount Factor (gamma)**: Determines the importance of future rewards

## Requirements

- Python 3.5+
- NumPy 1.15
- OpenAI Gym 0.7

## Installation

```bash
pip install --user gym
```

## Usage

Run the main file to train and test the Q-learning agent:

```bash
python3 3-q_learning.py
```

## Project Author

Alexa Orrico, Software Engineer at Holberton School

## Task Descriptions

### Task 0: Load the Environment
Load the FrozenLakeEnv with custom or pre-made maps, and control ice slipperiness.

### Task 1: Initialize Q-table
Create a Q-table initialized with zeros for all state-action pairs.

### Task 2: Epsilon Greedy
Implement the epsilon-greedy algorithm for balancing exploration and exploitation.

### Task 3: Q-learning
Train the agent using the Q-learning algorithm with Bellman equation updates.

### Task 4: Play
Display the agent's gameplay using the learned policy through exploitation.
