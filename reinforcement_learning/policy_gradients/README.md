# Policy Gradients

This project implements Policy Gradient methods in Reinforcement Learning, specifically the Monte-Carlo policy gradient algorithm (REINFORCE).

## Learning Objectives

- What is a policy in reinforcement learning?
- How to calculate policy gradients?
- Implementing and using Monte-Carlo policy gradient (REINFORCE) algorithm

## Files

- `policy_gradient.py`: Core policy and policy gradient functions
  - `policy(matrix, weight)`: Computes action probabilities using softmax
  - `policy_gradient(state, weight)`: Computes policy gradients using Monte-Carlo method

- `train.py`: Training implementation
  - `train(env, nb_episodes, alpha, gamma, show_result)`: Trains the policy using REINFORCE algorithm

## Implementation Details

### Policy Function

The policy function computes action probabilities using the softmax function applied to the dot product of state and weight:

```
π(a|s,w) = softmax(s @ w)
```

### Policy Gradient

The Monte-Carlo policy gradient (REINFORCE) uses:

```
∇w log π(a|s,w) = s^T @ (one_hot(a) - π(a|s,w))
```

### Training

The training uses the policy gradient to update weights proportional to the returns:

```
w ← w + α * ∇gradient * G_t
```

where G_t is the discounted return from time t.

## Testing

Test files are provided in the main project structure:

- `0-main.py`: Tests the policy function
- `1-main.py`: Tests the policy gradient function
- `2-main.py`: Tests training and plots the learning curve
- `3-main.py`: Tests training with visualization

## Requirements

- Python 3.5+
- NumPy >= 1.15
- Gym >= 0.7
- Matplotlib (for visualization)

## Notes

- The training may take some time for 10,000 episodes
- Results can vary due to random initialization and sampling
- Adjusting `alpha` (learning rate) and `gamma` (discount factor) can improve convergence
