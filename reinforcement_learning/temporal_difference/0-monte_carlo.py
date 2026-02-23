#!/usr/bin/env python3
"""
Monte Carlo policy evaluation for reinforcement learning.
"""

import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99):
    """
    Monte Carlo algorithm for value estimation.

    env: openAI environment instance
    V: numpy.ndarray of shape (s,) containing the value estimate
    policy: function that takes a state and returns the next action
    episodes: total number of episodes to train over
    max_steps: maximum number of steps per episode
    alpha: learning rate
    gamma: discount rate

    Returns: V, the updated value estimate.
    """
    nS = V.shape[0]
    V = np.copy(V)
    for _ in range(episodes):
        state = env.reset()
        states = [state]
        rewards = []
        for _ in range(max_steps):
            action = policy(state)
            state, reward, done, _ = env.step(action)
            rewards.append(reward)
            states.append(state)
            if done:
                break
        states = np.array(states)
        rewards = np.array(rewards)
        T = len(rewards)
        G = np.zeros(T)
        G[T - 1] = rewards[T - 1]
        for t in range(T - 2, -1, -1):
            G[t] = rewards[t] + gamma * G[t + 1]
        for t in range(T):
            s = states[t]
            V[s] += alpha * (G[t] - V[s])
    return V
