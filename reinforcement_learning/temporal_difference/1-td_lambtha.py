#!/usr/bin/env python3
"""
TD(λ) algorithm with eligibility traces for value estimation.
"""

import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000, max_steps=100,
               alpha=0.1, gamma=0.99):
    """
    TD(λ) algorithm with eligibility traces.

    env: openAI environment instance
    V: numpy.ndarray of shape (s,) containing the value estimate
    policy: function that takes a state and returns the next action
    lambtha: eligibility trace factor
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
        e = np.zeros(nS)
        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            if done:
                delta = reward - V[state]
            else:
                delta = reward + gamma * V[next_state] - V[state]
            e *= gamma * lambtha
            e[state] += 1.0
            V += alpha * delta * e
            if done:
                break
            state = next_state
    return V
