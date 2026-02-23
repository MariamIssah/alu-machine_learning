#!/usr/bin/env python3
"""
SARSA(λ) algorithm with eligibility traces for control.
"""

import numpy as np


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1,
                  gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    SARSA(λ) with eligibility traces and epsilon-greedy policy.

    env: openAI environment instance
    Q: numpy.ndarray of shape (s, a) containing the Q table
    lambtha: eligibility trace factor
    episodes: total number of episodes to train over
    max_steps: maximum number of steps per episode
    alpha: learning rate
    gamma: discount rate
    epsilon: initial threshold for epsilon greedy
    min_epsilon: minimum value for epsilon
    epsilon_decay: decay rate for epsilon between episodes

    Returns: Q, the updated Q table.
    """
    nS, nA = Q.shape
    Q = np.copy(Q)
    for _ in range(episodes):
        state = env.reset()
        action = np.argmax(Q[state]) if np.random.uniform() >= epsilon else \
            np.random.randint(nA)
        E = np.zeros((nS, nA))
        for _ in range(max_steps):
            next_state, reward, done, _ = env.step(action)
            if done:
                next_action = None
                target = reward
            else:
                if np.random.uniform() >= epsilon:
                    next_action = np.argmax(Q[next_state])
                else:
                    next_action = np.random.randint(nA)
                target = reward + gamma * Q[next_state, next_action]
            delta = target - Q[state, action]
            E *= gamma * lambtha
            E[state, action] += 1.0
            Q += alpha * delta * E
            if done:
                break
            state = next_state
            action = next_action
        epsilon = max(min_epsilon, epsilon - epsilon_decay)
    return Q
