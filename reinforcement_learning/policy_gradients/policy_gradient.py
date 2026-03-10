#!/usr/bin/env python3
"""
Module for policy gradient implementation
"""
import numpy as np


def policy(matrix, weight):
    """
    Computes the policy with a weight matrix using softmax

    Args:
        matrix: state matrix of shape (1, n) or (m, n)
        weight: weight matrix of shape (n, p)

    Returns:
        policy: probability distribution over actions, shape (m, p) or (1, p)
    """
    z = np.matmul(matrix, weight)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def policy_gradient(state, weight):
    """
    Computes the Monte-Carlo policy gradient

    Args:
        state: current observation of the environment, shape (1, n)
        weight: weight matrix, shape (n, p)

    Returns:
        action: sampled action (scalar)
        gradient: policy gradient, shape (n, p)
    """
    # Compute policy
    p = policy(state, weight)

    # Sample an action from the policy
    action = np.random.choice(p.shape[1], p=p.flatten())

    # Compute gradient: state.T @ (one_hot(action) - policy)
    one_hot = np.zeros(p.shape[1])
    one_hot[action] = 1

    gradient = np.matmul(state.T, one_hot.reshape(1, -1) - p)

    return action, gradient
