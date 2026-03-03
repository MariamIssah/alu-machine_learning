#!/usr/bin/env python3
"""
Module for initializing the Q-table.
"""
import numpy as np


def q_init(env):
    """
    Initialize the Q-table with zeros.
    
    Args:
        env: The FrozenLakeEnv instance
    
    Returns:
        The Q-table as a numpy.ndarray of zeros with shape
        (number_of_states, number_of_actions)
    """
    return np.zeros((env.observation_space.n, env.action_space.n))
