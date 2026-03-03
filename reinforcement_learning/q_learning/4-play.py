#!/usr/bin/env python3
"""
Module for playing the game using a trained Q-learning agent.
"""
import numpy as np


def play(env, Q, max_steps=100):
    """
    Have the trained agent play an episode.
    
    Args:
        env: The FrozenLakeEnv instance
        Q: A numpy.ndarray containing the Q-table
        max_steps: The maximum number of steps in the episode
    
    Returns:
        The total rewards for the episode
    """
    state = env.reset()
    total_reward = 0
    
    # Display initial board state
    print()
    print(env.render(mode='ansi'))
    
    for step in range(max_steps):
        # Always exploit (pick the best action)
        action = np.argmax(Q[state, :])
        
        # Take the action
        next_state, reward, done, info = env.step(action)
        
        total_reward += reward
        state = next_state
        
        # Display board state after action
        print("  (" + ["Left", "Down", "Right", "Up"][action] + ")")
        print(env.render(mode='ansi'))
        
        # Break if episode is done
        if done:
            break
    
    return total_reward
