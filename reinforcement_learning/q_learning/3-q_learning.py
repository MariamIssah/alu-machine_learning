#!/usr/bin/env python3
"""
Module for implementing the Q-learning algorithm.
"""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    Use epsilon-greedy to determine the next action.
    
    Args:
        Q: A numpy.ndarray containing the q-table
        state: The current state
        epsilon: The epsilon to use for the calculation
    
    Returns:
        The next action index
    """
    p = np.random.uniform(0, 1)
    if p > epsilon:
        action = np.argmax(Q[state, :])
    else:
        action = np.random.randint(0, Q.shape[1])
    return action


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99,
          epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Perform Q-learning training.
    
    Args:
        env: The FrozenLakeEnv instance
        Q: A numpy.ndarray containing the Q-table
        episodes: The total number of episodes to train over
        max_steps: The maximum number of steps per episode
        alpha: The learning rate
        gamma: The discount rate
        epsilon: The initial threshold for epsilon greedy
        min_epsilon: The minimum value that epsilon should decay to
        epsilon_decay: The decay rate for updating epsilon between episodes
    
    Returns:
        Q: The updated Q-table
        total_rewards: A list containing the rewards per episode
    """
    total_rewards = []
    
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            # Use epsilon-greedy to select an action
            action = epsilon_greedy(Q, state, epsilon)
            
            # Take the action in the environment
            next_state, reward, done, info = env.step(action)
            
            # Change reward for holes to -1
            if reward == 0 and done and next_state != state:
                reward = -1
            
            # Update Q-table using Bellman equation
            Q[state, action] = Q[state, action] + alpha * (
                reward + gamma * np.max(Q[next_state, :]) - Q[state, action]
            )
            
            episode_reward += reward
            state = next_state
            
            # Break if episode is done
            if done:
                break
        
        # Decay epsilon
        epsilon = max(min_epsilon, epsilon - epsilon_decay)
        
        # Track rewards
        total_rewards.append(episode_reward)
    
    return Q, total_rewards
