#!/usr/bin/env python3
"""
Training module for policy gradient
"""
import numpy as np
from policy_gradient import policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98, show_result=False):
    """
    Implements training using Monte-Carlo policy gradient

    Args:
        env: initial environment
        nb_episodes: number of episodes used for training
        alpha: learning rate (default 0.000045)
        gamma: discount factor (default 0.98)
        show_result: whether to render the environment every 1000 episodes

    Returns:
        scores: list of scores (sum of rewards) for each episode
    """
    # Initialize weight matrix
    weight = np.random.rand(env.observation_space.shape[0], env.action_space.n)

    scores = []

    for episode in range(nb_episodes):
        # Reset environment
        state = env.reset()
        if isinstance(state, tuple):  # gym >= 0.26 returns (state, info)
            state = state[0]
        state = state.reshape(1, -1)

        # Store gradients and rewards for this episode
        episode_gradients = []
        episode_rewards = []

        done = False
        score = 0

        # Play one episode
        while not done:
            # Get action and gradient
            action, gradient = policy_gradient(state, weight)

            # Take action in environment
            result = env.step(action)
            if len(result) == 4:  # gym < 0.26
                next_state, reward, done, _ = result
            else:  # gym >= 0.26
                next_state, reward, terminated, truncated, _ = result
                done = terminated or truncated

            # Store gradient and reward
            episode_gradients.append(gradient)
            episode_rewards.append(reward)
            score += reward

            # Move to next state
            state = next_state.reshape(1, -1)

        # Update weights using accumulated gradients and rewards
        # Compute discounted returns
        G = 0
        for t in range(len(episode_rewards) - 1, -1, -1):
            G = episode_rewards[t] + gamma * G
            weight = weight + alpha * episode_gradients[t] * G

        scores.append(score)

        # Print progress
        print(f"Episode: {episode} Score: {score}", end="\r", flush=True)

        # Render every 1000 episodes if show_result is True
        if show_result and (episode + 1) % 1000 == 0:
            env.render()

    print()  # Clear the last line
    return scores
