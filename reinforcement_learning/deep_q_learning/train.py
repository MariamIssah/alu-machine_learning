#!/usr/bin/env python3
"""
Train a DQN agent to play Atari Breakout using Keras-RL.

This script builds a convolutional policy network, wraps the environment
with standard Atari preprocessing (84x84 grayscale + frame stacking),
trains a DQNAgent using SequentialMemory and EpsGreedyQPolicy, and saves
weights to `policy.h5`.

Notes:
- Requires: gym (with Atari), keras, keras-rl
- Training can be long; adjust `nb_steps` for quicker runs.
"""
import os
import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, Permute
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory


def build_model(nb_actions, input_shape=(4, 84, 84)):
    """Build a convolutional model compatible with framed Atari inputs.

    Args:
        nb_actions: number of discrete actions from the env
        input_shape: (window_length, height, width)
    Returns:
        Keras model
    """
    model = Sequential()
    # input_shape is (4, 84, 84) -> permute to (84, 84, 4)
    model.add(Permute((2, 3, 1), input_shape=input_shape))
    model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu'))
    model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(nb_actions, activation='linear'))
    return model


def make_env():
    """Create an Atari Breakout env with standard preprocessing and frame stack.

    Uses gym wrappers available in gym (AtariPreprocessing and FrameStack).
    If those wrappers are unavailable, a plain `Breakout-v0` env is returned.
    """
    try:
        env = gym.make('BreakoutNoFrameskip-v4')
        # AtariPreprocessing and FrameStack are available in gym>=0.15
        from gym.wrappers import AtariPreprocessing, FrameStack
        env = AtariPreprocessing(env, screen_size=84, grayscale_obs=True, scale_obs=False)
        env = FrameStack(env, num_stack=4)
    except Exception:
        # Fallback to basic env (may not be optimal for DQN)
        env = gym.make('Breakout-v0')
    return env


def main():
    env = make_env()
    np.random.seed(123)
    env.seed(123)

    nb_actions = env.action_space.n

    # Build model assuming stacked frames -> input shape (4,84,84)
    model = build_model(nb_actions)
    print(model.summary())

    # Configure memory and policy
    memory = SequentialMemory(limit=1000000, window_length=4)
    policy = EpsGreedyQPolicy()

    # Create DQN agent
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory,
                   nb_steps_warmup=50000, target_model_update=10000,
                   policy=policy, gamma=0.99, train_interval=4, delta_clip=1.0)
    dqn.compile(Adam(lr=1e-4), metrics=['mae'])

    # Train the agent. nb_steps can be increased for better performance.
    nb_steps = int(os.environ.get('DQN_TRAIN_STEPS', 50000))
    dqn.fit(env, nb_steps=nb_steps, visualize=False, verbose=2)

    # Save the weights
    dqn.save_weights('policy.h5', overwrite=True)
    print('Saved policy.h5')


if __name__ == '__main__':
    main()
