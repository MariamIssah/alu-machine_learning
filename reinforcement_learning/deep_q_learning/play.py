#!/usr/bin/env python3
"""
Load a trained DQN policy and play Atari Breakout using GreedyQPolicy.

This script re-constructs the same model and agent topology used in
`train.py`, loads `policy.h5` weights, and runs a small number of
evaluation episodes with a greedy policy while rendering the env.
"""
import os
import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, Permute
from rl.agents.dqn import DQNAgent
from rl.policy import GreedyQPolicy
from rl.memory import SequentialMemory


def build_model(nb_actions, input_shape=(4, 84, 84)):
    model = Sequential()
    model.add(Permute((2, 3, 1), input_shape=input_shape))
    model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu'))
    model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(nb_actions, activation='linear'))
    return model


def make_env():
    try:
        env = gym.make('BreakoutNoFrameskip-v4')
        from gym.wrappers import AtariPreprocessing, FrameStack
        env = AtariPreprocessing(env, screen_size=84, grayscale_obs=True, scale_obs=False)
        env = FrameStack(env, num_stack=4)
    except Exception:
        env = gym.make('Breakout-v0')
    return env


def main():
    env = make_env()
    nb_actions = env.action_space.n

    model = build_model(nb_actions)

    memory = SequentialMemory(limit=1000000, window_length=4)
    policy = GreedyQPolicy()

    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory,
                   nb_steps_warmup=0, target_model_update=10000,
                   policy=policy, gamma=0.99, train_interval=4, delta_clip=1.0)
    dqn.compile('adam', metrics=['mae'])

    if not os.path.exists('policy.h5'):
        raise SystemExit('policy.h5 not found - run train.py first')

    dqn.load_weights('policy.h5')

    # Run evaluation episodes with rendering
    dqn.test(env, nb_episodes=5, visualize=True)


if __name__ == '__main__':
    main()
