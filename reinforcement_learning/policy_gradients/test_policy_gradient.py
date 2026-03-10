#!/usr/bin/env python3
"""Test the policy_gradient function"""
import gym
import numpy as np
from policy_gradient import policy_gradient

env = gym.make('CartPole-v1')
np.random.seed(1)

weight = np.random.rand(4, 2)
state = env.reset()[None,:]
print("Weight:")
print(weight)
print("\nState:")
print(state)

action, grad = policy_gradient(state, weight)
print("\nAction:", action)
print("\nGradient:")
print(grad)

env.close()
