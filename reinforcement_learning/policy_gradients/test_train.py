#!/usr/bin/env python3
"""Test the train function"""
import gym
import numpy as np
from train import train

env = gym.make('CartPole-v1')
np.random.seed(42)

print("Testing train function with 10 episodes...")
scores = train(env, 10)

print(f"\nNumber of scores returned: {len(scores)}")
print(f"Score values: {scores}")
print(f"All positive: {all(s > 0 for s in scores)}")

env.close()
print("\nTrain function test completed successfully!")
