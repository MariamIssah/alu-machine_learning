#!/usr/bin/env python3
"""Test the policy function"""
import numpy as np
from policy_gradient import policy

# Test case 1
weight = np.ndarray((4, 2), buffer=np.array([
    [4.17022005e-01, 7.20324493e-01], 
    [1.14374817e-04, 3.02332573e-01], 
    [1.46755891e-01, 9.23385948e-02], 
    [1.86260211e-01, 3.45560727e-01]
    ]))
state = np.ndarray((1, 4), buffer=np.array([
    [-0.04428214,  0.01636746,  0.01196594, -0.03095031]
    ]))

res = policy(state, weight)
print('Test 1 Result:', res)
print('Test 1 Expected: [[0.50351642 0.49648358]]')
print('Match:', np.allclose(res, [[0.50351642, 0.49648358]], atol=1e-6))

# Test case 2
weight2 = np.ndarray((4, 2), buffer=np.array([
    [2.1022005e-01, 3.20324493e-01], 
    [2.14374817e-04, 1.02332573e-01], 
    [4.4675891e-01, 3.23385948e-02], 
    [0.8626011e-01, 1.45560727e-01]
    ]))
state2 = np.ndarray((1, 4), buffer=np.array([
    [0.0428214,  0.11636746,  1.01196594, 0.03095031]
    ]))

res2 = policy(state2, weight2)
print('\nTest 2 Result:', res2)
print('Test 2 Expected: [[0.59891488 0.40108512]]')
print('Match:', np.allclose(res2, [[0.59891488, 0.40108512]], atol=1e-6))
