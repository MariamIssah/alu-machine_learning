#!/usr/bin/env python3
"""
Maximization step in the EM algorithm for a GMM.
"""

import numpy as np


def maximization(X, g):
    """
    Calculate the maximization step in the EM algorithm for a GMM.

    Args:
        X: numpy.ndarray of shape (n, d) - the data set
        g: numpy.ndarray of shape (k, n) - posterior probabilities

    Returns:
        (pi, m, S) or (None, None, None) on failure.
        pi: shape (k,) updated priors
        m: shape (k, d) updated centroid means
        S: shape (k, d, d) updated covariance matrices
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None
    if not isinstance(g, np.ndarray) or g.ndim != 2:
        return None, None, None
    n, d = X.shape
    k = g.shape[0]
    if g.shape[1] != n:
        return None, None, None
    if k == 0:
        return None, None, None
    if np.any(g < 0):
        return None, None, None
    n_soft = np.sum(g, axis=1, keepdims=True)
    if np.any(n_soft <= 0):
        return None, None, None
    pi = (np.sum(g, axis=1) / n).reshape(k)
    m = (g @ X) / n_soft
    S = np.zeros((k, d, d))
    for j in range(k):
        diff = X - m[j]
        weighted_diff = diff * g[j].reshape(-1, 1)
        S[j] = (weighted_diff.T @ diff) / n_soft[j, 0]
    return pi, m, S
